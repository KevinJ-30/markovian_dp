from typing import Callable, Iterator, Optional
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import ModuleList, Parameter
from core.nn.mlp import MLP
from core.nn.jk import JumpingKnowledge
from torch_geometric.data import Data
from core.modules.base import Metrics, Phase, TrainableModule


class ProgressiveModule(TrainableModule):
    def __init__(self, *,
                 num_classes: int,
                 num_stages: int,
                 hidden_dim: int = 16,  
                 base_layers: int = 2, 
                 head_layers: int = 1, 
                 normalize: bool = True,
                 jk_mode: str = 'cat',
                 dropout: float = 0.0, 
                 activation_fn: Callable[[Tensor], Tensor] = torch.relu_, 
                 batch_norm: bool = True,
                 layerwise: bool = False,
                 regression: bool = False,
                 **kwargs,
                 ):

        super().__init__(**kwargs)

        self.num_classes = num_classes
        self.regression = regression
        self.num_stages = num_stages
        self.hidden_dim = hidden_dim
        self.base_layers = base_layers
        self.head_layers = head_layers
        self.normalize = normalize
        self.jk_mode = jk_mode
        self.dropout = dropout
        self.activation_fn = activation_fn
        self.batch_norm = batch_norm
        self.layerwise = layerwise

        self.current_stage = 0

        self.base = ModuleList(
            MLP(
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                num_layers=base_layers,
                activation_fn=activation_fn,
                dropout=dropout,
                batch_norm=batch_norm,
                plain_last=False,
            ) for _ in range(num_stages)
        )

        self.jk = ModuleList(
            JumpingKnowledge(
                mode=jk_mode,
                hidden_dim=hidden_dim,
                channels=hidden_dim,
                num_layers=2,
                num_heads=2
            ) for _ in range(num_stages)
        )

        self.head = ModuleList(
            MLP(
                hidden_dim=hidden_dim,
                output_dim=num_classes,
                num_layers=head_layers,
                dropout=dropout,
                activation_fn=activation_fn,
                batch_norm=batch_norm,
                plain_last=True,
            ) for _ in range(num_stages)
        )

    @property
    def regression(self) -> bool:
        return self._regression

    @regression.setter
    def regression(self, value: bool) -> None:
        if value and self.num_classes != 1:
            raise ValueError('Regression requires a scalar head (num_classes=1)')
        self._regression = value

    def set_stage(self, stage: int):
        self.current_stage = stage
        if self.layerwise:
            # freeze previous layers
            for i in range(self.current_stage):
                for param in self.base[i].parameters():
                    param.requires_grad = False
                for param in self.jk[i].parameters():
                    param.requires_grad = False
                for param in self.head[i].parameters():
                    param.requires_grad = False

    def forward(self, xs: list[Tensor]) -> tuple[Tensor, Tensor]:
        """forward propagation

        Args:
            xs (list[Tensor]): list of aggregate node embeddings

        Returns:
            tuple[Tensor, Tensor]: node embeddings, node unnormalized predictions
        """

        for i in range(self.current_stage + 1):
            xs[i] = self.base[i](xs[i])
        
        h = xs[-1]
        x = self.jk[self.current_stage](torch.stack(xs, dim=-1))

        if self.normalize:
            x = F.normalize(x, p=2, dim=-1)
        
        y = self.head[self.current_stage](x)
        return h, y

    def step(self, data: Data, phase: Phase) -> tuple[Optional[Tensor], Metrics]:
        xs = [data[f'x{i}'][data.batch_nodes] for i in range(self.current_stage + 1)]
        y = data.y[data.batch_nodes]

        preds: Tensor = self(xs)[1]
        if self.regression:
            self.regression_r2 = RegressionR2()
            self.regression_r2.update(preds, y)
            metrics = {f'{phase}/r2': self.regression_r2.compute()}
        elif getattr(self, 'binary', False):
            scores = preds.detach().squeeze(-1)
            score = binary_auroc(scores, y) * 100
            metrics = {f'{phase}/auroc': score}
        elif y.ndim == 2:
            positive, actual = preds.detach() >= 0, y.bool()
            numerator = 2 * (positive & actual).sum()
            denominator = positive.sum() + actual.sum()
            score = numerator / denominator.clamp_min(1) * 100
            metrics = {f'{phase}/micro_f1': score}
        else:
            score = preds.detach().argmax(dim=1).eq(y).float().mean() * 100
            metrics = {f'{phase}/acc': score}

        loss = None
        if phase != 'test':
            if getattr(self, 'binary', False) and not self.regression:
                loss = F.binary_cross_entropy_with_logits(
                    preds.squeeze(-1), y.float(), reduction='mean'
                )
            else:
                loss = self.root_losses(preds, y, regression=self.regression).mean()
            metrics[f'{phase}/loss'] = loss.detach()

        return loss, metrics

    @staticmethod
    def root_losses(preds: Tensor, y: Tensor, *, regression: bool = False) -> Tensor:
        if regression:
            if (
                preds.ndim not in (1, 2) or (preds.ndim == 2 and preds.shape[1] != 1)
                or y.ndim not in (1, 2) or (y.ndim == 2 and y.shape[1] != 1)
                or preds.shape[0] != y.shape[0]
            ):
                raise ValueError('Regression requires one scalar prediction and target per root')
            return (preds.reshape(-1) - y.reshape(-1).to(preds.dtype)).square()
        if y.ndim == 2:
            return F.binary_cross_entropy_with_logits(
                preds, y.float(), reduction='none'
            ).mean(dim=1)
        return F.cross_entropy(preds, y, reduction='none')

    def predict(self, data: Data) -> tuple[Tensor, Tensor]:
        xs = [data[f'x{i}'][data.batch_nodes] for i in range(self.current_stage + 1)]
        x, y = self(xs)
        if self.regression:
            return x, y
        probabilities = (
            torch.sigmoid(y)
            if getattr(self, 'binary', False) or getattr(self, 'multilabel', False)
            else torch.softmax(y, dim=-1)
        )
        return x, probabilities
        
    def reset_parameters(self):
        self.current_stage = 0
        for encoder in self.base:
            encoder.reset_parameters()
        for jk in self.jk:
            jk.reset_parameters()
        for head in self.head:
            head.reset_parameters()
        for param in super().parameters():
            param.requires_grad = True

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        if not self.layerwise:
            for i in range(self.current_stage):
                yield from self.base[i].parameters(recurse=recurse)
        yield from self.base[self.current_stage].parameters(recurse=recurse)
        yield from self.jk[self.current_stage].parameters(recurse=recurse)
        yield from self.head[self.current_stage].parameters(recurse=recurse)


class RegressionR2:
    """Merge centered float64 statistics without retaining predictions or targets."""

    def __init__(self):
        self.count = 0
        self.mean = None
        self.centered_sum = None
        self.residual_sum = None

    def update(self, preds: Tensor, target: Tensor) -> None:
        batch = RegressionR2()
        target = target.detach().reshape(-1).double()
        preds = preds.detach().reshape(-1).double()
        batch.count = target.numel()
        if batch.count:
            batch.mean = target.mean()
            batch.centered_sum = (target - batch.mean).square().sum()
            batch.residual_sum = (target - preds).square().sum()
            self.merge(batch)

    def merge(self, other: 'RegressionR2') -> None:
        if not other.count:
            return
        if not self.count:
            self.count = other.count
            self.mean = other.mean
            self.centered_sum = other.centered_sum
            self.residual_sum = other.residual_sum
            return
        count = self.count + other.count
        delta = other.mean - self.mean
        self.centered_sum = (
            self.centered_sum + other.centered_sum
            + delta.square() * (self.count * other.count / count)
        )
        self.mean = self.mean + delta * (other.count / count)
        self.residual_sum = self.residual_sum + other.residual_sum
        self.count = count

    def compute(self) -> Tensor:
        if not self.count:
            return torch.tensor(float('nan'), dtype=torch.float64)
        if self.count < 2:
            return self.mean.new_tensor(float('nan'))
        # Match sklearn's force_finite=True convention for constant targets.
        score = torch.where(
            self.centered_sum == 0,
            (self.residual_sum == 0).to(self.mean.dtype),
            1 - self.residual_sum / self.centered_sum,
        )
        return torch.where(
            torch.isfinite(self.residual_sum) & torch.isfinite(self.centered_sum),
            score,
            self.mean.new_tensor(float('nan')),
        )


def binary_auroc(scores: Tensor, target: Tensor) -> Tensor:
    """Return rank AUROC with average ranks for tied scores."""
    scores = scores.reshape(-1)
    target = target.reshape(-1).bool()
    positives = target.sum()
    negatives = target.numel() - positives
    if positives == 0 or negatives == 0:
        return scores.new_tensor(float('nan'))

    order = torch.argsort(scores)
    sorted_scores = scores[order]
    sorted_target = target[order]
    _, counts = torch.unique_consecutive(sorted_scores, return_counts=True)
    ends = counts.cumsum(0).to(dtype=scores.dtype)
    average_ranks = ends - (counts.to(dtype=scores.dtype) - 1) / 2
    ranks = torch.repeat_interleave(average_ranks, counts)
    positive_rank_sum = ranks[sorted_target].sum()
    positives = positives.to(dtype=scores.dtype)
    negatives = negatives.to(dtype=scores.dtype)
    return (
        positive_rank_sum - positives * (positives + 1) / 2
    ) / (positives * negatives)
        