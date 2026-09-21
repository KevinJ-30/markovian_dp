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
                 **kwargs,
                 ):

        super().__init__(**kwargs)

        self.num_classes = num_classes
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
        if y.ndim == 2:
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
            loss = self.root_losses(preds, y).mean()
            metrics[f'{phase}/loss'] = loss.detach()

        return loss, metrics

    @staticmethod
    def root_losses(preds: Tensor, y: Tensor) -> Tensor:
        if y.ndim == 2:
            return F.binary_cross_entropy_with_logits(
                preds, y.float(), reduction='none'
            ).mean(dim=1)
        return F.cross_entropy(preds, y, reduction='none')

    def predict(self, data: Data) -> tuple[Tensor, Tensor]:
        xs = [data[f'x{i}'][data.batch_nodes] for i in range(self.current_stage + 1)]
        x, y = self(xs)
        probabilities = torch.sigmoid(y) if getattr(self, 'multilabel', False) else torch.softmax(y, dim=-1)
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
        