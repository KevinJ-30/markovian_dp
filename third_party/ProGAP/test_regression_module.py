import pytest
import torch
from opacus import GradSampleModule
from torch_geometric.data import Data

from core.modules.prog import ProgressiveModule, RegressionR2
from core.trainer.trainer import Trainer


def _module(**kwargs):
    options = dict(
        num_classes=1, num_stages=1, base_layers=0, head_layers=1,
        hidden_dim=1, normalize=False, batch_norm=False, regression=True,
        optimizer='sgd', learning_rate=0.0, weight_decay=0.0,
    )
    options.update(kwargs)
    model = ProgressiveModule(**options)
    model([torch.ones(1, 1)])
    with torch.no_grad():
        model.head[0].layers[0].weight.fill_(1)
        model.head[0].layers[0].bias.zero_()
    return model


def _batch(predictions, targets):
    return Data(
        x0=torch.tensor(predictions, dtype=torch.float32).reshape(-1, 1),
        y=torch.tensor(targets, dtype=torch.float32),
        batch_nodes=torch.arange(len(targets)),
    )


def test_regression_requires_scalar_head_in_constructor_and_assignment():
    with pytest.raises(ValueError, match='scalar head'):
        _module(num_classes=2)
    classification = _module(num_classes=2, regression=False)
    with pytest.raises(ValueError, match='scalar head'):
        classification.regression = True


def test_float_targets_have_one_squared_error_and_gradient_per_root():
    predictions = torch.tensor([[-3.5], [4.0]], requires_grad=True)
    targets = torch.tensor([-2.5, 1.5])
    losses = ProgressiveModule.root_losses(predictions, targets, regression=True)
    torch.testing.assert_close(losses, torch.tensor([1.0, 6.25]))
    torch.testing.assert_close(
        ProgressiveModule.root_losses(predictions, targets[:, None], regression=True),
        losses,
    )
    gradient, = torch.autograd.grad(losses.sum(), predictions)
    torch.testing.assert_close(gradient, torch.tensor([[-2.0], [5.0]]))
    with pytest.raises(ValueError, match='one scalar'):
        ProgressiveModule.root_losses(predictions.expand(-1, 2), targets, regression=True)


def test_regression_step_and_predict_preserve_unbounded_scalars():
    model = _module()
    data = _batch([-3.5, 4.0], [-2.5, 1.5])
    _, predictions = model.predict(data)
    torch.testing.assert_close(predictions, data.x0)
    loss, metrics = model.step(data, phase='val')
    assert loss.item() == pytest.approx(3.625)
    assert metrics['val/r2'].item() == pytest.approx(1 - 7.25 / 8)
    test_loss, test_metrics = model.step(data, phase='test')
    assert test_loss is None
    assert test_metrics['test/r2'].item() == pytest.approx(1 - 7.25 / 8)


def test_regression_opacus_gradients_match_individual_root_gradients():
    model = _module()
    features = torch.tensor([[-3.5], [4.0]])
    targets = torch.tensor([-2.5, 1.5])
    parameters = tuple(model.parameters())
    individual = []
    for index in range(2):
        predictions = model([features[index:index + 1]])[1]
        loss = model.root_losses(predictions, targets[index:index + 1], regression=True).sum()
        individual.append(torch.autograd.grad(loss, parameters))
    private_model = GradSampleModule(model)
    predictions = private_model([features])[1]
    model.root_losses(predictions, targets, regression=True).mean().backward()
    for index, parameter in enumerate(parameters):
        expected = torch.stack([gradients[index] for gradients in individual])
        torch.testing.assert_close(parameter.grad_sample, expected)


def test_r2_handles_constant_targets_and_undefined_singletons():
    perfect = RegressionR2()
    perfect.update(torch.tensor([[3.0], [3.0]]), torch.tensor([3.0, 3.0]))
    assert perfect.compute().item() == 1.0
    imperfect = RegressionR2()
    imperfect.update(torch.tensor([[3.0], [2.0]]), torch.tensor([3.0, 3.0]))
    assert imperfect.compute().item() == 0.0
    singleton = RegressionR2()
    singleton.update(torch.tensor([3.0]), torch.tensor([3.0]))
    assert torch.isnan(singleton.compute())
    assert torch.isnan(RegressionR2().compute())


def test_r2_does_not_hide_nonfinite_predictions_on_constant_targets():
    for prediction in (float('nan'), float('inf')):
        score = RegressionR2()
        score.update(torch.tensor([3.0, prediction]), torch.tensor([3.0, 3.0]))
        assert torch.isnan(score.compute())


def test_r2_keeps_centered_variance_for_large_offset_chunked_targets():
    targets = torch.arange(5, dtype=torch.float64) + 1e12
    score = RegressionR2()
    for target in targets.split(1):
        score.update((target + 2).reshape(-1, 1), target)
    assert score.compute().item() == pytest.approx(-1.0)


def test_native_trainer_uses_whole_phase_r2_not_mean_batch_r2():
    model = _module()
    batches = [_batch([0, 2], [0, 2]), _batch([0, 0, 0], [10, 12, 20])]
    targets = torch.cat([batch.y for batch in batches]).double()
    predictions = torch.cat([batch.x0[:, 0] for batch in batches]).double()
    expected = 1 - (targets - predictions).square().sum() / (targets - targets.mean()).square().sum()
    batch_average = sum(
        model.step(batch, 'test')[1]['test/r2'] * len(batch.y) for batch in batches
    ) / len(targets)
    assert expected < 0
    assert not torch.isclose(expected, batch_average)

    trainer = Trainer(epochs=1, monitor='val/r2', device='cpu', verbose=False)
    metrics = trainer.fit(model, batches, batches, batches)
    for phase in ('train', 'val', 'test'):
        torch.testing.assert_close(metrics[f'{phase}/r2'], expected)
    assert metrics['train/loss'].item() == pytest.approx(644 / 5)
    combined = _batch(predictions.tolist(), targets.tolist())
    torch.testing.assert_close(trainer.test([combined])['test/r2'], expected)
    # Reset between phases and merge singleton chunks without averaging NaNs.
    singletons = [_batch([prediction], [target]) for prediction, target in zip(predictions, targets)]
    torch.testing.assert_close(trainer.test(singletons)['test/r2'], expected)
    assert trainer.test([_batch([3, 3], [3, 3])])['test/r2'].item() == 1.0
