import opti
import pytest
from plotly.graph_objects import Figure

import mbo


@pytest.fixture
def optimizer():
    problem = opti.problems.ZDT1(n_inputs=2)
    problem.create_initial_data(10)
    optimizer = mbo.algorithm.ParEGO(problem, restarts=1)
    optimizer._fit_model()
    return optimizer


def test_residuals_plot(optimizer):
    figs = mbo.plot.residuals(optimizer)
    assert isinstance(figs, list)
    assert isinstance(figs[0], Figure)


def test_parallel_data(optimizer):
    fig = mbo.plot.parallel_data(optimizer._problem)
    assert isinstance(fig, Figure)


def test_scatter_data(optimizer):
    fig = mbo.plot.scatter_data(optimizer._problem)
    assert isinstance(fig, Figure)


def test_correlation_data(optimizer):
    fig = mbo.plot.correlation_data(optimizer._problem)
    assert isinstance(fig, Figure)


def test_parallel_model(optimizer):
    fig = mbo.plot.parallel_model(optimizer)
    assert isinstance(fig, Figure)


def test_parallel_parameters(optimizer):
    fig = mbo.plot.parallel_parameters(optimizer)
    assert isinstance(fig, Figure)
