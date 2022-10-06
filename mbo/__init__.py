__version__ = "0.1.2"

# flake8: noqa: F401
from mbo import algorithm, error, metric, optimize, plot, preproc

try:
    import torch_tools
except ImportError:
    pass
