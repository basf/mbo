# flake8: noqa: F401
# flake8: noqa: F401
from loguru import logger

from mbo.algorithm.algorithm import Algorithm
from mbo.algorithm.doe import DOptimalDesign
from mbo.algorithm.random import RandomSearch
from mbo.algorithm.rf import RandomForest

try:
    # requires mbo.orch
    from mbo.algorithm.parego import ParEGO
    from mbo.algorithm.sobo import SOBO
    from mbo.algorithm.tsemo import TSEMO
except ModuleNotFoundError as err:
    logger.info(f"Skipping ParEGO, SOBO, TSEMO: {err}")
    ParEGO = None
    SOBO = None
    TSEMO = None

try:
    # requires gurobipy and entmoot
    from mbo.algorithm.entmoot import Entmoot
except ModuleNotFoundError as err:
    logger.info(f"Skipping Entmoot: {err}")
    Entmoot = None
