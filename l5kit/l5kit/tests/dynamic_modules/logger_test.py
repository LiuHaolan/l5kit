from l5kit.dynamic_modules.loaders.runtime_params import RuntimeParams
from l5kit.dynamic_modules.loaders.logger import init_logger, get_logger


logger = get_logger(__name__)


def test_default_runtime_params() -> None:
    rp = RuntimeParams.get()
    init_logger(runtime_params=rp)
    logger.info("Sample info log")
    # Does nothing without raising any error
    logger.log_custom("data", data={"metric_a": 1})
    logger.log_custom("commit")