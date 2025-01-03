import logging
from rich.logging import RichHandler

# Configure logging with rich
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)

class EfficientLogger(logging.Logger):
    def debug(self, msg, *args, **kwargs):
        if self.isEnabledFor(logging.DEBUG):
            super().debug(msg, *args, **kwargs)

logging.setLoggerClass(EfficientLogger)

# Get the package logger
logger = logging.getLogger(__name__)