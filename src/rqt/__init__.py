import logging
from rich.logging import RichHandler

# Configure logging with rich
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)

# Get the package logger
logger = logging.getLogger(__name__)
