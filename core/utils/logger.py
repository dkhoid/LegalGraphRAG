import logging
import sys
import os
import warnings

# Suppress Neo4j deprecation warnings
try:
    from neo4j.exceptions import Neo4jDeprecationWarning

    warnings.filterwarnings("ignore", category=Neo4jDeprecationWarning)
except ImportError:
    pass
logging.getLogger("neo4j").setLevel(logging.ERROR)
logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)


def setup_logger(name="LegalGraphRAG", log_file="logs/app.log", level=logging.INFO):
    """
    Sets up a configured logger that outputs to both console and a file.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding handlers multiple times
    if not logger.handlers:
        # Formatter
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File Handler
        try:
            # Ensure the logs directory exists if a path is provided
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)

            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            logger.warning(f"Failed to setup file handler for logger: {e}")

    return logger


# Create a default logger instance
logger = setup_logger()
