import yaml
from pathlib import Path
from photo_insight.utils.app_logger import AppLogger

logger = AppLogger().get_logger()


def parse_env_file(file_path: Path) -> dict:
    """Parse a conda environment YAML file and return its content as a dictionary."""
    try:
        with file_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        logger.debug(f"Successfully parsed environment file: {file_path}")
        return data
    except yaml.YAMLError as e:
        logger.error(f"YAML parsing error in {file_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to read file {file_path}: {e}")
        raise
