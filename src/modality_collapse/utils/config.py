"""OmegaConf-based YAML configuration loading and merging."""

import sys
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def load_config(path: str) -> DictConfig:
    """Load a YAML configuration file.

    Args:
        path: Path to the YAML file.

    Returns:
        DictConfig loaded from the file.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    return OmegaConf.load(path)


def merge_configs(*configs: DictConfig) -> DictConfig:
    """Merge multiple configs. Later configs override earlier ones.

    Args:
        *configs: Variable number of DictConfig objects.

    Returns:
        Merged DictConfig.
    """
    return OmegaConf.merge(*configs)


def config_from_cli(config_path: str) -> DictConfig:
    """Load a YAML config then merge CLI overrides on top.

    CLI overrides are parsed from sys.argv via OmegaConf.from_cli().
    Example usage:
        python script.py --config base.yaml model.name=llama batch_size=32

    Args:
        config_path: Path to the base YAML config file.

    Returns:
        DictConfig with CLI overrides merged on top of file config.
    """
    base = load_config(config_path)
    cli = OmegaConf.from_cli()
    return merge_configs(base, cli)
