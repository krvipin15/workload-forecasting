"""Application Settings and Configuration Management.

This module defines the type-safe application configuration schema using
Pydantic Settings and Pydantic v2. It manages environment variables, `.env` file
overrides, runtime directories, and hyperparameter configuration (`params.yaml`).
"""

from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict


def _find_project_root(current_path: Path) -> Path:
    """Locate the project root by searching for known anchor files.

    Starting from ``current_path``, recursively walks up the directory tree
    until a directory containing the project's anchor files is found.

    Parameters
    ----------
    current_path : pathlib.Path
        Starting path from which the parent directories are searched.

    Returns
    -------
    pathlib.Path
        Path to the detected project root directory.
    """
    for parent in [current_path, *list(current_path.parents)]:
        if (
            (parent / "pyproject.toml").exists()
            or (parent / "params.yaml").exists()
            or (parent / ".git").exists()
        ):
            return parent

    return current_path.resolve().parents[2]


# Get the base directory for the project
BASE_DIR = _find_project_root(Path(__file__).resolve())
DEFAULT_PARAMS_FILE = BASE_DIR / "params.yaml"


class Environment(StrEnum):
    """Enumeration of supported application logging levels."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class LogLevel(StrEnum):
    """Enumeration of logging levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ImmutableBaseModel(BaseModel):
    """Base Pydantic model enforcing immutable and strict configuration.

    Subclasses inherit validation behavior intended to prevent accidental
    mutation of application configuration after initialization.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        populate_by_name=True,
    )


class PreprocessingParams(ImmutableBaseModel):
    """Define the preprocessing parameters.

    This model defines the parameters used for data preprocessing.
    The parameters are validated to ensure they meet the required constraints.
    """

    step_seconds: int = Field(default=300, ge=1)
    holdout: int = Field(default=24, ge=1)


class ForecastingParams(ImmutableBaseModel):
    """Define rolling-origin CV and candidate-model parameters."""

    cv_windows: int = Field(default=4, ge=1)
    cv_step: int = Field(default=12, ge=1)
    min_train: int = Field(default=24, ge=2)
    window_sizes: list[int] = Field(default_factory=lambda: [6, 12])
    season_lengths: list[int] = Field(default_factory=lambda: [12])
    interval_level: int = Field(default=90, ge=1, le=99)
    peak_quantile: float = Field(default=0.9, gt=0, lt=1)


class PipelineParams(ImmutableBaseModel):
    """Define the complete validated pipeline configuration.

    Aggregates dataset schema, preprocessing, training, evaluation, and
    metadata configuration into a single type-safe parameter object.
    """

    preprocessing: PreprocessingParams = Field(default_factory=PreprocessingParams)
    forecasting: ForecastingParams = Field(default_factory=ForecastingParams)

    @classmethod
    def load_from_yaml(cls, path: Path = DEFAULT_PARAMS_FILE) -> "PipelineParams":
        """Load and validate pipeline parameters from a YAML configuration file.

        The YAML contents are parsed into the validated parameter model. If the
        configuration file does not exist, the model's default configuration is
        returned instead.

        Parameters
        ----------
        path : pathlib.Path, default=DEFAULT_PARAMS_FILE
            Path to the YAML parameter configuration file.

        Returns
        -------
        PipelineParams
            Validated pipeline parameter configuration.

        Raises
        ------
        ValueError
            If the YAML file exists but contains invalid parameter values or
            cannot be parsed into the expected configuration schema.
        """
        if not path.is_file():
            return cls()

        try:
            with path.open(encoding="utf-8") as f:
                raw_data: dict[str, Any] = yaml.safe_load(f) or {}
            return cls.model_validate(raw_data)
        except yaml.YAMLError as exc:
            raise RuntimeError(f"Malformed YAML configuration in '{path}': {exc}") from exc
        except ValidationError as exc:
            raise RuntimeError(f"Invalid pipeline parameter schema in '{path}': {exc}") from exc


class Settings(BaseSettings):
    """Define validated application-wide runtime configuration.

    Configuration values are loaded from environment variables and the
    project-level ``.env`` file, with environment variables taking
    precedence.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Runtime configuration
    ENVIRONMENT: Environment = Field(default=Environment.DEVELOPMENT)
    LOG_LEVEL: LogLevel = Field(default=LogLevel.INFO)
    LOGGER_NAME: str = Field(default="workload_forecast", min_length=1)

    # Workspace Paths
    LOGS_DIR: Path = Field(default_factory=lambda: BASE_DIR / "logs")
    PREDICTION_DATA_DIR: Path = Field(default_factory=lambda: BASE_DIR / "predictions")
    TRAINING_REPORT_DIR: Path = Field(default_factory=lambda: BASE_DIR / "reports" / "training")
    VALIDATION_REPORT_DIR: Path = Field(default_factory=lambda: BASE_DIR / "reports" / "validation")
    FIGURE_DIR: Path = Field(default_factory=lambda: BASE_DIR / "reports" / "figures")

    # Pipeline Parameters
    _params_cache: PipelineParams | None = None

    @property
    def PARAMS(self) -> PipelineParams:
        """Return the lazily loaded pipeline configuration.

        Loads and validates ``params.yaml`` on first access and caches the
        resulting configuration for subsequent accesses.

        Returns
        -------
        PipelineParams
            Validated pipeline hyperparameters and processing configuration.
        """
        if self._params_cache is None:
            self._params_cache = PipelineParams.load_from_yaml(DEFAULT_PARAMS_FILE)
        return self._params_cache

    def ensure_directories(self) -> None:
        """Create required application workspace directories.

        Ensures that directories required for data ingestion, preprocessing,
        models, reports, predictions, logs, and other runtime artifacts exist.
        Existing directories are left unchanged.
        """
        directories = [
            self.LOGS_DIR,
            self.PREDICTION_DATA_DIR,
            self.VALIDATION_REPORT_DIR,
            self.TRAINING_REPORT_DIR,
            self.FIGURE_DIR,
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Return the cached application settings.

    Local development may use a project-level environment file, while
    deployment environments such as Render provide configuration directly
    through environment variables.
    """
    settings = Settings()
    settings.ensure_directories()

    return settings
