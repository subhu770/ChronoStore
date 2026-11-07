"""
Configuration settings for ChronoStore Feature Store.
Utilizes Pydantic Settings for environment-variable driven configuration.
"""

from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration with environment variable overrides."""

    model_config = SettingsConfigDict(
        env_prefix="CHRONOSTORE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application Metadata
    APP_NAME: str = "ChronoStore"
    VERSION: str = "1.0.0"
    ENV: str = "development"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Offline Store Configuration
    OFFLINE_STORE_DIR: Path = Path("./data/offline")
    OFFLINE_PARQUET_COMPRESSION: str = "snappy"

    # Online Store Configuration
    ONLINE_STORE_MAX_ENTRIES: Optional[int] = None
    ONLINE_STORE_DEFAULT_TTL_SECONDS: Optional[int] = None

    # Temporal / ASOF Configuration
    ASOF_STRICT_LEAKAGE_CHECK: bool = True
    ASOF_DEFAULT_LOOKBACK_SECONDS: Optional[float] = None

    def get_offline_dir(self) -> Path:
        """Ensure and return the offline store directory path."""
        self.OFFLINE_STORE_DIR.mkdir(parents=True, exist_ok=True)
        return self.OFFLINE_STORE_DIR


# Global singleton settings instance
settings = Settings()

