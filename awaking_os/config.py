"""Awaking OS settings — pulled from env with `AWAKING_` prefix."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AwakingSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AWAKING_", env_file=".env", extra="ignore")

    data_dir: Path = Path(".awaking")
    log_level: str = "INFO"
    kernel_max_concurrent: int = Field(default=4, ge=1)
    kernel_dispatch_timeout_s: float = Field(default=30.0, gt=0)

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
