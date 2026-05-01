"""Awaking OS settings — pulled from env with `AWAKING_` prefix."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class AwakingSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AWAKING_", env_file=".env", extra="ignore")

    data_dir: Path = Path(".awaking")
    log_level: str = "INFO"
    kernel_max_concurrent: int = 4
    kernel_dispatch_timeout_s: float = 30.0

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
