"""AwakingSettings constraint tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from awaking_os.config import AwakingSettings


def test_defaults_are_valid() -> None:
    s = AwakingSettings()
    assert s.kernel_max_concurrent == 4
    assert s.kernel_dispatch_timeout_s == 30.0


def test_kernel_max_concurrent_must_be_positive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWAKING_KERNEL_MAX_CONCURRENT", "0")
    with pytest.raises(ValidationError):
        AwakingSettings()


def test_kernel_max_concurrent_rejects_negative(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWAKING_KERNEL_MAX_CONCURRENT", "-1")
    with pytest.raises(ValidationError):
        AwakingSettings()


def test_dispatch_timeout_must_be_positive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWAKING_KERNEL_DISPATCH_TIMEOUT_S", "0")
    with pytest.raises(ValidationError):
        AwakingSettings()


def test_dispatch_timeout_rejects_negative(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWAKING_KERNEL_DISPATCH_TIMEOUT_S", "-1")
    with pytest.raises(ValidationError):
        AwakingSettings()
