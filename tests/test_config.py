from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.config import Settings


@pytest.mark.parametrize("depth", [-1, 4])
def test_report_depth_must_be_between_zero_and_three(depth: int) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, report_depth=depth)


def test_report_depth_defaults_to_two() -> None:
    assert Settings(_env_file=None).report_depth == 2
