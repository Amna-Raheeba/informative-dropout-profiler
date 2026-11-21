import pytest

from dropout_profiler.generate import apply_dropout, generate_fleet
from dropout_profiler.models import REGIMES


@pytest.fixture(scope="module")
def fleet():
    return generate_fleet(5, 24)


@pytest.fixture(scope="module")
def masked_by_regime(fleet):
    return dict(
        (regime, apply_dropout(fleet, regime, 5)) for regime in REGIMES
    )
