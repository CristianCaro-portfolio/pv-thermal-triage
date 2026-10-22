from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "data" / "sample"


@pytest.fixture(scope="session")
def sample_dir() -> Path:
    return SAMPLE
