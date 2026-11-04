import pytest

from dlnn.data import narma10
from dlnn.model import reservoir


@pytest.fixture(scope="session")
def narma_ds():
    return narma10.dataset()


@pytest.fixture(scope="session")
def narma_model_npz(tmp_path_factory, narma_ds):
    return reservoir.fit(narma_ds).save(tmp_path_factory.mktemp("m") / "model.npz")
