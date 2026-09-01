import pytest


@pytest.mark.gpu
def test_cuml_import_skipped_without_gpu():
    pytest.importorskip("cuml")
    import cuml  # noqa: F401
