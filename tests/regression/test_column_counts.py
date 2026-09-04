import pytest


@pytest.mark.regression
def test_profile_column_counts(regression_bases):
    legacy = regression_bases["preprocess"]["legacy"].result
    pns = regression_bases["preprocess"]["pns"].result

    assert legacy.metrics["rows_out"] == 87678
    assert legacy.metrics["binary_columns"] == 179
    assert legacy.metrics["categorical_columns"] == 88

    assert pns.metrics["rows_out"] == 87678
    assert pns.metrics["rows_out"] == legacy.metrics["rows_out"]
    assert pns.metrics["binary_columns"] >= legacy.metrics["binary_columns"]
    assert pns.metrics["categorical_columns"] >= legacy.metrics["categorical_columns"]
