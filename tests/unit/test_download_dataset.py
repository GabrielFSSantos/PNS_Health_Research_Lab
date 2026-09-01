"""Resolução de dataset no download."""

from pns_health_research.data.catalog import resolve_edition


def test_resolve_edition_for_download_dataset_key():
    edition = resolve_edition(2019, "pns_2019")
    assert edition.year == 2019
    assert edition.key == "pns_2019"
    assert "pns_2019" in str(edition.raw_path)
