"""Testes do catálogo PNS."""

from pns_health_research.data.catalog import resolve_edition, resolve_raw_path


def test_resolve_edition_2019():
    ed = resolve_edition(2019)
    assert ed.is_supported
    assert ed.year == 2019
    assert ed.raw_path.name == "pns2019.parquet"
    assert ed.columns["J014"] == "Serviço de Saúde"


def test_resolve_edition_by_key():
    ed = resolve_edition(2019, "pns_2019")
    assert ed.key == "pns_2019"
    assert ed.columns["V0001"] == "UF"


def test_resolve_raw_path_with_year():
    p = resolve_raw_path(dataset="pns", year=2019)
    assert "2019" in str(p)
