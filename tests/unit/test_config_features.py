from pns_health_research.config.features import load_classification_features, load_column_map


def test_load_column_map_from_datasets():
    cmap = load_column_map()
    assert "V0001" in cmap
    assert cmap["V0001"] == "UF"
    assert "J014" in cmap


def test_load_classification_features_from_preprocessing():
    feats = load_classification_features()
    assert feats["target_column"] == "Serviço de Saúde"
    assert "Serviço de Saúde" in feats.get("exclude_columns", [])
