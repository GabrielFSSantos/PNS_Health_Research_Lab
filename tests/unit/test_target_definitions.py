import pandas as pd

from pns_health_research.config.models import TargetColumnSpec, TargetConfig
from pns_health_research.modules.preprocessing.filtering import select_and_rename_legacy
from pns_health_research.modules.preprocessing.targets import build_target_columns


def test_v00291_preserved_after_select():
    df = pd.DataFrame({"V00291": [100.0], "J014": ["Sim"], "C008": [30], "P00104": [70], "P00404": [170]})
    cmap = {"V00291": "Peso Amostral", "J014": "Serviço de Saúde", "C008": "Idade", "P00104": "Peso", "P00404": "Altura"}
    out = select_and_rename_legacy(df, column_map=cmap)
    assert "Peso Amostral" in out.columns
    assert float(out["Peso Amostral"].iloc[0]) == 100.0


def test_dual_targets_prevalence():
    df = pd.DataFrame(
        {
            "Serviço de Saúde": [1, 0, 1, 0],
            "Internação": [0, 1, 0, 0],
        }
    )
    cfg = TargetConfig(
        primary_column="servico_composto",
        columns={
            "servico_j014": TargetColumnSpec(source="Serviço de Saúde", merge_internacao=False),
            "servico_composto": TargetColumnSpec(
                source="Serviço de Saúde", merge_internacao=True
            ),
        },
    )
    out = build_target_columns(df, cfg)
    assert list(out["servico_j014"]) == [1, 0, 1, 0]
    assert list(out["servico_composto"]) == [1, 1, 1, 0]
