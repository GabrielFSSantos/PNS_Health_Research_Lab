"""Testes do chart_block colapsável."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from pns_health_research.dashboard.components.chart_block import chart_block


@patch("pns_health_research.dashboard.components.chart_block.st")
def test_chart_block_collapsible_uses_expander(mock_st):
    mock_ctx = MagicMock()
    mock_st.expander.return_value.__enter__ = MagicMock(return_value=mock_ctx)
    mock_st.expander.return_value.__exit__ = MagicMock(return_value=False)
    called = []

    chart_block(
        "distribution",
        "Distribuições",
        lambda: called.append(True),
        collapsible=True,
        expanded=False,
    )
    mock_st.expander.assert_any_call("Distribuições", expanded=False)
    assert called == [True]


@patch("pns_health_research.dashboard.components.chart_block.CHART_HELP", {"distribution": ""})
@patch("pns_health_research.dashboard.components.chart_block.st")
def test_chart_block_non_collapsible_uses_heading(mock_st):
    called = []

    chart_block(
        "distribution",
        "Distribuições",
        lambda: called.append(True),
        collapsible=False,
    )
    mock_st.expander.assert_not_called()
    mock_st.markdown.assert_called_with("#### Distribuições")
    assert called == [True]
