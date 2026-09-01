"""Painel de insights."""

from __future__ import annotations

import streamlit as st

from pns_health_research.dashboard.insights.rules import Insight, InsightLevel


def render_insights(insights: list[Insight], title: str = "Insights") -> None:
    if not insights:
        return
    st.markdown(f"#### {title}")
    for item in insights:
        css = "pns-insight"
        if item.level == InsightLevel.WARNING:
            css += " pns-insight-warning"
        elif item.level == InsightLevel.SUCCESS:
            css += " pns-insight-success"
        elif item.level == InsightLevel.DANGER:
            css += " pns-insight-danger"
        st.markdown(f'<div class="{css}">{item.message}</div>', unsafe_allow_html=True)
