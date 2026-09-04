"""Detalhe de agrupamento (clustering)."""

from __future__ import annotations

import streamlit as st

from pns_health_research.modules.analysis.cluster_service_usage import VizVariableOption
from pns_health_research.dashboard.components import charts
from pns_health_research.dashboard.components.chart_block import chart_block
from pns_health_research.dashboard.components.filters import init_session_state, run_selector
from pns_health_research.dashboard.components.insight_panel import render_insights
from pns_health_research.dashboard.components.kpi_cards import kpi_row
from pns_health_research.dashboard.components.metric_card import metric_row
from pns_health_research.dashboard.components.layout import page_header
from pns_health_research.dashboard.components.section_block import section_block
from pns_health_research.dashboard.components.states import empty_state
from pns_health_research.dashboard.content import pt_br
from pns_health_research.dashboard.content.value_labels import sort_health_categories_bubble
from pns_health_research.dashboard.insights import engine as insights_engine
from pns_health_research.dashboard.services import catalog, clustering

_NONE_CROSS = "— Nenhum —"
_DEFAULT_SPLIT = "Serviço de Saúde"
_DEFAULT_CATEGORY_KEY = "group:Saúde"


def _silhouette_interpretation(score: float | None) -> str:
    if score is None:
        return ""
    if score >= 0.5:
        return "Estrutura de clusters **forte** (silhueta ≥ 0,5)."
    if score >= 0.25:
        return "Estrutura **razoável** — clusters parcialmente separados."
    return "Estrutura **fraca** — clusters podem se sobrepor."


def _option_index(options: list[VizVariableOption], key: str, fallback: int = 0) -> int:
    for idx, opt in enumerate(options):
        if opt.key == key or opt.label == key:
            return idx
    return min(fallback, max(len(options) - 1, 0))


def _render_variable_selectors(
    catalog_data: dict,
    *,
    split_key: str,
    category_key: str,
) -> tuple[VizVariableOption | None, VizVariableOption | None]:
    split_options: list[VizVariableOption] = catalog_data.get("split", [])
    category_options: list[VizVariableOption] = catalog_data.get("category", [])

    if not split_options or not category_options:
        return None, None

    col1, col2 = st.columns(2)
    split_idx = _option_index(split_options, _DEFAULT_SPLIT)
    category_idx = _option_index(category_options, _DEFAULT_CATEGORY_KEY, fallback=0)
    with col1:
        split_sel = st.selectbox(
            pt_br.LABEL_BUBBLE_SPLIT,
            options=range(len(split_options)),
            format_func=lambda i: split_options[i].label,
            index=split_idx,
            key=split_key,
        )
    with col2:
        category_sel = st.selectbox(
            pt_br.LABEL_BUBBLE_CATEGORY,
            options=range(len(category_options)),
            format_func=lambda i: category_options[i].label,
            index=category_idx,
            key=category_key,
        )
    return split_options[split_sel], category_options[category_sel]


def render() -> None:
    init_session_state()
    page_header(
        "Agrupamento (clustering)",
        "K-Means + PCA: qualidade, método, perfis e variáveis por cluster.",
    )

    run_id = run_selector("Execução")
    if not run_id:
        empty_state("Nenhuma execução disponível.")
        return

    metrics = clustering.load_clustering_metrics(run_id)
    sil_summary = clustering.load_silhouette_summary(run_id)
    sil_score = sil_summary.get("overall") or sil_summary.get("score") or metrics.get("silhouette")
    best_k = clustering.resolve_best_k(metrics)

    render_insights(
        insights_engine.generate(
            {
                "page": "clustering",
                "silhouette_score": sil_score,
                "artifacts_status": catalog.artifacts_status(run_id),
            }
        )
    )

    sizes = metrics.get("cluster_sizes") or {}
    kpi_row(
        [
            ("k ótimo", best_k if best_k is not None else "—", None),
            ("Silhueta", f"{sil_score:.4f}" if sil_score is not None else "—", None),
            ("Clusters", len(sizes) if sizes else best_k or "—", None),
        ]
    )

    tabs = st.tabs(
        [
            pt_br.TAB_PANORAMA,
            pt_br.TAB_QUALITY,
            pt_br.TAB_METHOD,
            pt_br.TAB_PROFILES,
            pt_br.TAB_VARIABLES_BY_CLUSTER,
        ]
    )

    bubble_df_legacy = clustering.load_bubble_health(run_id)
    cross_df = clustering.load_cross_analysis(run_id)
    profiles = clustering.load_profiles(run_id)
    viz_catalog = clustering.get_viz_variable_catalog(run_id)
    has_binary = not clustering.load_clustered_binary(run_id).empty

    with tabs[0]:
        split_opt, category_opt = _render_variable_selectors(
            viz_catalog,
            split_key=f"bubble_split_{run_id}",
            category_key=f"bubble_category_{run_id}",
        )
        if split_opt and category_opt and has_binary:
            bubble_df = clustering.build_bubble_df(
                run_id, split_opt.columns[0], category_opt.columns
            )
            category_sorter = (
                sort_health_categories_bubble
                if category_opt.key == _DEFAULT_CATEGORY_KEY
                else None
            )
            chart_block(
                "bubble_usage",
                f"Bolhas — {split_opt.label} × {category_opt.label}",
                lambda: charts.bubble_cross_plot(
                    bubble_df,
                    x_label=category_opt.label,
                    y_label=f"{split_opt.label} × cluster",
                    split_label_name=split_opt.label,
                    category_sorter=category_sorter,
                ),
                chart_id="clustering_overview_bubble",
            )
        elif not bubble_df_legacy.empty:
            if not has_binary:
                st.caption(pt_br.MSG_CLUSTERED_BINARY_UNAVAILABLE)
            chart_block(
                "bubble_usage",
                "Gráfico de bolhas — saúde × uso de serviços",
                lambda: charts.bubble_health_service_plot(bubble_df_legacy),
                chart_id="clustering_overview_bubble",
            )
        elif not cross_df.empty:
            chart_block(
                "bubble_usage",
                "Uso de serviços por cluster (visão resumida)",
                lambda: charts.bubble_fallback_cross(cross_df),
                chart_id="clustering_overview_bubble_fallback",
            )
        if not cross_df.empty:
            chart_block(
                "cluster_size_bar",
                "Tamanho dos clusters",
                lambda: charts.cluster_size_bar(cross_df),
                chart_id="clustering_overview_size",
            )

    with tabs[1]:
        with section_block(
            "Indicadores de qualidade",
            "silhouette",
            intro=(
                "Quanto **maior a silhueta** e **menor o Davies-Bouldin**, melhor a separação "
                "dos clusters formados pelo K-Means."
            ),
        ):
            metric_row(
                [
                    ("silhouette", sil_score if isinstance(sil_score, (int, float)) else None, False),
                    ("davies_bouldin", metrics.get("davies_bouldin"), False),
                    ("calinski_harabasz", metrics.get("calinski_harabasz"), False),
                ]
            )
            interp = _silhouette_interpretation(
                sil_score if isinstance(sil_score, (int, float)) else None
            )
            if interp:
                st.markdown(interp)
        chart_block(
            "silhouette",
            "Análise de silhueta e dispersão PCA",
            lambda: charts.silhouette_dual_plot(
                clustering.load_silhouette_samples(run_id),
                clustering.load_pca_scatter(run_id),
                sil_summary,
            ),
            chart_id="clustering_silhouette",
        )

    with tabs[2]:
        chart_block(
            "elbow",
            "Método do cotovelo (escolha de k)",
            lambda: charts.elbow_plot(clustering.load_elbow(run_id), best_k=best_k),
            chart_id="clustering_elbow",
        )
        chart_block(
            "pca_variance",
            "Variância explicada (PCA)",
            lambda: charts.pca_variance_plot(clustering.load_pca_variance(run_id)),
            chart_id="clustering_pca_variance",
        )
        chart_block(
            "pca_loadings",
            "Loadings da PCA",
            lambda: charts.pca_loadings_heatmap(clustering.load_pca_loadings(run_id)),
            chart_id="clustering_pca_loadings",
        )
        chart_block(
            "pca_scatter",
            "Dispersão no espaço PCA",
            lambda: charts.pca_scatter_plot(clustering.load_pca_scatter(run_id)),
            chart_id="clustering_pca_scatter",
        )

    with tabs[3]:
        chart_block(
            "radar_profiles",
            "Perfis radar por cluster",
            lambda: charts.cluster_radar_plot(profiles),
            chart_id="clustering_radar",
        )
        chart_block(
            "heatmap_profiles",
            "Mapa de calor dos perfis",
            lambda: charts.profile_heatmap(profiles),
            chart_id="clustering_profiles_heatmap",
        )

    with tabs[4]:
        split_options: list[VizVariableOption] = viz_catalog.get("split", [])
        category_options: list[VizVariableOption] = viz_catalog.get("category", [])

        if not split_options and profiles.empty and cross_df.empty:
            st.info("Dados de variáveis por cluster indisponíveis nesta execução.")
        else:
            primary_idx = _option_index(split_options, _DEFAULT_SPLIT) if split_options else 0
            col1, col2 = st.columns(2)
            with col1:
                if split_options:
                    primary_sel = st.selectbox(
                        pt_br.LABEL_PRIMARY_VARIABLE,
                        options=range(len(split_options)),
                        format_func=lambda i: split_options[i].label,
                        index=primary_idx,
                        key=f"usage_primary_{run_id}",
                    )
                    primary_opt = split_options[primary_sel]
                else:
                    primary_opt = None
                    st.caption(pt_br.MSG_PROFILES_ONLY_VARIABLES)

            cross_choices = [_NONE_CROSS] + [opt.label for opt in category_options]
            with col2:
                cross_label = st.selectbox(
                    pt_br.LABEL_CROSS_VARIABLE,
                    options=cross_choices,
                    index=0,
                    key=f"usage_cross_{run_id}",
                )

            if primary_opt:
                rate_df = clustering.build_rate_df(run_id, primary_opt.columns[0])
                if rate_df.empty:
                    st.warning(
                        f"Variável **{primary_opt.label}** não disponível nos perfis desta execução."
                    )
                else:
                    if (
                        not rate_df.empty
                        and "from_profiles" in rate_df.columns
                        and bool(rate_df["from_profiles"].iloc[0])
                    ):
                        st.caption(pt_br.MSG_PROFILES_RATE_FALLBACK)
                    chart_block(
                        "cross_analysis_bar",
                        f"Taxa de {primary_opt.label} por cluster",
                        lambda df=rate_df, lbl=primary_opt.label: charts.variable_rate_bar(
                            df, variable_label=lbl
                        ),
                        chart_id="clustering_usage_bar",
                    )

                    if cross_label != _NONE_CROSS:
                        cross_opt = next(
                            (o for o in category_options if o.label == cross_label), None
                        )
                        if cross_opt:
                            cross_df_heat = clustering.build_cross_rate_df(
                                run_id,
                                primary_opt.columns[0],
                                cross_opt.columns,
                            )
                            if cross_df_heat.empty:
                                st.caption(pt_br.MSG_CROSS_UNAVAILABLE)
                            else:
                                chart_block(
                                    "cluster_cross_heatmap",
                                    f"{primary_opt.label} × {cross_opt.label}",
                                    lambda df=cross_df_heat, p=primary_opt.label, s=cross_opt.label: charts.cluster_cross_heatmap(
                                        df, primary_label=p, secondary_label=s
                                    ),
                                    chart_id="clustering_usage_cross",
                                )
            elif not cross_df.empty:
                chart_block(
                    "cross_analysis_bar",
                    "Taxa de uso de serviços de saúde por cluster",
                    lambda: charts.cross_analysis_bar(cross_df),
                    chart_id="clustering_usage_bar",
                )

    detail = catalog.get_detail(run_id)
    with st.expander(pt_br.TECHNICAL_DETAILS_DEV):
        st.markdown("Métricas brutas e linhagem da execução (para desenvolvedores).")
        st.json({"metrics": metrics, "lineage": detail.get("lineage") or {}})
