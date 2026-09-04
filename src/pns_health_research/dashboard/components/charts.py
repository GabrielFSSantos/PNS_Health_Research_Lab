"""Gráficos Plotly padronizados — estilo notebooks legados."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from pns_health_research.dashboard.components.chart_utils import (
    add_bar_value_labels,
    as_cluster_category,
    cluster_color,
    cluster_color_map,
    cluster_label,
    current_chart_key,
    format_count,
    format_percent,
    resolve_cluster_column,
)
from pns_health_research.dashboard.config import MODEL_COLORS
from pns_health_research.dashboard.content.pt_br import model_label
from pns_health_research.dashboard.content.value_labels import (
    friendly_column_name,
    funnel_stage_caption,
    funnel_stage_label,
    health_category_label,
    label_for_value,
    sort_health_categories,
    sort_health_categories_bubble,
)

_LAYOUT = dict(margin=dict(l=40, r=30, t=50, b=40), height=560)
_LAYOUT_TALL = dict(margin=dict(l=40, r=30, t=50, b=40), height=620)


def _show(fig: go.Figure, *, tall: bool = False) -> None:
    layout = _LAYOUT_TALL if tall else _LAYOUT
    fig.update_layout(**layout)
    st.plotly_chart(fig, width="stretch", key=current_chart_key())


def elbow_plot(df: pd.DataFrame, *, title: str = "Método do cotovelo (WCSS)", best_k: int | None = None) -> None:
    if df.empty or "k" not in df.columns:
        st.caption("Sem dados para o método do cotovelo.")
        return
    y_col = "wcss" if "wcss" in df.columns else df.columns[-1]
    fig = px.line(df, x="k", y=y_col, markers=True, title=title, text=y_col)
    fig.update_traces(texttemplate="%{y:,.0f}", textposition="top center")
    fig.update_layout(
        xaxis_title="Número de clusters (k)",
        yaxis_title="WCSS (soma dos quadrados intra-cluster)",
        xaxis=dict(dtick=1, type="linear"),
    )
    if best_k is not None:
        fig.add_vline(x=best_k, line_dash="dash", line_color="red", annotation_text=f"k ótimo = {best_k}")
    _show(fig)


def pca_variance_plot(df: pd.DataFrame, *, title: str = "Variância explicada pela PCA") -> None:
    if df.empty:
        st.caption("Sem dados de variância PCA.")
        return
    x_col = "component" if "component" in df.columns else df.columns[0]
    y_col = "explained_variance_ratio" if "explained_variance_ratio" in df.columns else df.columns[1]
    cum_col = "cumulative_variance" if "cumulative_variance" in df.columns else None

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=df[x_col],
            y=df[y_col],
            name="Variância explicada",
            marker_color="#2563eb",
            text=[f"{v:.1%}" for v in df[y_col]],
            textposition="outside",
        ),
        secondary_y=False,
    )
    if cum_col and cum_col in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df[x_col],
                y=df[cum_col],
                name="Acumulada",
                mode="lines+markers+text",
                text=[f"{v:.0%}" for v in df[cum_col]],
                textposition="top center",
                line=dict(color="#dc2626", width=2),
            ),
            secondary_y=True,
        )
    fig.update_layout(title=title)
    fig.update_xaxes(title_text="Componente principal")
    fig.update_yaxes(title_text="Variância explicada", tickformat=".0%", secondary_y=False)
    fig.update_yaxes(title_text="Variância acumulada", tickformat=".0%", secondary_y=True)
    _show(fig, tall=True)


def pca_scatter_plot(df: pd.DataFrame, *, title: str = "Dispersão PCA por cluster") -> None:
    if df.empty:
        st.caption("Sem amostra de dispersão PCA.")
        return
    plot_df = as_cluster_category(df)
    x_col = "pc1" if "pc1" in plot_df.columns else plot_df.columns[0]
    y_col = "pc2" if "pc2" in plot_df.columns else plot_df.columns[1]
    color_map = cluster_color_map(plot_df["cluster"])
    fig = px.scatter(
        plot_df,
        x=x_col,
        y=y_col,
        color="cluster_label",
        title=title,
        opacity=0.55,
        color_discrete_map={cluster_label(k): v for k, v in color_map.items()},
        labels={x_col: "PC1", y_col: "PC2", "cluster_label": "Cluster"},
    )
    # Centróides
    if "cluster" in plot_df.columns:
        for c in sorted(plot_df["cluster"].unique()):
            sub = plot_df[plot_df["cluster"] == c]
            fig.add_trace(
                go.Scatter(
                    x=[sub[x_col].mean()],
                    y=[sub[y_col].mean()],
                    mode="markers",
                    marker=dict(size=14, symbol="x", color=color_map[int(c)], line=dict(width=2)),
                    name=f"Centróide {cluster_label(c)}",
                    showlegend=True,
                )
            )
    _show(fig, tall=True)


def pca_loadings_heatmap(df: pd.DataFrame, *, title: str = "Loadings da PCA") -> None:
    if df.empty:
        st.caption("Sem loadings PCA.")
        return
    idx_col = "feature" if "feature" in df.columns else df.columns[0]
    mat = df.set_index(idx_col)
    h = max(420, 28 * len(mat))
    fig = px.imshow(mat, aspect="auto", title=title, color_continuous_scale="RdBu_r", text_auto=".2f")
    fig.update_layout(height=h)
    st.plotly_chart(fig, width="stretch", key=current_chart_key())


def silhouette_dual_plot(
    samples: pd.DataFrame,
    pca_scatter: pd.DataFrame,
    summary: dict[str, Any],
    *,
    title: str = "Análise de silhueta e dispersão PCA",
) -> None:
    score = summary.get("overall") or summary.get("score")
    if samples.empty or "silhouette" not in samples.columns:
        if score is not None:
            st.metric("Coeficiente médio de silhueta", f"{score:.4f}")
        else:
            st.caption("Sem amostras de silhueta.")
        return

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Coeficiente de silhueta", "Dados no espaço PCA"),
    )

    clusters = sorted(samples["cluster"].astype(int).unique()) if "cluster" in samples.columns else [0]
    y_lower = 10
    for i, c in enumerate(clusters):
        vals = samples.loc[samples["cluster"] == c, "silhouette"].sort_values().values
        size_i = len(vals)
        y_upper = y_lower + size_i
        color = cluster_color(i)
        fig.add_trace(
            go.Scatter(
                x=vals,
                y=np.linspace(y_lower, y_upper, size_i),
                mode="lines",
                fill="tozerox",
                line=dict(color=color),
                name=cluster_label(c),
                showlegend=True,
            ),
            row=1,
            col=1,
        )
        y_lower = y_upper + 10

    if score is not None:
        fig.add_vline(x=float(score), line_dash="dash", line_color="red", row=1, col=1)

    if not pca_scatter.empty:
        scatter_df = as_cluster_category(pca_scatter)
        x_col = "pc1" if "pc1" in scatter_df.columns else scatter_df.columns[0]
        y_col = "pc2" if "pc2" in scatter_df.columns else scatter_df.columns[1]
        color_map = cluster_color_map(scatter_df["cluster"])
        for c in sorted(scatter_df["cluster"].unique()):
            sub = scatter_df[scatter_df["cluster"] == c]
            fig.add_trace(
                go.Scatter(
                    x=sub[x_col],
                    y=sub[y_col],
                    mode="markers",
                    marker=dict(size=4, opacity=0.5, color=color_map[int(c)]),
                    name=cluster_label(c),
                    showlegend=False,
                ),
                row=1,
                col=2,
            )
            fig.add_trace(
                go.Scatter(
                    x=[sub[x_col].mean()],
                    y=[sub[y_col].mean()],
                    mode="markers",
                    marker=dict(size=12, symbol="x", color=color_map[int(c)]),
                    showlegend=False,
                ),
                row=1,
                col=2,
            )

    fig.update_xaxes(title_text="Coeficiente de silhueta", row=1, col=1)
    fig.update_xaxes(title_text="PC1", row=1, col=2)
    fig.update_yaxes(title_text="PC2", row=1, col=2, showticklabels=False)
    fig.update_layout(title=title, height=580, legend=dict(orientation="h", y=-0.12))
    st.plotly_chart(fig, width="stretch", key=current_chart_key())


def cluster_radar_plot(profiles: pd.DataFrame, *, title: str = "Perfis de multimorbidade por cluster") -> None:
    if profiles.empty:
        st.caption("Sem perfis de cluster.")
        return
    df = profiles.copy()
    cluster_col = resolve_cluster_column(df)
    if cluster_col is None:
        st.caption("Sem coluna de cluster nos perfis.")
        st.dataframe(df, width="stretch")
        return
    feature_cols = [c for c in df.columns if c != cluster_col]
    fig = go.Figure()
    for _, row in df.iterrows():
        c = int(row[cluster_col])
        vals = [float(row[f]) for f in feature_cols]
        fig.add_trace(
            go.Scatterpolar(
                r=vals + [vals[0]],
                theta=feature_cols + [feature_cols[0]],
                fill="toself",
                name=cluster_label(c),
                line_color=cluster_color(c),
            )
        )
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), title=title, height=580)
    st.plotly_chart(fig, width="stretch", key=current_chart_key())


def profile_heatmap(profiles: pd.DataFrame, *, title: str = "Mapa de calor — perfis por cluster") -> None:
    if profiles.empty:
        st.caption("Sem perfis para heatmap.")
        return
    cluster_col = resolve_cluster_column(profiles)
    if cluster_col is None:
        st.caption("Sem coluna de cluster nos perfis.")
        return
    mat = profiles.copy()
    mat[cluster_col] = mat[cluster_col].astype(int).map(cluster_label)
    mat = mat.set_index(cluster_col)
    fig = px.imshow(
        mat,
        aspect="auto",
        title=title,
        color_continuous_scale="Blues",
        labels=dict(x="Comorbidade", y="Cluster", color="Média"),
        text_auto=".2f",
    )
    fig.update_layout(height=max(400, 40 * len(mat)))
    st.plotly_chart(fig, width="stretch", key=current_chart_key())


def variable_rate_bar(
    df: pd.DataFrame,
    *,
    variable_label: str = "Variável",
    title: str | None = None,
) -> None:
    if df.empty:
        st.caption("Sem dados de taxa por cluster.")
        return
    plot_df = as_cluster_category(df)
    rate_col = "rate" if "rate" in plot_df.columns else "service_usage_rate"
    plot_title = title or f"Taxa de {variable_label} por cluster"
    plot_df["bar_text"] = plot_df.apply(
        lambda r: (
            f"{format_percent(r[rate_col])}"
            + (f"<br>n={format_count(r['n'])}" if pd.notna(r.get("n")) else "")
        ),
        axis=1,
    )
    color_map = cluster_color_map(plot_df["cluster"])
    fig = px.bar(
        plot_df,
        x="cluster_label",
        y=rate_col,
        text="bar_text",
        title=plot_title,
        color="cluster_label",
        color_discrete_map={cluster_label(k): v for k, v in color_map.items()},
        labels={"cluster_label": "Cluster", rate_col: f"Taxa de {variable_label}"},
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, yaxis_tickformat=".0%")
    _show(fig)


def cross_analysis_bar(df: pd.DataFrame, *, title: str = "Uso de serviços por cluster") -> None:
    variable_label = "uso de serviços de saúde"
    if "variable" in df.columns and df["variable"].nunique() == 1:
        variable_label = str(df["variable"].iloc[0])
    variable_rate_bar(df, variable_label=variable_label, title=title)


def cluster_size_bar(df: pd.DataFrame, *, title: str = "Tamanho dos clusters") -> None:
    if df.empty:
        st.caption("Sem dados de tamanho por cluster.")
        return
    plot_df = as_cluster_category(df)
    color_map = cluster_color_map(plot_df["cluster"])
    fig = px.bar(
        plot_df,
        x="cluster_label",
        y="n",
        text="n",
        title=title,
        color="cluster_label",
        color_discrete_map={cluster_label(k): v for k, v in color_map.items()},
        labels={"cluster_label": "Cluster", "n": "Contagem (n)"},
    )
    add_bar_value_labels(fig, fmt="%{y:,.0f}")
    fig.update_layout(showlegend=False)
    _show(fig)


def cluster_cross_heatmap(
    df: pd.DataFrame,
    *,
    primary_label: str,
    secondary_label: str,
    title: str | None = None,
) -> None:
    if df.empty:
        st.caption("Sem dados para o cruzamento.")
        return
    plot_df = df.copy()
    plot_df["cluster_label"] = plot_df["cluster"].map(cluster_label)
    pivot = plot_df.pivot_table(
        index="cluster_label",
        columns="secondary_category",
        values="rate",
        aggfunc="first",
    )
    plot_title = title or f"{primary_label} por cluster × {secondary_label}"
    fig = px.imshow(
        pivot,
        aspect="auto",
        title=plot_title,
        color_continuous_scale="Blues",
        labels=dict(x=secondary_label, y="Cluster", color=f"Taxa de {primary_label}"),
        text_auto=".1%",
    )
    fig.update_layout(height=max(400, 50 * len(pivot)))
    _show(fig)


def bubble_cross_plot(
    df: pd.DataFrame,
    *,
    x_label: str = "Categoria",
    y_label: str = "Divisão × cluster",
    split_label_name: str = "Divisão",
    category_sorter: Any | None = None,
    title: str = "Gráfico de Bolhas por Cluster — Porcentagem",
) -> None:
    if df.empty:
        st.caption("Artifact de bolhas não disponível — reexecute o experimento para gerar.")
        return

    plot_df = df.copy()
    plot_df["cluster"] = plot_df["cluster"].astype(int)

    if "split_label" in plot_df.columns:
        split_col = "split_label"
    else:
        split_col = "servico"
    if "category" in plot_df.columns:
        plot_df["categoria_label"] = plot_df["category"].map(health_category_label)
    else:
        plot_df["categoria_label"] = plot_df["categoria_saude"].map(health_category_label)

    cluster_totals = plot_df.groupby("cluster")["count"].sum()
    plot_df["pct"] = plot_df.apply(
        lambda r: r["count"] / cluster_totals[r["cluster"]] * 100 if cluster_totals[r["cluster"]] else 0,
        axis=1,
    )

    clusters = sorted(plot_df["cluster"].unique())
    raw_categories = plot_df["categoria_label"].unique().tolist()
    if category_sorter:
        categories = category_sorter(raw_categories)
    else:
        categories = sorted(raw_categories)
    cat_index = {c: i for i, c in enumerate(categories)}

    split_values = plot_df[split_col].unique().tolist()
    split_order = sorted(split_values, key=lambda x: (x != "Não", x))
    split_titles = {val: f"{split_label_name} — {val}" for val in split_order}
    n_clusters = len(clusters)
    y_ticks: list[float] = []
    y_labels: list[str] = []

    for block_idx, split_val in enumerate(split_order):
        base = block_idx * (n_clusters + 1)
        for cl in clusters:
            y_pos = base + cl
            y_ticks.append(float(y_pos))
            y_labels.append(f"{split_titles[split_val]} | {cluster_label(cl)}")

    rows: list[dict[str, Any]] = []
    for _, row in plot_df.iterrows():
        split_block = split_order.index(row[split_col])
        y_pos = split_block * (n_clusters + 1) + int(row["cluster"])
        rows.append(
            {
                "x": cat_index[row["categoria_label"]],
                "y": y_pos,
                "size": max(row["pct"], 0.05),
                "pct": row["pct"],
                "count": row["count"],
                "cluster": int(row["cluster"]),
                "categoria": row["categoria_label"],
                "split": row[split_col],
                "label": f"{row['pct']:.2f}%".replace(".", ","),
            }
        )

    if not rows:
        st.caption("Sem dados para o gráfico de bolhas.")
        return

    plot_pts = pd.DataFrame(rows)
    color_map = cluster_color_map(clusters)
    fig = go.Figure()

    for cl in clusters:
        sub = plot_pts[plot_pts["cluster"] == cl]
        fig.add_trace(
            go.Scatter(
                x=sub["x"],
                y=sub["y"],
                mode="markers+text",
                name=cluster_label(cl),
                marker=dict(
                    size=sub["size"] * 2.5,
                    sizemode="area",
                    sizeref=2.0 * max(plot_pts["size"].max(), 1) / (40.0**2),
                    sizemin=4,
                    color=color_map[int(cl)],
                    opacity=0.85,
                    line=dict(width=1, color="white"),
                ),
                text=sub["label"],
                textposition="middle center",
                textfont=dict(size=9, color="white"),
                hovertemplate=(
                    f"<b>{cluster_label(cl)}</b><br>"
                    "%{customdata[0]}<br>"
                    f"{split_label_name}: %{{customdata[1]}}<br>"
                    "Contagem: %{customdata[2]:,}<br>"
                    "Percentual: %{customdata[3]:.2f}%"
                    "<extra></extra>"
                ),
                customdata=np.stack(
                    [sub["categoria"], sub["split"], sub["count"], sub["pct"]], axis=-1
                ),
            )
        )

    fig.update_layout(
        title=title,
        xaxis=dict(
            tickmode="array",
            tickvals=list(range(len(categories))),
            ticktext=categories,
            title=x_label,
            type="linear",
        ),
        yaxis=dict(
            tickmode="array",
            tickvals=y_ticks,
            ticktext=y_labels,
            title=y_label,
            autorange="reversed",
        ),
        showlegend=True,
        legend=dict(title="Cluster"),
        height=720,
        margin=dict(l=220, r=30, t=60, b=60),
    )
    st.plotly_chart(fig, width="stretch", key=current_chart_key())


def bubble_health_service_plot(
    df: pd.DataFrame,
    *,
    title: str = "Gráfico de Bolhas por Cluster — Porcentagem",
) -> None:
    bubble_cross_plot(
        df,
        x_label="Autoavaliação de saúde",
        y_label="Uso de serviços × cluster",
        split_label_name="Serviço de Saúde",
        category_sorter=sort_health_categories_bubble,
        title=title,
    )


def bubble_fallback_cross(df: pd.DataFrame, *, title: str = "Tamanho e uso de serviços por cluster") -> None:
    if df.empty:
        return
    plot_df = as_cluster_category(df)
    color_map = cluster_color_map(plot_df["cluster"])
    fig = px.scatter(
        plot_df,
        x="cluster_label",
        y="service_usage_rate",
        size="n",
        color="cluster_label",
        title=title,
        size_max=80,
        color_discrete_map={cluster_label(k): v for k, v in color_map.items()},
        labels={
            "cluster_label": "Cluster",
            "service_usage_rate": "Taxa de uso de serviços",
            "n": "Tamanho da amostra",
        },
        hover_data={"n": True, "service_usage_rate": ":.1%"},
    )
    fig.update_layout(yaxis_tickformat=".0%")
    _show(fig, tall=True)


def preprocess_funnel(funnel: dict[str, Any], *, title: str = "Funil de pré-processamento") -> None:
    stages = funnel.get("stages") or funnel.get("steps") or []
    if not stages:
        st.caption("Funil de pré-processamento não disponível.")
        return

    profile_kind = funnel.get("profile_kind")
    if profile_kind:
        st.caption(f"Perfil: **{profile_kind}** — estágios conforme pipeline materializado em `data/bases/`.")

    plot_stages = [
        s
        for s in stages
        if isinstance(s, dict)
        and not (s.get("name") == "gestational_imputation" and not s.get("enabled", True))
    ]
    if isinstance(plot_stages, list) and plot_stages and isinstance(plot_stages[0], dict):
        stage_ids = [s.get("stage") or s.get("name", "") for s in plot_stages]
        labels = [funnel_stage_label(sid) for sid in stage_ids]
        values = [s.get("count") or s.get("rows", 0) for s in plot_stages]
    else:
        stage_ids = list(funnel.keys())
        labels = [funnel_stage_label(sid) for sid in stage_ids]
        values = [funnel[k] if isinstance(funnel[k], (int, float)) else 0 for k in stage_ids]

    fig = px.funnel(x=values, y=labels, title=title, text=values)
    fig.update_traces(texttemplate="%{x:,.0f}", textposition="inside")
    fig.update_layout(height=max(480, 60 * len(labels)))
    st.plotly_chart(fig, width="stretch", key=current_chart_key())

    if isinstance(stages, list) and stages and isinstance(stages[0], dict):
        for stage in stages:
            sid = stage.get("stage") or stage.get("name", "")
            if sid == "gestational_imputation" and not stage.get("enabled", True):
                continue
            label = funnel_stage_label(sid)
            desc = stage.get("description") or funnel_stage_caption(sid)
            parts = [f"**{label}**"]
            if desc:
                parts.append(desc)
            removed = stage.get("removed")
            if removed:
                parts.append(f"Removidas: **{format_count(removed)}** linhas.")
            details = stage.get("details") or {}
            if details:
                detail_bits = []
                if details.get("removed_under_18"):
                    detail_bits.append(
                        f"menores de 18: {format_count(details['removed_under_18'])}"
                    )
                if details.get("removed_missing_weight"):
                    detail_bits.append(
                        f"sem peso: {format_count(details['removed_missing_weight'])}"
                    )
                if details.get("removed_missing_height"):
                    detail_bits.append(
                        f"sem altura: {format_count(details['removed_missing_height'])}"
                    )
                if details.get("removed_missing_health_service"):
                    detail_bits.append(
                        f"sem uso de saúde: {format_count(details['removed_missing_health_service'])}"
                    )
                if detail_bits:
                    parts.append("Motivos (podem sobrepor): " + "; ".join(detail_bits) + ".")
            cells = stage.get("cells_imputed")
            if cells:
                parts.append(f"Células imputadas: **{format_count(cells)}**.")
            top_cols = stage.get("top_columns") or {}
            if top_cols:
                top_txt = ", ".join(f"{k} ({format_count(v)})" for k, v in list(top_cols.items())[:5])
                parts.append(f"Colunas com mais NaN preenchidos: {top_txt}.")
            binary_cols = stage.get("binary_columns")
            cat_cols = stage.get("categorical_columns")
            if binary_cols is not None or cat_cols is not None:
                parts.append(
                    f"Colunas finais: **{binary_cols or 0}** binárias, **{cat_cols or 0}** categóricas."
                )
            with st.container(border=True):
                st.markdown(" ".join(parts))


def distribution_plot(distributions: dict[str, Any], *, title: str = "Distribuição da variável") -> None:
    if not distributions:
        st.caption("Sem distribuições.")
        return
    col_names = sorted(distributions.keys())
    friendly = {c: friendly_column_name(c) for c in col_names}
    selected = st.selectbox(
        "Variável",
        col_names,
        format_func=lambda c: friendly.get(c, c),
        key="distribution_var_select",
    )
    dist = distributions[selected]
    if not isinstance(dist, dict):
        st.caption("Formato de distribuição não reconhecido.")
        return

    if dist.get("type") == "categorical":
        vals = dist.get("values", {})
        rows = [
            {"categoria": label_for_value(selected, k), "contagem": v, "codigo": k}
            for k, v in vals.items()
        ]
        dff = pd.DataFrame(rows).sort_values("contagem", ascending=False)
        total = dff["contagem"].sum()
        fig = px.bar(
            dff,
            x="categoria",
            y="contagem",
            text="contagem",
            title=f"{title}: {friendly.get(selected, selected)}",
            labels={"categoria": "Categoria", "contagem": "Contagem (n)"},
        )
        add_bar_value_labels(fig, fmt="%{y:,.0f}")
        fig.update_layout(xaxis_tickangle=-25)
        st.caption(f"Total na amostra: **{format_count(total)}** observações.")
        if selected in ("Serviço de Saúde", "Internação", "Plano de Saúde"):
            st.caption(
                "_Legenda:_ valores `0` e `1` indicam ausência/presença da condição após codificação binária."
            )
        _show(fig)
        return

    if dist.get("type") == "histogram":
        bins = dist.get("bins", [])
        counts = dist.get("counts", [])
        if len(bins) < 2 or not counts:
            st.caption("Histograma vazio.")
            return
        labels = []
        for i in range(len(counts)):
            lo, hi = bins[i], bins[i + 1]
            labels.append(f"{lo:g}–{hi:g}")
        dff = pd.DataFrame({"intervalo": labels, "contagem": counts})
        total = sum(counts)
        fig = px.bar(
            dff,
            x="intervalo",
            y="contagem",
            text="contagem",
            title=f"{title}: {friendly.get(selected, selected)}",
            labels={"intervalo": "Intervalo", "contagem": "Contagem (n)"},
        )
        add_bar_value_labels(fig, fmt="%{y:,.0f}")
        fig.update_layout(xaxis_tickangle=-35)
        st.caption(f"Total na amostra: **{format_count(total)}** observações.")
        _show(fig)
        return

    st.caption("Tipo de distribuição não suportado.")


# Alias para compatibilidade
distribution_bars = distribution_plot


def roc_plot(curves: dict[str, Any], *, title: str = "Curva ROC") -> None:
    roc = curves.get("roc") or curves
    fpr = roc.get("fpr") if isinstance(roc, dict) else None
    tpr = roc.get("tpr") if isinstance(roc, dict) else None
    if not fpr or not tpr:
        st.caption("Curva ROC não disponível.")
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash"), showlegend=False))
    auc = roc.get("auc")
    if auc is not None:
        title = f"{title} (AUC = {auc:.3f})"
    fig.update_layout(
        title=title,
        xaxis_title="Taxa de falso positivo (FPR)",
        yaxis_title="Taxa de verdadeiro positivo (TPR / sensibilidade)",
    )
    _show(fig)


def pr_plot(curves: dict[str, Any], *, title: str = "Curva precisão-recall") -> None:
    pr = curves.get("pr") or curves.get("precision_recall")
    if not pr:
        st.caption("Curva precisão-recall não disponível.")
        return
    prec, rec = pr.get("precision"), pr.get("recall")
    if not prec or not rec:
        st.caption("Curva precisão-recall não disponível.")
        return
    fig = px.line(x=rec, y=prec, title=title, labels={"x": "Recall (sensibilidade)", "y": "Precisão"})
    _show(fig)


def calibration_plot(curves: dict[str, Any], *, title: str = "Curva de calibração") -> None:
    cal = curves.get("calibration")
    if not cal:
        st.caption("Curva de calibração não disponível.")
        return
    prob = cal.get("mean_predicted_value") or cal.get("prob_pred")
    frac = cal.get("fraction_of_positives") or cal.get("frac_pos")
    if not prob or not frac:
        st.caption("Curva de calibração não disponível.")
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prob, y=frac, mode="lines+markers", name="Calibração"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash"), name="Ideal"))
    fig.update_layout(
        title=title,
        xaxis_title="Probabilidade predita média",
        yaxis_title="Fração de positivos observada",
    )
    _show(fig)


def confusion_heatmap(
    matrix: list[list[float]],
    *,
    normalize: bool = False,
    title: str = "Matriz de confusão",
) -> None:
    arr = np.array(matrix, dtype=float)
    row_sums = arr.sum(axis=1, keepdims=True)
    pct_arr = np.divide(arr, row_sums, out=np.zeros_like(arr), where=row_sums != 0) * 100
    labels = ["Negativo", "Positivo"]

    display = pct_arr if normalize else arr
    text_cells = [
        [f"{pct_arr[i, j]:.1f}%<br>({int(arr[i, j]):,})" for j in range(arr.shape[1])]
        for i in range(arr.shape[0])
    ]
    if normalize:
        text_cells = [[f"{pct_arr[i, j]:.1f}%" for j in range(arr.shape[1])] for i in range(arr.shape[0])]

    fig = go.Figure(
        data=go.Heatmap(
            z=display,
            x=labels,
            y=labels,
            text=text_cells,
            texttemplate="%{text}",
            hovertemplate=(
                "Real: %{y}<br>Predito: %{x}<br>"
                "Contagem: %{customdata[0]:,}<br>"
                "Percentual: %{customdata[1]:.1f}%"
                "<extra></extra>"
            ),
            customdata=np.stack([arr, pct_arr], axis=-1),
            colorscale="Blues",
            showscale=True,
        )
    )
    suffix = " (porcentagem por linha)" if normalize else " (contagens)"
    fig.update_layout(
        title=title + suffix,
        height=440,
        xaxis=dict(fixedrange=True, title="Predito"),
        yaxis=dict(fixedrange=True, title="Real", autorange="reversed"),
    )
    st.plotly_chart(fig, width="stretch", key=current_chart_key())


def confusion_matrix_plot(matrix: list[list[float]], *, title: str = "Matriz de confusão") -> None:
    mode = st.radio("Exibir", ["Porcentagem", "Contagem"], horizontal=True, key="cm_display_mode")
    confusion_heatmap(matrix, normalize=(mode == "Porcentagem"), title=title)


def importance_bar(df: pd.DataFrame, *, title: str = "Importância das variáveis", top_n: int = 15) -> None:
    if df.empty:
        st.caption("Sem importância de variáveis.")
        return
    feat_col = "feature" if "feature" in df.columns else df.columns[0]
    imp_col = "importance" if "importance" in df.columns else df.columns[-1]
    data = df.nlargest(top_n, imp_col) if imp_col in df.columns else df.head(top_n)
    fig = px.bar(data, x=imp_col, y=feat_col, orientation="h", title=title, text=imp_col)
    fig.update_traces(texttemplate="%{x:.3f}", textposition="outside")
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    _show(fig)


def _params_match_row(row: pd.Series, best_params: dict[str, Any]) -> bool:
    if not best_params:
        return False
    for key, val in best_params.items():
        col = key if key in row.index else f"param_{key}"
        if col not in row.index:
            continue
        row_val = row[col]
        if pd.isna(row_val) and val is None:
            continue
        if str(row_val) != str(val):
            return False
    return True


def _format_fit_time(seconds: float) -> str:
    if seconds >= 60:
        return f"{seconds / 60:.1f} min ({seconds:.1f} s)"
    return f"{seconds:.2f} s"


def gridsearch_scatter(
    df: pd.DataFrame,
    *,
    best_params: dict[str, Any] | None = None,
    title: str = "Grid Search — desempenho × tempo de treino",
) -> None:
    if df.empty:
        st.caption("Sem candidatos de Grid Search.")
        return
    score_col = "mean_test_score" if "mean_test_score" in df.columns else "score"
    time_col = "mean_fit_time" if "mean_fit_time" in df.columns else None
    if score_col not in df.columns:
        st.dataframe(df, width="stretch")
        return

    plot_df = df.copy()
    if best_params:
        plot_df["melhor"] = plot_df.apply(lambda r: _params_match_row(r, best_params), axis=1)
    else:
        plot_df["melhor"] = plot_df[score_col] == plot_df[score_col].max()

    param_cols = [c for c in plot_df.columns if c.startswith("param_")]
    hover_cols = param_cols[:8] if param_cols else list(plot_df.columns[:6])

    fig = go.Figure()
    others = plot_df[~plot_df["melhor"]]
    best = plot_df[plot_df["melhor"]]

    if time_col and time_col in plot_df.columns:
        x_col = time_col
        x_title = "Tempo médio de treino por combinação (segundos)"
        max_time = plot_df[time_col].max()
        if max_time and max_time >= 60:
            st.caption(
                f"Tempo máximo testado: **{_format_fit_time(float(max_time))}** "
                "(cada ponto = uma combinação de hiperparâmetros)."
            )
        else:
            st.caption("Cada ponto = uma combinação de hiperparâmetros testada na validação cruzada.")
    else:
        x_col = None
        x_title = "Índice da combinação"

    if not others.empty:
        fig.add_trace(
            go.Scatter(
                x=others[x_col] if x_col else list(range(len(others))),
                y=others[score_col],
                mode="markers",
                name="Outras combinações",
                marker=dict(color="rgba(148, 163, 184, 0.65)", size=9),
                customdata=others[hover_cols].values,
                hovertemplate=(
                    f"F1 (CV): %{{y:.3f}}<br>"
                    + (f"Tempo: %{{x:.2f}} s<br>" if x_col else "")
                    + "<extra>Combinação</extra>"
                ),
            )
        )

    if not best.empty:
        fig.add_trace(
            go.Scatter(
                x=best[x_col] if x_col else [0],
                y=best[score_col],
                mode="markers",
                name="Melhor combinação",
                marker=dict(color="#ca8a04", size=16, symbol="star", line=dict(width=1, color="white")),
                customdata=best[hover_cols].values,
                hovertemplate=(
                    f"<b>Melhor combinação</b><br>F1 (CV): %{{y:.3f}}<br>"
                    + (f"Tempo: %{{x:.2f}} s<br>" if x_col else "")
                    + "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title=x_title if x_col else x_title,
        yaxis_title="Desempenho na validação cruzada (F1)",
        height=560,
    )
    _show(fig)


def grouped_metric_bar(df: pd.DataFrame, x: str, y: str, color: str, *, title: str = "", as_percent: bool = False) -> None:
    if df.empty:
        st.caption("Sem dados para comparação.")
        return
    fig = px.bar(df, x=x, y=y, color=color, barmode="group", title=title, text=y)
    fmt = "%{y:.1%}" if as_percent else "%{y:.3f}"
    fig.update_traces(texttemplate=fmt, textposition="outside")
    if as_percent:
        fig.update_layout(yaxis_tickformat=".0%")
    _show(fig)


def metric_bar(labels: list[str], values: list[float], *, title: str = "", color: str | None = None, as_percent: bool = True) -> None:
    if not labels:
        st.caption("Sem métricas.")
        return
    c = color or MODEL_COLORS.get(labels[0], "#2563eb")
    display_labels = [model_label(lbl) if lbl in MODEL_COLORS else lbl for lbl in labels]
    fig = px.bar(x=display_labels, y=values, title=title, color_discrete_sequence=[c], text=values)
    fmt = "%{y:.1%}" if as_percent else "%{y:.3f}"
    fig.update_traces(texttemplate=fmt, textposition="outside")
    fig.update_layout(xaxis_tickangle=-30)
    if as_percent:
        fig.update_layout(yaxis_tickformat=".0%")
    _show(fig)
