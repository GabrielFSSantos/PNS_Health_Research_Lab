"""Utilitários para catálogo de colunas de saída do pré-processamento."""

from __future__ import annotations

from pns_health_research.data.variable_registry import VariableRegistry, load_variable_registry
from pns_health_research.modules.preprocessing.contract import ColumnSpec, OutputFrame, PreprocessingContext


def record_column(
    ctx: PreprocessingContext,
    name: str,
    output_frame: OutputFrame,
    *,
    semantic_type: str = "categorical",
    source_column: str | None = None,
) -> None:
    """Registra metadados de uma coluna (evita duplicatas pelo nome + frame)."""
    for existing in ctx.column_catalog:
        if existing.name == name and existing.output_frame == output_frame:
            return
    ctx.column_catalog.append(
        ColumnSpec(
            name=name,
            output_frame=output_frame,
            semantic_type=semantic_type,
            source_column=source_column,
        )
    )


def _reverse_columns_map(columns_map: dict[str, str]) -> dict[str, str]:
    return {label: code for code, label in columns_map.items()}


def _infer_semantic_type(label: str, registry: VariableRegistry) -> str:
    spec = registry.get_by_label(label)
    if spec:
        return spec.semantic_type
    lowered = label.lower()
    if "peso" in lowered and "amostral" in lowered:
        return "weight"
    if label.startswith("servico_"):
        return "target"
    return "categorical"


def finalize_column_catalog(ctx: PreprocessingContext) -> list[ColumnSpec]:
    """Completa o catálogo com colunas finais e enriquece metadados."""
    registry = load_variable_registry()
    label_to_code = _reverse_columns_map(ctx.pp.columns_map or {})
    known = {(c.name, c.output_frame) for c in ctx.column_catalog}

    frames: list[tuple[OutputFrame, list[str]]] = []
    if ctx.binary_numeric is not None:
        frames.append(("binary", list(ctx.binary_numeric.columns)))
    if ctx.categorical_numeric is not None:
        frames.append(("categorical", list(ctx.categorical_numeric.columns)))

    for output_frame, columns in frames:
        for name in columns:
            if (name, output_frame) in known:
                continue
            source = label_to_code.get(name)
            semantic = _infer_semantic_type(name, registry)
            record_column(
                ctx,
                name,
                output_frame,
                semantic_type=semantic,
                source_column=source,
            )
            known.add((name, output_frame))

    return list(ctx.column_catalog)
