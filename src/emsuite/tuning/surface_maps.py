"""Surface-map property helpers for tuning."""

from __future__ import annotations

import numpy as np

from .properties.fukui_spatial import build_surface_fukui_maps


def is_surface_map_property(prop: str) -> bool:
    from .properties.registry import is_surface_map_property as _is_map

    return _is_map(prop)


def split_property_lists(properties: list[str]) -> tuple[list[str], list[str]]:
    qm_props = [p for p in properties if not is_surface_map_property(p)]
    map_props = [p for p in properties if is_surface_map_property(p)]
    return qm_props, map_props


def surface_map_effects_for_point(
    map_props: list[str],
    point_index: int,
    precomputed: dict[str, np.ndarray],
    baselines: dict[str, float] | None = None,
) -> dict[str, float]:
    """Build effect dict entries for static surface-projected properties."""
    effects: dict[str, float] = {}
    baselines = baselines or {}
    for prop in map_props:
        values = precomputed.get(prop)
        if values is None or point_index >= len(values):
            continue
        value = float(values[point_index])
        baseline = baselines.get(prop, float(np.mean(values)))
        effects[f"{prop}_effect"] = value - baseline
    return effects


def precompute_surface_maps(
    neutral_mf,
    anion_mf,
    cation_mf,
    surface_coords: np.ndarray,
    map_props: list[str],
    projection: str = "nearest",
) -> dict[str, np.ndarray]:
    maps = build_surface_fukui_maps(
        neutral_mf, anion_mf, cation_mf, surface_coords, projection=projection
    )
    return {k: v for k, v in maps.items() if k in map_props}
