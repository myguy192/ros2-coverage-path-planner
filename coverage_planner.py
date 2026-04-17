from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from CFS_sim import (
    PlannerConfig,
    ScanCoverageResult,
    build_line_segment,
    build_scan_coverage_result,
    concatenate_paths,
    plan_dubins_segment,
    sample_dubins_path,
)


MIDDLE_COLUMN_FRACTION = 0.75
PATH_SAMPLE_SPACING = 0.12
GROUP_TOLERANCE = 1e-6


@dataclass
class LawnmowerBaselinePlan:
    config: PlannerConfig
    scan_result: ScanCoverageResult
    all_stop_points: np.ndarray
    selected_indices: np.ndarray
    selected_points: np.ndarray
    column_ids: np.ndarray
    selected_column_ids: np.ndarray
    selected_column_xs: np.ndarray
    common_y_min: float
    common_y_max: float
    column_spacing: float
    serviced_column_step: int
    skipped_columns_per_turn: int
    first_pass_indices: np.ndarray
    second_pass_indices: np.ndarray
    visit_order: np.ndarray
    ordered_points: np.ndarray
    first_pass_path: np.ndarray
    second_pass_path: np.ndarray
    full_path: np.ndarray


def _column_ids(points: np.ndarray, grid_size: float) -> np.ndarray:
    if len(points) == 0:
        return np.empty(0, dtype=int)
    return np.rint(points[:, 0] / grid_size).astype(int)


def _group_points_by_column(indices: np.ndarray, points: np.ndarray, column_ids: np.ndarray) -> dict[int, np.ndarray]:
    grouped: dict[int, np.ndarray] = {}
    for column_id in np.unique(column_ids[indices]):
        grouped[int(column_id)] = indices[column_ids[indices] == column_id]
    return grouped


def _select_middle_columns(column_ids: np.ndarray, fraction: float) -> np.ndarray:
    unique_columns = np.unique(column_ids)
    if len(unique_columns) == 0:
        return np.empty(0, dtype=int)

    keep_count = max(1, int(np.ceil(len(unique_columns) * fraction)))
    keep_count = min(keep_count, len(unique_columns))
    start = (len(unique_columns) - keep_count) // 2
    end = start + keep_count
    return unique_columns[start:end]


def _compute_common_y_band(points: np.ndarray, selected_indices: np.ndarray, selected_column_ids: np.ndarray, column_ids: np.ndarray) -> tuple[float, float]:
    column_minima: list[float] = []
    column_maxima: list[float] = []

    for column_id in selected_column_ids:
        column_points = points[selected_indices[column_ids[selected_indices] == column_id]]
        column_minima.append(float(np.min(column_points[:, 1])))
        column_maxima.append(float(np.max(column_points[:, 1])))

    return max(column_minima), min(column_maxima)


def _compute_min_feasible_column_step(column_spacing: float, min_turn_radius: float) -> int:
    if column_spacing <= 1e-9:
        return 1
    return max(1, int(np.ceil((2.0 * min_turn_radius) / column_spacing)))


def _build_subpass_column_order(selected_column_ids: np.ndarray, serviced_step: int) -> list[np.ndarray]:
    if len(selected_column_ids) == 0:
        return []

    ordered_subpasses: list[np.ndarray] = []
    ordered_residues = [0] + list(range(serviced_step - 1, 0, -1))

    for subpass_idx, residue in enumerate(ordered_residues):
        local_columns = selected_column_ids[np.mod(selected_column_ids - selected_column_ids[0], serviced_step) == residue]
        if len(local_columns) == 0:
            continue
        if subpass_idx % 2 == 1:
            local_columns = local_columns[::-1]
        ordered_subpasses.append(local_columns)

    return ordered_subpasses


def _build_subpass_stop_order(
    subpass_columns: np.ndarray,
    grouped_selected: dict[int, np.ndarray],
    points: np.ndarray,
    start_upward: bool,
) -> np.ndarray:
    ordered_indices: list[int] = []

    for local_idx, column_id in enumerate(subpass_columns):
        column_indices = grouped_selected[int(column_id)]
        column_points = points[column_indices]
        sort_order = np.argsort(column_points[:, 1])
        sorted_indices = column_indices[sort_order]

        upward = start_upward if local_idx % 2 == 0 else not start_upward
        if not upward:
            sorted_indices = sorted_indices[::-1]

        ordered_indices.extend(sorted_indices.tolist())

    return np.asarray(ordered_indices, dtype=int)


def _path_headings_for_visit_order(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.empty(0, dtype=float)
    if len(points) == 1:
        return np.array([0.0], dtype=float)

    headings = np.zeros(len(points), dtype=float)
    for idx in range(len(points)):
        if idx == 0:
            delta = points[1] - points[0]
        elif idx == len(points) - 1:
            delta = points[-1] - points[-2]
        else:
            delta = points[idx + 1] - points[idx - 1]
        headings[idx] = float(np.arctan2(delta[1], delta[0]))
    return headings


def _build_dubins_connector(
    start_point: np.ndarray,
    start_heading: float,
    end_point: np.ndarray,
    end_heading: float,
    config: PlannerConfig,
    spacing: float,
) -> np.ndarray:
    planned = plan_dubins_segment(start_point, start_heading, end_point, end_heading, config.min_turn_radius)
    if planned is None:
        return build_line_segment(start_point, end_point, spacing)

    mode, lengths = planned
    connector = sample_dubins_path(
        start_point,
        start_heading,
        mode,
        lengths,
        config.min_turn_radius,
        spacing,
    )
    connector[0] = np.asarray(start_point, dtype=float)
    connector[-1] = np.asarray(end_point, dtype=float)
    return connector


def _build_path_from_ordered_points(points: np.ndarray, config: PlannerConfig, spacing: float) -> np.ndarray:
    if len(points) == 0:
        return np.empty((0, 2), dtype=float)
    if len(points) == 1:
        return points.copy()

    headings = _path_headings_for_visit_order(points)
    chunks: list[np.ndarray] = [points[0][None, :]]

    for idx in range(len(points) - 1):
        start_point = points[idx]
        end_point = points[idx + 1]

        same_column = abs(start_point[0] - end_point[0]) <= GROUP_TOLERANCE
        if same_column:
            segment = build_line_segment(start_point, end_point, spacing)
        else:
            segment = _build_dubins_connector(
                start_point,
                headings[idx],
                end_point,
                headings[idx + 1],
                config,
                spacing,
            )
        chunks.append(segment)

    return concatenate_paths(chunks)


def build_lawnmower_baseline_plan(
    config: PlannerConfig | None = None,
    middle_column_fraction: float = MIDDLE_COLUMN_FRACTION,
    path_sample_spacing: float = PATH_SAMPLE_SPACING,
) -> LawnmowerBaselinePlan:
    config = PlannerConfig() if config is None else config
    scan_result = build_scan_coverage_result(config)
    all_stop_points = scan_result.stop_points

    if len(all_stop_points) == 0:
        empty = np.empty((0, 2), dtype=float)
        empty_idx = np.empty(0, dtype=int)
        return LawnmowerBaselinePlan(
            config=config,
            scan_result=scan_result,
            all_stop_points=all_stop_points,
            selected_indices=empty_idx,
            selected_points=empty,
            column_ids=empty_idx,
            selected_column_ids=empty_idx,
            selected_column_xs=np.empty(0, dtype=float),
            common_y_min=0.0,
            common_y_max=0.0,
            column_spacing=config.grid_size,
            serviced_column_step=1,
            skipped_columns_per_turn=0,
            first_pass_indices=empty_idx,
            second_pass_indices=empty_idx,
            visit_order=empty_idx,
            ordered_points=empty,
            first_pass_path=empty,
            second_pass_path=empty,
            full_path=empty,
        )

    column_ids = _column_ids(all_stop_points, config.grid_size)
    selected_column_ids = _select_middle_columns(column_ids, middle_column_fraction)
    prelim_selected_indices = np.flatnonzero(np.isin(column_ids, selected_column_ids))
    common_y_min, common_y_max = _compute_common_y_band(
        all_stop_points,
        prelim_selected_indices,
        selected_column_ids,
        column_ids,
    )

    band_mask = (
        np.isin(column_ids, selected_column_ids)
        & (all_stop_points[:, 1] >= common_y_min - GROUP_TOLERANCE)
        & (all_stop_points[:, 1] <= common_y_max + GROUP_TOLERANCE)
    )
    selected_indices = np.flatnonzero(band_mask)
    selected_points = all_stop_points[selected_indices]

    grouped_selected = _group_points_by_column(selected_indices, all_stop_points, column_ids)
    selected_column_ids = np.asarray(sorted(grouped_selected.keys()), dtype=int)
    selected_column_xs = selected_column_ids.astype(float) * config.grid_size

    if len(selected_column_xs) > 1:
        column_spacing = float(np.min(np.diff(selected_column_xs)))
    else:
        column_spacing = config.grid_size

    serviced_column_step = _compute_min_feasible_column_step(column_spacing, config.min_turn_radius)
    skipped_columns_per_turn = max(0, serviced_column_step - 1)

    subpasses = _build_subpass_column_order(selected_column_ids, serviced_column_step)
    first_pass_indices = (
        _build_subpass_stop_order(subpasses[0], grouped_selected, all_stop_points, start_upward=True)
        if subpasses
        else np.empty(0, dtype=int)
    )

    second_pass_chunks: list[np.ndarray] = []
    for subpass_idx, subpass_columns in enumerate(subpasses[1:], start=1):
        start_upward = bool(subpass_idx % 2 == 1)
        second_pass_chunks.append(
            _build_subpass_stop_order(subpass_columns, grouped_selected, all_stop_points, start_upward=start_upward)
        )

    second_pass_indices = (
        np.concatenate(second_pass_chunks) if second_pass_chunks else np.empty(0, dtype=int)
    )
    visit_order = np.concatenate([first_pass_indices, second_pass_indices]) if len(selected_indices) else np.empty(0, dtype=int)
    ordered_points = all_stop_points[visit_order] if len(visit_order) else np.empty((0, 2), dtype=float)

    first_pass_points = all_stop_points[first_pass_indices] if len(first_pass_indices) else np.empty((0, 2), dtype=float)
    second_pass_points = all_stop_points[second_pass_indices] if len(second_pass_indices) else np.empty((0, 2), dtype=float)

    first_pass_path = _build_path_from_ordered_points(first_pass_points, config, path_sample_spacing)
    second_pass_path = _build_path_from_ordered_points(second_pass_points, config, path_sample_spacing)

    full_path = concatenate_paths([first_pass_path, second_pass_path])

    return LawnmowerBaselinePlan(
        config=config,
        scan_result=scan_result,
        all_stop_points=all_stop_points,
        selected_indices=selected_indices,
        selected_points=selected_points,
        column_ids=column_ids,
        selected_column_ids=selected_column_ids,
        selected_column_xs=selected_column_xs,
        common_y_min=float(common_y_min),
        common_y_max=float(common_y_max),
        column_spacing=float(column_spacing),
        serviced_column_step=int(serviced_column_step),
        skipped_columns_per_turn=int(skipped_columns_per_turn),
        first_pass_indices=first_pass_indices,
        second_pass_indices=second_pass_indices,
        visit_order=visit_order,
        ordered_points=ordered_points,
        first_pass_path=first_pass_path,
        second_pass_path=second_pass_path,
        full_path=full_path,
    )


def validate_lawnmower_baseline_plan(plan: LawnmowerBaselinePlan) -> dict[str, object]:
    selected_stop_hits = [
        np.any(np.linalg.norm(plan.full_path - stop, axis=1) <= 1e-9) for stop in plan.selected_points
    ]

    return {
        "all_selected_middle_points_explicitly_visited": bool(np.all(selected_stop_hits)),
        "missing_selected_point_count": int(np.count_nonzero(~np.asarray(selected_stop_hits, dtype=bool))),
        "selected_point_count": int(len(plan.selected_points)),
        "selected_column_count": int(len(plan.selected_column_ids)),
        "first_pass_point_count": int(len(plan.first_pass_indices)),
        "second_pass_point_count": int(len(plan.second_pass_indices)),
        "minimum_turn_based_column_step": int(plan.serviced_column_step),
        "skipped_columns_per_turn": int(plan.skipped_columns_per_turn),
    }


def print_plan_summary(plan: LawnmowerBaselinePlan) -> None:
    validation = validate_lawnmower_baseline_plan(plan)
    print(f"Total required stop points: {len(plan.all_stop_points)}")
    print(f"Number of points selected for this middle-region baseline: {validation['selected_point_count']}")
    print(f"Number of columns selected: {validation['selected_column_count']}")
    print(f"Number of points visited by the first pass: {validation['first_pass_point_count']}")
    print(f"Number of points visited by the second pass: {validation['second_pass_point_count']}")
    print(
        "Whether all selected middle-region points were explicitly visited: "
        f"{validation['all_selected_middle_points_explicitly_visited']}"
    )
    print(f"Minimum feasible serviced-column step: {validation['minimum_turn_based_column_step']}")
    print(f"Skipped columns per turn: {validation['skipped_columns_per_turn']}")


def plot_lawnmower_baseline_plan(plan: LawnmowerBaselinePlan, arrow_stride: int = 45) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    config = plan.config
    fig, ax = plt.subplots(figsize=(10, 10))

    ax.add_patch(Circle((0.0, 0.0), config.tank_radius, fill=False, color="black", linewidth=2.0))

    if len(plan.all_stop_points):
        ax.scatter(
            plan.all_stop_points[:, 0],
            plan.all_stop_points[:, 1],
            s=18,
            color="#9ca3af",
            alpha=0.35,
            label="All required stop points",
            zorder=2,
        )

    if len(plan.selected_points):
        ax.scatter(
            plan.selected_points[:, 0],
            plan.selected_points[:, 1],
            s=32,
            color="#1d4ed8",
            alpha=0.9,
            label="Selected middle-region points",
            zorder=4,
        )

    if len(plan.first_pass_path):
        ax.plot(
            plan.first_pass_path[:, 0],
            plan.first_pass_path[:, 1],
            color="#111827",
            linewidth=2.0,
            label="First pass",
            zorder=3,
        )

    if len(plan.second_pass_path):
        ax.plot(
            plan.second_pass_path[:, 0],
            plan.second_pass_path[:, 1],
            color="#10b981",
            linewidth=2.0,
            label="Second pass",
            zorder=3,
        )

    if len(plan.first_pass_indices):
        first_pass_points = plan.all_stop_points[plan.first_pass_indices]
        ax.scatter(
            first_pass_points[:, 0],
            first_pass_points[:, 1],
            s=24,
            color="#111827",
            alpha=0.8,
            label="First-pass serviced points",
            zorder=5,
        )

    if len(plan.second_pass_indices):
        second_pass_points = plan.all_stop_points[plan.second_pass_indices]
        ax.scatter(
            second_pass_points[:, 0],
            second_pass_points[:, 1],
            s=24,
            color="#059669",
            alpha=0.9,
            label="Second-pass serviced points",
            zorder=5,
        )

    if len(plan.full_path) > 1:
        arrow_indices = np.arange(0, len(plan.full_path) - 1, max(1, arrow_stride))
        deltas = plan.full_path[arrow_indices + 1] - plan.full_path[arrow_indices]
        ax.quiver(
            plan.full_path[arrow_indices, 0],
            plan.full_path[arrow_indices, 1],
            deltas[:, 0],
            deltas[:, 1],
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.0035,
            color="#f97316",
            zorder=6,
            label="Travel direction",
        )

    ax.axhline(plan.common_y_min, color="#d97706", linestyle=":", linewidth=1.0, alpha=0.8)
    ax.axhline(plan.common_y_max, color="#d97706", linestyle=":", linewidth=1.0, alpha=0.8)
    if len(plan.selected_column_xs):
        ax.axvline(plan.selected_column_xs[0], color="#7c3aed", linestyle="--", linewidth=1.0, alpha=0.6)
        ax.axvline(plan.selected_column_xs[-1], color="#7c3aed", linestyle="--", linewidth=1.0, alpha=0.6)

    ax.scatter(0.0, 0.0, s=40, color="black", label="Tank center", zorder=7)
    ax.set_aspect("equal", adjustable="box")
    margin = config.grid_size * 1.5
    ax.set_xlim(-(config.tank_radius + margin), config.tank_radius + margin)
    ax.set_ylim(-(config.tank_radius + margin), config.tank_radius + margin)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Middle-region baseline lawnmower planner")
    ax.grid(True, alpha=0.22)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    default_plan = build_lawnmower_baseline_plan()
    print_plan_summary(default_plan)
    print("Validation:")
    for key, value in validate_lawnmower_baseline_plan(default_plan).items():
        print(f"  {key}: {value}")
    plot_lawnmower_baseline_plan(default_plan)
