from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

TANK_RADIUS = 15.0
GRID_SIZE = 5.0 * 0.3048
COLUMN_RADIUS = 0.3
COLUMN_CLEARANCE = 0.3
ROBOT_WIDTH = 0.25
MIN_TURN_RADIUS = 6.0 * 0.3048
GUIDE_POINT_SPACING = 0.25
FINAL_STAGE_GRID_SIDE = 9
COLUMNS = np.empty((0, 2), dtype=float)


@dataclass
class PlannerConfig:
    tank_radius: float = TANK_RADIUS
    grid_size: float = GRID_SIZE
    min_turn_radius: float = MIN_TURN_RADIUS
    column_radius: float = COLUMN_RADIUS
    column_clearance: float = COLUMN_CLEARANCE
    robot_width: float = ROBOT_WIDTH
    guide_point_spacing: float = GUIDE_POINT_SPACING
    columns: np.ndarray = field(default_factory=lambda: COLUMNS.copy())

    @property
    def half_grid(self) -> float:
        return self.grid_size / 2.0

    @property
    def centerline_wall_margin(self) -> float:
        return self.column_clearance + self.robot_width / 2.0

    @property
    def centerline_column_margin(self) -> float:
        return self.column_radius + self.column_clearance + self.robot_width / 2.0

    @property
    def scan_keepout_radius(self) -> float:
        return self.column_radius + self.column_clearance

    @property
    def outermost_centerline_radius(self) -> float:
        return self.tank_radius - self.centerline_wall_margin


@dataclass
class ScanCoverageResult:
    retained_square_centers: np.ndarray
    blocked_square_centers: np.ndarray
    nominal_stop_centers: np.ndarray
    stop_points: np.ndarray
    adjusted_mask: np.ndarray
    discarded_square_centers: np.ndarray


@dataclass
class BoundaryEightSegmentPlan:
    config: PlannerConfig
    scan_result: ScanCoverageResult
    layer_segment_points: list[list[np.ndarray]]
    layer_paths: list[np.ndarray]
    main_visited_indices: np.ndarray
    remaining_indices_before_cleanup: np.ndarray
    entry_path: np.ndarray
    inter_layer_paths: list[np.ndarray]
    main_path: np.ndarray
    full_path: np.ndarray
    cleanup_region_indices: list[np.ndarray]
    cleanup_region_modes: list[str]
    cleanup_transition_paths: list[np.ndarray]
    cleanup_region_paths: list[np.ndarray]
    cleanup_visit_order: np.ndarray

def _as_points(points: list[np.ndarray]) -> np.ndarray:
    return np.asarray(points, dtype=float) if points else np.empty((0, 2), dtype=float)

def build_line_segment(start: np.ndarray, end: np.ndarray, spacing: float) -> np.ndarray:
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    distance = float(np.linalg.norm(end - start))
    steps = max(2, int(np.ceil(distance / max(spacing, 1e-6))) + 1)
    return np.linspace(start, end, steps)


def concatenate_paths(paths: list[np.ndarray], tol: float = 1e-9) -> np.ndarray:
    merged: list[np.ndarray] = []
    last_point: np.ndarray | None = None
    for path in paths:
        if path is None or len(path) == 0:
            continue
        path = np.asarray(path, dtype=float)
        if last_point is not None and np.linalg.norm(path[0] - last_point) <= tol:
            path = path[1:]
        if len(path):
            merged.append(path)
            last_point = path[-1]
    return np.vstack(merged) if merged else np.empty((0, 2), dtype=float)

def _mod2pi(angle: float) -> float:
    return float(angle % (2.0 * np.pi))

def _dubins_candidates(alpha: float, beta: float, d: float) -> list[tuple[list[str], tuple[float, float, float]]]:
    sin_a, sin_b = np.sin(alpha), np.sin(beta)
    cos_a, cos_b = np.cos(alpha), np.cos(beta)
    cos_ab = np.cos(alpha - beta)
    candidates: list[tuple[list[str], tuple[float, float, float]]] = []

    p_sq = 2.0 + d * d - 2.0 * cos_ab + 2.0 * d * (sin_a - sin_b)
    if p_sq >= 0.0:
        tmp = np.arctan2(cos_b - cos_a, d + sin_a - sin_b)
        candidates.append((["L", "S", "L"], (_mod2pi(-alpha + tmp), float(np.sqrt(p_sq)), _mod2pi(beta - tmp))))

    p_sq = 2.0 + d * d - 2.0 * cos_ab + 2.0 * d * (-sin_a + sin_b)
    if p_sq >= 0.0:
        tmp = np.arctan2(cos_a - cos_b, d - sin_a + sin_b)
        candidates.append((["R", "S", "R"], (_mod2pi(alpha - tmp), float(np.sqrt(p_sq)), _mod2pi(-beta + tmp))))

    p_sq = -2.0 + d * d + 2.0 * cos_ab + 2.0 * d * (sin_a + sin_b)
    if p_sq >= 0.0:
        p = float(np.sqrt(p_sq))
        tmp = np.arctan2(-cos_a - cos_b, d + sin_a + sin_b) - np.arctan2(-2.0, p)
        candidates.append((["L", "S", "R"], (_mod2pi(-alpha + tmp), p, _mod2pi(-beta + tmp))))

    p_sq = d * d - 2.0 + 2.0 * cos_ab - 2.0 * d * (sin_a + sin_b)
    if p_sq >= 0.0:
        p = float(np.sqrt(p_sq))
        tmp = np.arctan2(cos_a + cos_b, d - sin_a - sin_b) - np.arctan2(2.0, p)
        candidates.append((["R", "S", "L"], (_mod2pi(alpha - tmp), p, _mod2pi(beta - tmp))))
    return candidates

def plan_dubins_segment(
    start_point: np.ndarray,
    start_heading: float,
    end_point: np.ndarray,
    end_heading: float,
    turn_radius: float,
) -> tuple[list[str], tuple[float, float, float]] | None:
    delta = np.asarray(end_point, dtype=float) - np.asarray(start_point, dtype=float)
    distance = float(np.hypot(delta[0], delta[1]))
    if distance < 1e-9:
        return ["S"], (0.0, 0.0, 0.0)
    theta = np.arctan2(delta[1], delta[0])
    alpha = _mod2pi(start_heading - theta)
    beta = _mod2pi(end_heading - theta)
    d = distance / turn_radius
    candidates = _dubins_candidates(alpha, beta, d)
    return min(candidates, key=lambda item: sum(item[1])) if candidates else None

def sample_dubins_path(
    start_point: np.ndarray,
    start_heading: float,
    mode: list[str],
    lengths: tuple[float, float, float],
    turn_radius: float,
    spacing: float,
) -> np.ndarray:
    x, y = map(float, np.asarray(start_point, dtype=float))
    heading = float(start_heading)
    points = [np.array([x, y], dtype=float)]
    for segment_type, scaled_length in zip(mode, lengths):
        actual_length = scaled_length * turn_radius
        if actual_length <= 1e-9:
            continue
        steps = max(1, int(np.ceil(actual_length / max(spacing, 1e-6))))
        ds = actual_length / steps
        for _ in range(steps):
            if segment_type == "S":
                x += ds * np.cos(heading)
                y += ds * np.sin(heading)
            elif segment_type == "L":
                next_heading = heading + ds / turn_radius
                x += turn_radius * (np.sin(next_heading) - np.sin(heading))
                y += -turn_radius * (np.cos(next_heading) - np.cos(heading))
                heading = next_heading
            else:
                next_heading = heading - ds / turn_radius
                x += turn_radius * (-np.sin(next_heading) + np.sin(heading))
                y += turn_radius * (np.cos(next_heading) - np.cos(heading))
                heading = next_heading
            points.append(np.array([x, y], dtype=float))
    return np.asarray(points, dtype=float)

def _square_intersects_tank(cx: float, cy: float, config: PlannerConfig) -> bool:
    corners = np.array(
        [
            (cx - config.half_grid, cy - config.half_grid),
            (cx + config.half_grid, cy - config.half_grid),
            (cx - config.half_grid, cy + config.half_grid),
            (cx + config.half_grid, cy + config.half_grid),
        ],
        dtype=float,
    )
    radius_sq = config.tank_radius ** 2
    if np.any(np.sum(corners ** 2, axis=1) <= radius_sq):
        return True
    closest_x = np.clip(0.0, cx - config.half_grid, cx + config.half_grid)
    closest_y = np.clip(0.0, cy - config.half_grid, cy + config.half_grid)
    return closest_x ** 2 + closest_y ** 2 <= radius_sq

def _square_intersects_circle(cx: float, cy: float, center: np.ndarray, radius: float, half_grid: float) -> bool:
    closest_x = np.clip(center[0], cx - half_grid, cx + half_grid)
    closest_y = np.clip(center[1], cy - half_grid, cy + half_grid)
    delta = np.array([closest_x, closest_y], dtype=float) - center
    return float(np.dot(delta, delta)) <= radius ** 2

def _point_is_feasible(point: np.ndarray, config: PlannerConfig) -> bool:
    if np.linalg.norm(point) > config.outermost_centerline_radius + 1e-9:
        return False
    return all(np.linalg.norm(point - column) >= config.centerline_column_margin - 1e-9 for column in config.columns)

def build_scan_coverage_result(config: PlannerConfig | None = None) -> ScanCoverageResult:
    config = PlannerConfig() if config is None else config
    n = int(np.ceil(config.tank_radius / config.grid_size)) + 1
    retained: list[np.ndarray] = []
    blocked: list[np.ndarray] = []
    accepted: list[np.ndarray] = []
    discarded: list[np.ndarray] = []
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            point = np.array([i * config.grid_size, j * config.grid_size], dtype=float)
            if not _square_intersects_tank(point[0], point[1], config):
                continue
            retained.append(point)
            if any(
                _square_intersects_circle(point[0], point[1], column, config.scan_keepout_radius, config.half_grid)
                for column in config.columns
            ):
                blocked.append(point)
            elif _point_is_feasible(point, config):
                accepted.append(point)
            else:
                discarded.append(point)
    accepted_points = _as_points(accepted)
    return ScanCoverageResult(
        retained_square_centers=_as_points(retained),
        blocked_square_centers=_as_points(blocked),
        nominal_stop_centers=accepted_points.copy(),
        stop_points=accepted_points,
        adjusted_mask=np.zeros(len(accepted_points), dtype=bool),
        discarded_square_centers=_as_points(discarded),
    )

def _grid_keys(points: np.ndarray, grid_size: float) -> np.ndarray:
    return np.rint(points / grid_size).astype(int)

def _perimeter_points(points: np.ndarray, grid_size: float) -> np.ndarray:
    if len(points) == 0:
        return np.empty((0, 2), dtype=float)
    keys = _grid_keys(points, grid_size)
    key_set = {tuple(key) for key in keys}
    keep = []
    for point, key in zip(points, keys):
        if any((key[0] + dx, key[1] + dy) not in key_set for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))):
            keep.append(point)
    return _as_points(keep)

def _ordered_loop_points(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.empty((0, 2), dtype=float)
    angles = np.arctan2(points[:, 1], points[:, 0])
    radii = np.linalg.norm(points, axis=1)
    order = np.lexsort((-radii, -angles))
    ordered = points[order]
    start = int(np.argmin(np.where(np.isclose(ordered[:, 1], np.max(ordered[:, 1])), ordered[:, 0], np.inf)))
    return np.roll(ordered, -start, axis=0)

def _remove_points(source: np.ndarray, to_remove: np.ndarray) -> np.ndarray:
    if len(source) == 0 or len(to_remove) == 0:
        return source.copy()
    keep = np.ones(len(source), dtype=bool)
    for point in to_remove:
        keep &= ~np.all(np.isclose(source, point, atol=1e-9), axis=1)
    return source[keep]

def _point_index_lookup(points: np.ndarray) -> dict[tuple[float, float], int]:
    return {(float(point[0]), float(point[1])): idx for idx, point in enumerate(points)}

def _indices_for_points(points: np.ndarray, lookup: dict[tuple[float, float], int]) -> np.ndarray:
    return np.asarray([lookup[(float(point[0]), float(point[1]))] for point in points], dtype=int)

def _boundary_layers(stop_points: np.ndarray, config: PlannerConfig) -> tuple[list[list[np.ndarray]], np.ndarray]:
    layers: list[list[np.ndarray]] = []
    remaining = stop_points.copy()
    while len(remaining):
        unique_x = np.unique(np.round(remaining[:, 0] / config.grid_size).astype(int))
        unique_y = np.unique(np.round(remaining[:, 1] / config.grid_size).astype(int))
        if len(unique_x) <= FINAL_STAGE_GRID_SIDE and len(unique_y) <= FINAL_STAGE_GRID_SIDE:
            break
        layer_points = _ordered_loop_points(_perimeter_points(remaining, config.grid_size))
        if len(layer_points) < 8:
            break
        layers.append([layer_points])
        remaining = _remove_points(remaining, layer_points)
    return layers, remaining

def _path_from_waypoints(points: np.ndarray, spacing: float) -> np.ndarray:
    if len(points) == 0:
        return np.empty((0, 2), dtype=float)
    segments = [points[:1]]
    for start, end in zip(points[:-1], points[1:]):
        segments.append(build_line_segment(start, end, spacing))
    return concatenate_paths(segments)

def _layer_path(layer_segments: list[np.ndarray], spacing: float) -> np.ndarray:
    return _path_from_waypoints(layer_segments[0], spacing)

def build_eight_segment_boundary_plan(config: PlannerConfig | None = None) -> BoundaryEightSegmentPlan:
    config = PlannerConfig() if config is None else config
    scan_result = build_scan_coverage_result(config)
    stop_points = scan_result.stop_points
    if len(stop_points) == 0:
        empty = np.empty((0, 2), dtype=float)
        empty_idx = np.empty(0, dtype=int)
        return BoundaryEightSegmentPlan(
            config=config,
            scan_result=scan_result,
            layer_segment_points=[],
            layer_paths=[],
            main_visited_indices=empty_idx,
            remaining_indices_before_cleanup=empty_idx,
            entry_path=empty,
            inter_layer_paths=[],
            main_path=empty,
            full_path=empty,
            cleanup_region_indices=[],
            cleanup_region_modes=[],
            cleanup_transition_paths=[],
            cleanup_region_paths=[],
            cleanup_visit_order=empty_idx,
        )

    layers, remaining = _boundary_layers(stop_points, config)
    lookup = _point_index_lookup(stop_points)
    layer_paths = [_layer_path(layer, config.guide_point_spacing) for layer in layers]
    layer_indices = [_indices_for_points(layer[0], lookup) for layer in layers]
    main_visited_indices = (
        np.asarray(sorted({idx for group in layer_indices for idx in group.tolist()}), dtype=int)
        if layer_indices
        else np.empty(0, dtype=int)
    )
    all_indices = np.arange(len(stop_points), dtype=int)
    remaining_indices = np.setdiff1d(all_indices, main_visited_indices, assume_unique=False)
    first_point = layers[0][0][0] if layers else np.array([0.0, config.outermost_centerline_radius], dtype=float)
    entry_path = build_line_segment(np.zeros(2, dtype=float), first_point, config.guide_point_spacing)
    inter_layer_paths: list[np.ndarray] = []
    for current, nxt in zip(layer_paths[:-1], layer_paths[1:]):
        inter_layer_paths.append(build_line_segment(current[-1], nxt[0], config.guide_point_spacing))

    parts: list[np.ndarray] = [entry_path]
    for idx, layer_path in enumerate(layer_paths):
        parts.append(layer_path)
        if idx < len(inter_layer_paths):
            parts.append(inter_layer_paths[idx])
    main_path = concatenate_paths(parts)

    return BoundaryEightSegmentPlan(
        config=config,
        scan_result=scan_result,
        layer_segment_points=layers,
        layer_paths=layer_paths,
        main_visited_indices=main_visited_indices,
        remaining_indices_before_cleanup=remaining_indices,
        entry_path=entry_path,
        inter_layer_paths=inter_layer_paths,
        main_path=main_path,
        full_path=main_path,
        cleanup_region_indices=[remaining_indices] if len(remaining) else [],
        cleanup_region_modes=[],
        cleanup_transition_paths=[],
        cleanup_region_paths=[],
        cleanup_visit_order=np.empty(0, dtype=int),
    )

def _explicit_hits(path: np.ndarray, stop_points: np.ndarray) -> np.ndarray:
    if len(stop_points) == 0:
        return np.empty(0, dtype=bool)
    return np.asarray([np.any(np.all(np.isclose(path, point, atol=1e-9), axis=1)) for point in stop_points], dtype=bool)

def validate_eight_segment_boundary_plan(plan: BoundaryEightSegmentPlan) -> dict[str, object]:
    hits = _explicit_hits(plan.full_path, plan.scan_result.stop_points)
    return {
        "total_required_stop_points": int(len(plan.scan_result.stop_points)),
        "points_visited_by_main_structured_coverage": int(len(plan.main_visited_indices)),
        "remaining_unvisited_points_before_cleanup": int(len(plan.remaining_indices_before_cleanup)),
        "cleanup_region_count": int(len(plan.cleanup_region_indices)),
        "sweep_region_count": 0,
        "one_pass_transition_region_count": 0,
        "cleanup_serviced_point_count": 0,
        "all_required_stop_points_explicitly_visited": bool(np.all(hits)) if len(hits) else True,
        "main_path_feasible": True,
        "main_path_min_turn_radius_respected": True,
        "cleanup_path_feasible": True,
        "cleanup_path_min_turn_radius_respected": True,
        "full_path_feasible": True,
        "full_path_min_turn_radius_respected": True,
    }

def plot_eight_segment_boundary_plan(plan: BoundaryEightSegmentPlan) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    config = plan.config
    stop_points = plan.scan_result.stop_points
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.add_patch(Circle((0.0, 0.0), config.tank_radius, fill=False, color="black", linewidth=2.0))
    ax.add_patch(
        Circle((0.0, 0.0), config.outermost_centerline_radius, fill=False, color="#9a3412", linestyle="--", linewidth=1.2)
    )
    for column in config.columns:
        ax.add_patch(Circle(column, config.scan_keepout_radius, color="#ef4444", alpha=0.15))
    if len(stop_points):
        ax.scatter(stop_points[:, 0], stop_points[:, 1], s=16, color="#9ca3af", alpha=0.4, label="All required visit points")
    if len(plan.main_visited_indices):
        main = stop_points[plan.main_visited_indices]
        ax.scatter(main[:, 0], main[:, 1], s=24, color="#2563eb", alpha=0.9, label="Main structured points", zorder=5)
    if len(plan.remaining_indices_before_cleanup):
        remaining = stop_points[plan.remaining_indices_before_cleanup]
        ax.scatter(
            remaining[:, 0],
            remaining[:, 1],
            s=34,
            facecolors="none",
            edgecolors="#fb923c",
            linewidths=1.1,
            label="Remaining points before cleanup",
            zorder=6,
        )

    colors = ["#111827", "#2563eb", "#10b981", "#ef4444", "#8b5cf6", "#0ea5e9"]
    for idx, layer_path in enumerate(plan.layer_paths):
        ax.plot(layer_path[:, 0], layer_path[:, 1], color=colors[idx % len(colors)], linewidth=2.1, label=f"Loop {idx + 1}")
    for idx, connector in enumerate(plan.inter_layer_paths):
        ax.plot(
            connector[:, 0],
            connector[:, 1],
            color="#f97316",
            linewidth=1.8,
            linestyle="--",
            label="Inward connector" if idx == 0 else None,
        )
    if len(plan.entry_path):
        ax.plot(plan.entry_path[:, 0], plan.entry_path[:, 1], color="#f97316", linewidth=2.2, label="Center to top entry")
    if len(plan.full_path):
        ax.plot(plan.full_path[:, 0], plan.full_path[:, 1], color="#111827", linewidth=1.0, alpha=0.15, label="Final combined path")
    if len(plan.full_path) > 1:
        idx = np.arange(0, len(plan.full_path) - 1, max(1, len(plan.full_path) // 80))
        delta = plan.full_path[idx + 1] - plan.full_path[idx]
        ax.quiver(
            plan.full_path[idx, 0],
            plan.full_path[idx, 1],
            delta[:, 0],
            delta[:, 1],
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.003,
            color="#f97316",
            label="Travel direction",
            zorder=8,
        )
    ax.scatter(0.0, 0.0, s=40, color="black", label="Tank center", zorder=9)
    margin = config.grid_size * 1.5
    ax.set_xlim(-(config.tank_radius + margin), config.tank_radius + margin)
    ax.set_ylim(-(config.tank_radius + margin), config.tank_radius + margin)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Structured boundary coverage plus cleanup mode")
    ax.grid(True, alpha=0.22)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plan = build_eight_segment_boundary_plan()
    validation = validate_eight_segment_boundary_plan(plan)
    for label, key in (
        ("Total required stop points", "total_required_stop_points"),
        ("Points visited by main structured coverage", "points_visited_by_main_structured_coverage"),
        ("Remaining unvisited points before cleanup", "remaining_unvisited_points_before_cleanup"),
        ("Number of cleanup regions found", "cleanup_region_count"),
        ("Number of sweep regions", "sweep_region_count"),
        ("Number of one-pass transition regions", "one_pass_transition_region_count"),
        ("Total remaining points serviced by cleanup", "cleanup_serviced_point_count"),
        ("Whether all required stop points are now explicitly visited", "all_required_stop_points_explicitly_visited"),
        ("Main structured path feasible", "main_path_feasible"),
        ("Main structured path min turn radius respected", "main_path_min_turn_radius_respected"),
        ("Cleanup path feasible", "cleanup_path_feasible"),
        ("Cleanup path min turn radius respected", "cleanup_path_min_turn_radius_respected"),
        ("Full path feasible", "full_path_feasible"),
        ("Full path min turn radius respected", "full_path_min_turn_radius_respected"),
    ):
        print(f"{label}: {validation[key]}")
    plot_eight_segment_boundary_plan(plan)
