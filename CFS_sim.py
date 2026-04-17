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
DEFAULT_MIN_COLUMN_JUMP = 3
ROUTING_CLEARANCE_PADDING = 0.15
ROUTE_SAMPLE_STEP = 0.3
DEFAULT_COLUMNS = np.array(
    [
        (0.0, 0.0),
        (5.0, 3.0),
        (-4.0, 5.0),
        (6.0, -4.0),
        (-5.0, -4.0),
        (3.0, -6.0),
    ],
    dtype=float,
)
COLUMNS = DEFAULT_COLUMNS.copy()


@dataclass
class PlannerConfig:
    tank_radius: float = TANK_RADIUS
    grid_size: float = GRID_SIZE
    min_turn_radius: float = MIN_TURN_RADIUS
    column_radius: float = COLUMN_RADIUS
    column_clearance: float = COLUMN_CLEARANCE
    robot_width: float = ROBOT_WIDTH
    guide_point_spacing: float = GUIDE_POINT_SPACING
    cleanup_min_column_jump: int = DEFAULT_MIN_COLUMN_JUMP
    cleanup_max_column_jump: int | None = None
    routing_clearance_padding: float = ROUTING_CLEARANCE_PADDING
    route_sample_step: float = ROUTE_SAMPLE_STEP
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

    @property
    def routing_keepout_radius(self) -> float:
        return self.centerline_column_margin + self.routing_clearance_padding


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
    cleanup_path: np.ndarray
    full_path: np.ndarray
    cleanup_region_indices: list[np.ndarray]
    cleanup_region_modes: list[str]
    cleanup_transition_paths: list[np.ndarray]
    cleanup_region_paths: list[np.ndarray]
    cleanup_visit_order: np.ndarray
    cleanup_region_statuses: list[str]
    cleanup_region_failures: list[str | None]
    cleanup_region_active_columns: list[np.ndarray]
    cleanup_region_adjacency: list[dict[int, tuple[int, ...]]]
    cleanup_region_degree_counts: list[dict[int, int]]
    cleanup_region_start_columns: list[int | None]
    cleanup_region_column_orders: list[np.ndarray]
    cleanup_region_jump_sizes: list[np.ndarray]


@dataclass
class CleanupColumnOrderResult:
    active_columns: np.ndarray
    adjacency: dict[int, tuple[int, ...]]
    degree_counts: dict[int, int]
    start_column: int
    order: np.ndarray
    jumps: np.ndarray

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


def _routing_point_is_clear(point: np.ndarray, config: PlannerConfig) -> bool:
    point = np.asarray(point, dtype=float)
    if np.linalg.norm(point) > config.outermost_centerline_radius + 1e-9:
        return False
    return all(np.linalg.norm(point - column) >= config.routing_keepout_radius - 1e-9 for column in config.columns)


def _push_out_of_obstacles(point: np.ndarray, config: PlannerConfig) -> np.ndarray:
    adjusted = np.asarray(point, dtype=float).copy()
    for _ in range(12):
        push = np.zeros(2, dtype=float)
        radius = float(np.linalg.norm(adjusted))
        if radius > config.outermost_centerline_radius:
            push -= adjusted / max(radius, 1e-9) * (radius - config.outermost_centerline_radius + 1e-3)
        for column in config.columns:
            delta = adjusted - column
            distance = float(np.linalg.norm(delta))
            if distance >= config.routing_keepout_radius:
                continue
            if distance <= 1e-9:
                delta = np.array([1.0, 0.0], dtype=float)
                distance = 1.0
            push += delta / distance * (config.routing_keepout_radius - distance + 1e-3)
        if np.linalg.norm(push) <= 1e-6:
            break
        adjusted += push
    return adjusted


def _path_is_clear(points: np.ndarray, config: PlannerConfig) -> bool:
    return bool(np.all([_routing_point_is_clear(point, config) for point in np.asarray(points, dtype=float)]))


def _point_to_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> tuple[float, float]:
    segment = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
    length_sq = float(np.dot(segment, segment))
    if length_sq <= 1e-12:
        return float(np.linalg.norm(np.asarray(point, dtype=float) - np.asarray(start, dtype=float))), 0.0
    t = float(np.clip(np.dot(np.asarray(point, dtype=float) - np.asarray(start, dtype=float), segment) / length_sq, 0.0, 1.0))
    closest = np.asarray(start, dtype=float) + t * segment
    return float(np.linalg.norm(np.asarray(point, dtype=float) - closest)), t


def _segment_hits_circle(
    start: np.ndarray,
    end: np.ndarray,
    center: np.ndarray,
    radius: float,
    tol: float = 1e-6,
) -> tuple[bool, float]:
    distance, t = _point_to_segment_distance(center, start, end)
    return distance < radius - tol, t


def _find_blocking_columns(start: np.ndarray, end: np.ndarray, config: PlannerConfig) -> list[tuple[float, int]]:
    blockers: list[tuple[float, int]] = []
    for idx, column in enumerate(config.columns):
        hits, t = _segment_hits_circle(start, end, column, config.routing_keepout_radius)
        if hits:
            blockers.append((t, idx))
    blockers.sort(key=lambda item: item[0])
    return blockers


def _sample_arc(
    center: np.ndarray,
    radius: float,
    start_angle: float,
    end_angle: float,
    clockwise: bool,
    spacing: float,
) -> np.ndarray:
    if clockwise and end_angle > start_angle:
        end_angle -= 2.0 * np.pi
    if not clockwise and end_angle < start_angle:
        end_angle += 2.0 * np.pi
    sweep = abs(end_angle - start_angle)
    steps = max(3, int(np.ceil(radius * sweep / max(spacing, 1e-6))) + 1)
    angles = np.linspace(start_angle, end_angle, steps)
    return np.column_stack((center[0] + radius * np.cos(angles), center[1] + radius * np.sin(angles)))


def _build_column_detour_path(
    start: np.ndarray,
    end: np.ndarray,
    column_center: np.ndarray,
    side: int,
    config: PlannerConfig,
    depth: int,
) -> np.ndarray | None:
    clearance = config.routing_keepout_radius
    start_delta = np.asarray(start, dtype=float) - column_center
    end_delta = np.asarray(end, dtype=float) - column_center
    start_dist = float(np.linalg.norm(start_delta))
    end_dist = float(np.linalg.norm(end_delta))
    if start_dist <= clearance + 1e-6 or end_dist <= clearance + 1e-6:
        return None

    start_angle = float(np.arctan2(start_delta[1], start_delta[0]))
    end_angle = float(np.arctan2(end_delta[1], end_delta[0]))
    start_alpha = float(np.arccos(np.clip(clearance / start_dist, -1.0, 1.0)))
    end_alpha = float(np.arccos(np.clip(clearance / end_dist, -1.0, 1.0)))
    tangent_start_angle = start_angle - side * start_alpha
    tangent_end_angle = end_angle + side * end_alpha
    tangent_start = column_center + clearance * np.array([np.cos(tangent_start_angle), np.sin(tangent_start_angle)], dtype=float)
    tangent_end = column_center + clearance * np.array([np.cos(tangent_end_angle), np.sin(tangent_end_angle)], dtype=float)
    if np.linalg.norm(tangent_start) > config.outermost_centerline_radius + 1e-6:
        return None
    if np.linalg.norm(tangent_end) > config.outermost_centerline_radius + 1e-6:
        return None

    approach = _route_between_points(start, tangent_start, config, depth + 1)
    arc = _sample_arc(
        column_center,
        clearance,
        tangent_start_angle,
        tangent_end_angle,
        clockwise=side > 0,
        spacing=config.route_sample_step,
    )
    departure = _route_between_points(tangent_end, end, config, depth + 1)
    candidate = concatenate_paths([approach, arc, departure])
    return candidate if _path_is_clear(candidate, config) else None


def _route_between_points(
    start: np.ndarray,
    end: np.ndarray,
    config: PlannerConfig,
    depth: int = 0,
) -> np.ndarray:
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    if not _routing_point_is_clear(start, config):
        start = _push_out_of_obstacles(start, config)
    if not _routing_point_is_clear(end, config):
        end = _push_out_of_obstacles(end, config)
    if np.linalg.norm(end - start) <= 1e-9:
        return start[None, :]

    direct = build_line_segment(start, end, config.route_sample_step)
    blockers = _find_blocking_columns(start, end, config)
    if (not blockers and _path_is_clear(direct, config)) or depth >= 5:
        return direct

    _, blocking_idx = blockers[0]
    column_center = config.columns[blocking_idx]
    candidates = [
        candidate
        for side in (-1, 1)
        for candidate in [_build_column_detour_path(start, end, column_center, side, config, depth)]
        if candidate is not None and len(candidate)
    ]
    return min(candidates, key=_path_length) if candidates else direct


def _reroute_polyline(points: np.ndarray, config: PlannerConfig) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        return np.empty((0, 2), dtype=float)
    if len(config.columns) == 0:
        return points.copy()
    if len(points) == 1:
        point = points[0] if _routing_point_is_clear(points[0], config) else _push_out_of_obstacles(points[0], config)
        return point[None, :]
    chunks = [_route_between_points(start, end, config) for start, end in zip(points[:-1], points[1:])]
    return concatenate_paths(chunks)


def _config_without_columns(config: PlannerConfig) -> PlannerConfig:
    return PlannerConfig(
        tank_radius=config.tank_radius,
        grid_size=config.grid_size,
        min_turn_radius=config.min_turn_radius,
        column_radius=config.column_radius,
        column_clearance=config.column_clearance,
        robot_width=config.robot_width,
        guide_point_spacing=config.guide_point_spacing,
        cleanup_min_column_jump=config.cleanup_min_column_jump,
        cleanup_max_column_jump=config.cleanup_max_column_jump,
        routing_clearance_padding=config.routing_clearance_padding,
        route_sample_step=config.route_sample_step,
        columns=np.empty((0, 2), dtype=float),
    )

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
    accepted_points = np.asarray(accepted, dtype=float) if accepted else np.empty((0, 2), dtype=float)
    return ScanCoverageResult(
        retained_square_centers=np.asarray(retained, dtype=float) if retained else np.empty((0, 2), dtype=float),
        blocked_square_centers=np.asarray(blocked, dtype=float) if blocked else np.empty((0, 2), dtype=float),
        nominal_stop_centers=accepted_points.copy(),
        stop_points=accepted_points,
        adjusted_mask=np.zeros(len(accepted_points), dtype=bool),
        discarded_square_centers=np.asarray(discarded, dtype=float) if discarded else np.empty((0, 2), dtype=float),
    )

def _perimeter_points(points: np.ndarray, grid_size: float) -> np.ndarray:
    if len(points) == 0:
        return np.empty((0, 2), dtype=float)
    keys = np.rint(points / grid_size).astype(int)
    key_set = {tuple(key) for key in keys}
    keep: list[np.ndarray] = []
    for point, key in zip(points, keys):
        if any((key[0] + dx, key[1] + dy) not in key_set for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))):
            keep.append(point)
    return np.asarray(keep, dtype=float) if keep else np.empty((0, 2), dtype=float)

def _sort_segment(points: np.ndarray, primary: np.ndarray, secondary: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.empty((0, 2), dtype=float)
    order = np.lexsort((secondary, primary))
    return points[order]


def _ordered_boundary_segments(points: np.ndarray) -> list[np.ndarray]:
    if len(points) == 0:
        return [np.empty((0, 2), dtype=float) for _ in range(8)]

    x = points[:, 0]
    y = points[:, 1]
    left, right = float(np.min(x)), float(np.max(x))
    bottom, top = float(np.min(y)), float(np.max(y))
    top_mask = np.isclose(y, top)
    right_mask = np.isclose(x, right)
    bottom_mask = np.isclose(y, bottom)
    left_mask = np.isclose(x, left)

    top_edge = _sort_segment(points[top_mask], points[top_mask][:, 0], np.zeros(np.count_nonzero(top_mask)))
    right_edge = _sort_segment(points[right_mask], -points[right_mask][:, 1], np.zeros(np.count_nonzero(right_mask)))
    bottom_edge = _sort_segment(points[bottom_mask], -points[bottom_mask][:, 0], np.zeros(np.count_nonzero(bottom_mask)))
    left_edge = _sort_segment(points[left_mask], points[left_mask][:, 1], np.zeros(np.count_nonzero(left_mask)))

    edge_mask = top_mask | right_mask | bottom_mask | left_mask
    interior = points[~edge_mask]
    if len(interior) == 0:
        return [top_edge, np.empty((0, 2), dtype=float), right_edge, np.empty((0, 2), dtype=float), bottom_edge, np.empty((0, 2), dtype=float), left_edge, np.empty((0, 2), dtype=float)]

    ix = interior[:, 0]
    iy = interior[:, 1]
    ur_mask = (ix > 0) & (iy > 0)
    lr_mask = (ix > 0) & (iy < 0)
    ll_mask = (ix < 0) & (iy < 0)
    ul_mask = (ix < 0) & (iy > 0)

    upper_right = interior[ur_mask]
    lower_right = interior[lr_mask]
    lower_left = interior[ll_mask]
    upper_left = interior[ul_mask]

    if len(upper_right):
        target = float(np.max(np.sum(upper_right, axis=1)))
        upper_right = upper_right[np.isclose(np.sum(upper_right, axis=1), target)]
    if len(lower_right):
        target = float(np.max(lower_right[:, 0] - lower_right[:, 1]))
        lower_right = lower_right[np.isclose(lower_right[:, 0] - lower_right[:, 1], target)]
    if len(lower_left):
        target = float(np.min(np.sum(lower_left, axis=1)))
        lower_left = lower_left[np.isclose(np.sum(lower_left, axis=1), target)]
    if len(upper_left):
        target = float(np.max(upper_left[:, 1] - upper_left[:, 0]))
        upper_left = upper_left[np.isclose(upper_left[:, 1] - upper_left[:, 0], target)]

    upper_right = _sort_segment(upper_right, -upper_right[:, 1], upper_right[:, 0]) if len(upper_right) else np.empty((0, 2), dtype=float)
    lower_right = _sort_segment(lower_right, -lower_right[:, 1], -lower_right[:, 0]) if len(lower_right) else np.empty((0, 2), dtype=float)
    lower_left = _sort_segment(lower_left, lower_left[:, 1], -lower_left[:, 0]) if len(lower_left) else np.empty((0, 2), dtype=float)
    upper_left = _sort_segment(upper_left, upper_left[:, 1], upper_left[:, 0]) if len(upper_left) else np.empty((0, 2), dtype=float)

    return [top_edge, upper_right, right_edge, lower_right, bottom_edge, lower_left, left_edge, upper_left]


def _flatten_segments(segments: list[np.ndarray]) -> np.ndarray:
    return np.vstack([segment for segment in segments if len(segment)]) if any(len(segment) for segment in segments) else np.empty((0, 2), dtype=float)

def _remove_points(source: np.ndarray, to_remove: np.ndarray) -> np.ndarray:
    if len(source) == 0 or len(to_remove) == 0:
        return source.copy()
    keep = np.ones(len(source), dtype=bool)
    for point in to_remove:
        keep &= ~np.all(np.isclose(source, point, atol=1e-9), axis=1)
    return source[keep]

def _boundary_layers(stop_points: np.ndarray, config: PlannerConfig) -> tuple[list[list[np.ndarray]], np.ndarray]:
    layers: list[list[np.ndarray]] = []
    remaining = stop_points.copy()
    while len(remaining):
        unique_x = np.unique(np.round(remaining[:, 0] / config.grid_size).astype(int))
        unique_y = np.unique(np.round(remaining[:, 1] / config.grid_size).astype(int))
        if len(unique_x) <= FINAL_STAGE_GRID_SIDE and len(unique_y) <= FINAL_STAGE_GRID_SIDE:
            break
        segments = _ordered_boundary_segments(_perimeter_points(remaining, config.grid_size))
        layer_points = _flatten_segments(segments)
        if len(layer_points) < 8:
            break
        layers.append(segments)
        remaining = _remove_points(remaining, layer_points)
    return layers, remaining


def _point_grid_key(point: np.ndarray, grid_size: float) -> tuple[int, int]:
    scaled = np.rint(np.asarray(point, dtype=float) / grid_size).astype(int)
    return int(scaled[0]), int(scaled[1])


def _extract_cleanup_regions(stop_points: np.ndarray, remaining_indices: np.ndarray, grid_size: float) -> list[np.ndarray]:
    if len(remaining_indices) == 0:
        return []

    key_to_index = {
        _point_grid_key(stop_points[idx], grid_size): int(idx)
        for idx in np.asarray(remaining_indices, dtype=int)
    }
    visited: set[int] = set()
    regions: list[np.ndarray] = []

    for start_idx in np.asarray(remaining_indices, dtype=int):
        if int(start_idx) in visited:
            continue

        stack = [int(start_idx)]
        component: list[int] = []
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component.append(current)
            cx, cy = _point_grid_key(stop_points[current], grid_size)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                neighbor = key_to_index.get((cx + dx, cy + dy))
                if neighbor is not None and neighbor not in visited:
                    stack.append(neighbor)

        regions.append(np.asarray(sorted(component), dtype=int))

    regions.sort(key=lambda region: (len(region), float(np.mean(stop_points[region, 0]))), reverse=True)
    return regions


def extract_active_cleanup_columns(region_points: np.ndarray, grid_size: float) -> np.ndarray:
    if len(region_points) == 0:
        return np.empty(0, dtype=int)
    return np.unique(np.rint(region_points[:, 0] / grid_size).astype(int))


def build_feasible_column_graph(
    active_columns: np.ndarray,
    min_column_jump: int,
    max_column_jump: int | None = None,
) -> dict[int, tuple[int, ...]]:
    columns = np.asarray(active_columns, dtype=int)
    adjacency: dict[int, tuple[int, ...]] = {}
    for column in columns:
        neighbors = []
        for other in columns:
            jump = abs(int(other) - int(column))
            if jump == 0 or jump < min_column_jump:
                continue
            if max_column_jump is not None and jump > max_column_jump:
                continue
            neighbors.append(int(other))
        adjacency[int(column)] = tuple(sorted(neighbors))
    return adjacency


def remaining_columns_still_reachable(
    current_column: int,
    remaining_columns: set[int],
    adjacency: dict[int, tuple[int, ...]],
) -> bool:
    if not remaining_columns:
        return True

    frontier = {current_column, *remaining_columns}
    if not any(neighbor in remaining_columns for neighbor in adjacency.get(current_column, ())):
        return False

    for column in remaining_columns:
        degree = sum(neighbor in frontier for neighbor in adjacency.get(column, ()))
        if degree == 0:
            return False

    stack = [current_column]
    seen: set[int] = set()
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        stack.extend(neighbor for neighbor in adjacency.get(node, ()) if neighbor in frontier and neighbor not in seen)
    return frontier <= seen


def search_column_order(
    current_column: int,
    remaining_columns: set[int],
    adjacency: dict[int, tuple[int, ...]],
    order: list[int],
) -> list[int] | None:
    if not remaining_columns:
        return order.copy()

    candidates = [neighbor for neighbor in adjacency.get(current_column, ()) if neighbor in remaining_columns]
    candidates.sort(
        key=lambda neighbor: (
            abs(neighbor - current_column),
            sum(next_neighbor in remaining_columns for next_neighbor in adjacency.get(neighbor, ())),
            abs(neighbor),
            neighbor,
        )
    )

    for next_column in candidates:
        next_remaining = set(remaining_columns)
        next_remaining.remove(next_column)
        if not remaining_columns_still_reachable(next_column, next_remaining, adjacency):
            continue
        order.append(next_column)
        found = search_column_order(next_column, next_remaining, adjacency, order)
        if found is not None:
            return found
        order.pop()
    return None


def find_column_order(
    active_columns: np.ndarray,
    min_column_jump: int = DEFAULT_MIN_COLUMN_JUMP,
    max_column_jump: int | None = None,
) -> CleanupColumnOrderResult | None:
    columns = np.asarray(active_columns, dtype=int)
    if len(columns) == 0:
        return None

    adjacency = build_feasible_column_graph(columns, min_column_jump, max_column_jump)
    degree_counts = {column: len(adjacency[column]) for column in adjacency}
    if len(columns) == 1:
        return CleanupColumnOrderResult(
            active_columns=columns.copy(),
            adjacency=adjacency,
            degree_counts=degree_counts,
            start_column=int(columns[0]),
            order=columns.copy(),
            jumps=np.empty(0, dtype=int),
        )

    start_columns = sorted(columns.tolist(), key=lambda column: (degree_counts[column], abs(column), column))
    for start_column in start_columns:
        remaining = set(int(column) for column in columns if int(column) != int(start_column))
        if not remaining_columns_still_reachable(int(start_column), remaining, adjacency):
            continue
        order = search_column_order(int(start_column), remaining, adjacency, [int(start_column)])
        if order is not None:
            order_array = np.asarray(order, dtype=int)
            return CleanupColumnOrderResult(
                active_columns=columns.copy(),
                adjacency=adjacency,
                degree_counts=degree_counts,
                start_column=int(start_column),
                order=order_array,
                jumps=np.abs(np.diff(order_array)).astype(int),
            )
    return None


def _path_length(points: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))) if len(points) > 1 else 0.0


def _estimate_local_turn_radii(points: np.ndarray) -> np.ndarray:
    if len(points) < 3:
        return np.empty(0, dtype=float)
    radii: list[float] = []
    for a, b, c in zip(points[:-2], points[1:-1], points[2:]):
        ab = float(np.linalg.norm(b - a))
        bc = float(np.linalg.norm(c - b))
        ca = float(np.linalg.norm(a - c))
        area2 = abs(float(np.cross(b - a, c - a)))
        radii.append(np.inf if area2 <= 1e-9 else ab * bc * ca / (2.0 * area2))
    return np.asarray(radii, dtype=float)


def _path_respects_turn_radius(points: np.ndarray, min_turn_radius: float) -> bool:
    finite = _estimate_local_turn_radii(np.asarray(points, dtype=float))
    finite = finite[np.isfinite(finite)]
    return bool(len(finite) == 0 or np.min(finite) >= min_turn_radius - 1e-6)


def _path_end_heading(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    delta = points[-1] - points[-2]
    return float(np.arctan2(delta[1], delta[0]))


def _build_cleanup_connector(
    start_point: np.ndarray,
    start_heading: float,
    end_point: np.ndarray,
    end_heading: float,
    config: PlannerConfig,
) -> np.ndarray:
    planned = plan_dubins_segment(start_point, start_heading, end_point, end_heading, config.min_turn_radius)
    if planned is None:
        return build_line_segment(start_point, end_point, config.guide_point_spacing)

    mode, lengths = planned
    connector = sample_dubins_path(
        start_point,
        start_heading,
        mode,
        lengths,
        config.min_turn_radius,
        config.guide_point_spacing,
    )
    connector[0] = np.asarray(start_point, dtype=float)
    connector[-1] = np.asarray(end_point, dtype=float)
    return connector


def _build_soft_turn_connector(
    start_point: np.ndarray,
    start_heading: float,
    end_point: np.ndarray,
    end_heading: float,
    config: PlannerConfig,
) -> np.ndarray:
    start_point = np.asarray(start_point, dtype=float)
    end_point = np.asarray(end_point, dtype=float)
    distance = float(np.linalg.norm(end_point - start_point))
    if distance <= 1e-9:
        return start_point[None, :]

    handle = min(0.35 * distance, 0.45 * config.grid_size, 0.35 * config.min_turn_radius)
    start_tangent = handle * np.array([np.cos(start_heading), np.sin(start_heading)], dtype=float)
    end_tangent = handle * np.array([np.cos(end_heading), np.sin(end_heading)], dtype=float)
    steps = max(3, int(np.ceil(distance / max(config.guide_point_spacing, 1e-6))) + 1)
    t = np.linspace(0.0, 1.0, steps)
    h00 = 2.0 * t ** 3 - 3.0 * t ** 2 + 1.0
    h10 = t ** 3 - 2.0 * t ** 2 + t
    h01 = -2.0 * t ** 3 + 3.0 * t ** 2
    h11 = t ** 3 - t ** 2
    curve = (
        h00[:, None] * start_point
        + h10[:, None] * start_tangent
        + h01[:, None] * end_point
        + h11[:, None] * end_tangent
    )
    curve[0] = start_point
    curve[-1] = end_point
    return curve


def _group_region_points_by_column(
    region_indices: np.ndarray,
    stop_points: np.ndarray,
    grid_size: float,
) -> dict[int, np.ndarray]:
    groups: dict[int, list[int]] = {}
    for idx in np.asarray(region_indices, dtype=int):
        column_id = int(np.rint(stop_points[idx, 0] / grid_size))
        groups.setdefault(column_id, []).append(int(idx))
    return {
        column_id: np.asarray(sorted(indices, key=lambda idx: stop_points[idx, 1]), dtype=int)
        for column_id, indices in groups.items()
    }


def _column_heading(is_upward: bool) -> float:
    return float(np.pi / 2.0 if is_upward else -np.pi / 2.0)


def _build_column_segment(points: np.ndarray, spacing: float) -> np.ndarray:
    if len(points) == 0:
        return np.empty((0, 2), dtype=float)
    if len(points) == 1:
        return points.copy()
    return concatenate_paths([build_line_segment(start, end, spacing) for start, end in zip(points[:-1], points[1:])])


def _build_cleanup_region_variant(
    column_order: np.ndarray,
    grouped_points: dict[int, np.ndarray],
    stop_points: np.ndarray,
    entry_point: np.ndarray,
    entry_heading: float,
    start_upward: bool,
    config: PlannerConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    visit_order_chunks: list[np.ndarray] = []
    column_directions: list[bool] = []

    for idx, column_id in enumerate(np.asarray(column_order, dtype=int)):
        upward = start_upward if idx % 2 == 0 else not start_upward
        local_indices = grouped_points[int(column_id)]
        visit_order_chunks.append(local_indices if upward else local_indices[::-1])
        column_directions.append(upward)

    visit_order = np.concatenate(visit_order_chunks) if visit_order_chunks else np.empty(0, dtype=int)
    if len(visit_order) == 0:
        return visit_order, np.empty((0, 2), dtype=float), np.empty((0, 2), dtype=float), 0.0

    ordered_points = stop_points[visit_order]
    transition_path = _build_cleanup_connector(
        np.asarray(entry_point, dtype=float),
        entry_heading,
        ordered_points[0],
        _column_heading(column_directions[0]),
        config,
    )

    region_chunks: list[np.ndarray] = []
    cursor = 0
    for idx, column_indices in enumerate(visit_order_chunks):
        column_points = stop_points[column_indices]
        region_chunks.append(_build_column_segment(column_points, config.guide_point_spacing))
        cursor += len(column_indices)
        if idx == len(visit_order_chunks) - 1:
            continue
        next_points = stop_points[visit_order_chunks[idx + 1]]
        connector = _build_cleanup_connector(
            column_points[-1],
            _column_heading(column_directions[idx]),
            next_points[0],
            _column_heading(column_directions[idx + 1]),
            config,
        )
        region_chunks.append(connector)

    region_path = concatenate_paths(region_chunks)
    total_length = _path_length(transition_path) + _path_length(region_path)
    return visit_order, transition_path, region_path, total_length


def _segment_heading(segment: np.ndarray) -> float:
    if len(segment) < 2:
        return 0.0
    delta = segment[-1] - segment[0]
    return float(np.arctan2(delta[1], delta[0]))


def _flatten_segments_with_headings(layer_segments: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    ordered_points: list[np.ndarray] = []
    headings: list[float] = []
    for segment in layer_segments:
        if len(segment) == 0:
            continue
        heading = _segment_heading(segment)
        for point in np.asarray(segment, dtype=float):
            if ordered_points and np.linalg.norm(point - ordered_points[-1]) <= 1e-9:
                headings[-1] = heading
                continue
            ordered_points.append(point)
            headings.append(heading)
    if not ordered_points:
        return np.empty((0, 2), dtype=float), np.empty(0, dtype=float)
    return np.asarray(ordered_points, dtype=float), np.asarray(headings, dtype=float)


def _layer_path(layer_segments: list[np.ndarray], config: PlannerConfig) -> np.ndarray:
    points, headings = _flatten_segments_with_headings(layer_segments)
    if len(points) <= 1:
        return points.copy()

    chunks: list[np.ndarray] = [points[:1].copy()]
    for idx in range(len(points) - 1):
        start_point = points[idx]
        end_point = points[idx + 1]
        start_heading = float(headings[idx])
        end_heading = float(headings[idx + 1])
        if abs(np.sin(start_heading - end_heading)) <= 1e-9 and abs(np.cos(start_heading - end_heading) - 1.0) <= 1e-9:
            chunks.append(build_line_segment(start_point, end_point, config.guide_point_spacing))
        else:
            chunks.append(_build_soft_turn_connector(start_point, start_heading, end_point, end_heading, config))
    return concatenate_paths(chunks)

def build_eight_segment_boundary_plan(config: PlannerConfig | None = None) -> BoundaryEightSegmentPlan:
    config = PlannerConfig() if config is None else config
    planning_config = _config_without_columns(config)
    scan_result = build_scan_coverage_result(planning_config)
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
            cleanup_path=empty,
            full_path=empty,
            cleanup_region_indices=[],
            cleanup_region_modes=[],
            cleanup_transition_paths=[],
            cleanup_region_paths=[],
            cleanup_visit_order=empty_idx,
            cleanup_region_statuses=[],
            cleanup_region_failures=[],
            cleanup_region_active_columns=[],
            cleanup_region_adjacency=[],
            cleanup_region_degree_counts=[],
            cleanup_region_start_columns=[],
            cleanup_region_column_orders=[],
            cleanup_region_jump_sizes=[],
        )

    layers, remaining = _boundary_layers(stop_points, planning_config)
    nominal_layer_paths = [_layer_path(layer, planning_config) for layer in layers]
    lookup = {(float(point[0]), float(point[1])): idx for idx, point in enumerate(stop_points)}
    layer_indices = [
        np.asarray([lookup[(float(point[0]), float(point[1]))] for point in _flatten_segments(layer)], dtype=int)
        for layer in layers
    ]
    main_visited_indices = (
        np.asarray(sorted({idx for group in layer_indices for idx in group.tolist()}), dtype=int)
        if layer_indices
        else np.empty(0, dtype=int)
    )
    all_indices = np.arange(len(stop_points), dtype=int)
    remaining_indices = np.setdiff1d(all_indices, main_visited_indices, assume_unique=False)
    first_point = _flatten_segments(layers[0])[0] if layers else np.array([0.0, planning_config.outermost_centerline_radius], dtype=float)
    nominal_entry_path = build_line_segment(np.zeros(2, dtype=float), first_point, planning_config.guide_point_spacing)
    nominal_inter_layer_paths = [
        build_line_segment(current[-1], nxt[0], planning_config.guide_point_spacing)
        for current, nxt in zip(nominal_layer_paths[:-1], nominal_layer_paths[1:])
    ]
    entry_path = _reroute_polyline(nominal_entry_path, config)
    layer_paths = [_reroute_polyline(path, config) for path in nominal_layer_paths]
    inter_layer_paths = [_reroute_polyline(path, config) for path in nominal_inter_layer_paths]

    parts: list[np.ndarray] = [entry_path]
    for idx, layer_path in enumerate(layer_paths):
        parts.append(layer_path)
        if idx < len(inter_layer_paths):
            parts.append(inter_layer_paths[idx])
    main_path = concatenate_paths(parts)
    cleanup_region_indices = _extract_cleanup_regions(stop_points, remaining_indices, config.grid_size)
    cleanup_region_modes: list[str] = []
    cleanup_transition_paths: list[np.ndarray] = []
    cleanup_region_paths: list[np.ndarray] = []
    cleanup_visit_order_chunks: list[np.ndarray] = []
    cleanup_region_statuses: list[str] = []
    cleanup_region_failures: list[str | None] = []
    cleanup_region_active_columns: list[np.ndarray] = []
    cleanup_region_adjacency: list[dict[int, tuple[int, ...]]] = []
    cleanup_region_degree_counts: list[dict[int, int]] = []
    cleanup_region_start_columns: list[int | None] = []
    cleanup_region_column_orders: list[np.ndarray] = []
    cleanup_region_jump_sizes: list[np.ndarray] = []
    cleanup_path_chunks: list[np.ndarray] = []
    current_reference_path = main_path if len(main_path) else entry_path

    for region_indices in cleanup_region_indices:
        region_points = stop_points[region_indices]
        active_columns = extract_active_cleanup_columns(region_points, config.grid_size)
        search_result = find_column_order(
            active_columns,
            config.cleanup_min_column_jump,
            config.cleanup_max_column_jump,
        )
        adjacency = (
            search_result.adjacency
            if search_result is not None
            else build_feasible_column_graph(active_columns, config.cleanup_min_column_jump, config.cleanup_max_column_jump)
        )
        degree_counts = (
            search_result.degree_counts
            if search_result is not None
            else {column: len(adjacency[column]) for column in adjacency}
        )

        cleanup_region_active_columns.append(active_columns.copy())
        cleanup_region_adjacency.append(adjacency)
        cleanup_region_degree_counts.append(degree_counts)
        cleanup_region_start_columns.append(None if search_result is None else int(search_result.start_column))
        cleanup_region_column_orders.append(
            np.empty(0, dtype=int) if search_result is None else search_result.order.copy()
        )
        cleanup_region_jump_sizes.append(
            np.empty(0, dtype=int) if search_result is None else search_result.jumps.copy()
        )

        if search_result is None:
            cleanup_region_modes.append("unresolved")
            cleanup_region_statuses.append("unresolved")
            cleanup_region_failures.append("no_valid_column_order")
            cleanup_transition_paths.append(np.empty((0, 2), dtype=float))
            cleanup_region_paths.append(np.empty((0, 2), dtype=float))
            continue

        grouped_points = _group_region_points_by_column(region_indices, stop_points, config.grid_size)
        entry_point = current_reference_path[-1] if len(current_reference_path) else np.zeros(2, dtype=float)
        entry_heading = _path_end_heading(current_reference_path)
        variants: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []
        for start_upward in (True, False):
            visit_order, transition_nominal, region_nominal, _ = _build_cleanup_region_variant(
                search_result.order,
                grouped_points,
                stop_points,
                entry_point,
                entry_heading,
                start_upward,
                planning_config,
            )
            transition_path = _reroute_polyline(transition_nominal, config)
            region_path = _reroute_polyline(region_nominal, config)
            variants.append((visit_order, transition_path, region_path, _path_length(transition_path) + _path_length(region_path)))
        visit_order, transition_path, region_path, _ = min(variants, key=lambda variant: variant[3])

        cleanup_region_modes.append("column_ordered_sweep")
        cleanup_region_statuses.append("resolved")
        cleanup_region_failures.append(None)
        cleanup_transition_paths.append(transition_path)
        cleanup_region_paths.append(region_path)
        cleanup_visit_order_chunks.append(visit_order)
        cleanup_path_chunks.extend([transition_path, region_path])
        current_reference_path = region_path if len(region_path) else transition_path

    cleanup_visit_order = (
        np.concatenate(cleanup_visit_order_chunks) if cleanup_visit_order_chunks else np.empty(0, dtype=int)
    )
    cleanup_path = concatenate_paths(cleanup_path_chunks)
    full_path = concatenate_paths([main_path, cleanup_path])

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
        cleanup_path=cleanup_path,
        full_path=full_path,
        cleanup_region_indices=cleanup_region_indices,
        cleanup_region_modes=cleanup_region_modes,
        cleanup_transition_paths=cleanup_transition_paths,
        cleanup_region_paths=cleanup_region_paths,
        cleanup_visit_order=cleanup_visit_order,
        cleanup_region_statuses=cleanup_region_statuses,
        cleanup_region_failures=cleanup_region_failures,
        cleanup_region_active_columns=cleanup_region_active_columns,
        cleanup_region_adjacency=cleanup_region_adjacency,
        cleanup_region_degree_counts=cleanup_region_degree_counts,
        cleanup_region_start_columns=cleanup_region_start_columns,
        cleanup_region_column_orders=cleanup_region_column_orders,
        cleanup_region_jump_sizes=cleanup_region_jump_sizes,
    )

def validate_eight_segment_boundary_plan(plan: BoundaryEightSegmentPlan) -> dict[str, object]:
    stop_points = plan.scan_result.stop_points
    hits = (
        np.asarray([np.any(np.all(np.isclose(plan.full_path, point, atol=1e-9), axis=1)) for point in stop_points], dtype=bool)
        if len(stop_points)
        else np.empty(0, dtype=bool)
    )
    cleanup_hits = (
        np.asarray([np.any(np.all(np.isclose(plan.cleanup_path, point, atol=1e-9), axis=1)) for point in stop_points], dtype=bool)
        if len(stop_points)
        else np.empty(0, dtype=bool)
    )
    cleanup_region_summaries = []
    for idx, region_indices in enumerate(plan.cleanup_region_indices):
        active_columns = plan.cleanup_region_active_columns[idx] if idx < len(plan.cleanup_region_active_columns) else np.empty(0, dtype=int)
        order = plan.cleanup_region_column_orders[idx] if idx < len(plan.cleanup_region_column_orders) else np.empty(0, dtype=int)
        jumps = plan.cleanup_region_jump_sizes[idx] if idx < len(plan.cleanup_region_jump_sizes) else np.empty(0, dtype=int)
        status = plan.cleanup_region_statuses[idx] if idx < len(plan.cleanup_region_statuses) else "unknown"
        min_jump = plan.config.cleanup_min_column_jump
        cleanup_region_summaries.append(
            {
                "region_index": idx,
                "status": status,
                "failure": plan.cleanup_region_failures[idx] if idx < len(plan.cleanup_region_failures) else None,
                "active_columns": active_columns.tolist(),
                "degree_counts": plan.cleanup_region_degree_counts[idx] if idx < len(plan.cleanup_region_degree_counts) else {},
                "start_column": plan.cleanup_region_start_columns[idx] if idx < len(plan.cleanup_region_start_columns) else None,
                "column_order": order.tolist(),
                "jump_sizes": jumps.tolist(),
                "visits_every_active_column_once": bool(
                    len(order) == len(active_columns) and np.array_equal(np.sort(order), np.sort(active_columns))
                ),
                "all_jumps_respect_minimum": bool(len(jumps) == 0 or np.all(jumps >= min_jump)),
                "region_point_count": int(len(region_indices)),
            }
        )

    main_path_feasible = _path_is_clear(plan.main_path, plan.config) if len(plan.main_path) else True
    cleanup_path_feasible = _path_is_clear(plan.cleanup_path, plan.config) if len(plan.cleanup_path) else True
    full_path_feasible = _path_is_clear(plan.full_path, plan.config) if len(plan.full_path) else True

    return {
        "total_required_stop_points": int(len(plan.scan_result.stop_points)),
        "points_visited_by_main_structured_coverage": int(len(plan.main_visited_indices)),
        "remaining_unvisited_points_before_cleanup": int(len(plan.remaining_indices_before_cleanup)),
        "cleanup_region_count": int(len(plan.cleanup_region_indices)),
        "resolved_cleanup_region_count": int(sum(status == "resolved" for status in plan.cleanup_region_statuses)),
        "unresolved_cleanup_region_count": int(sum(status != "resolved" for status in plan.cleanup_region_statuses)),
        "sweep_region_count": int(sum(mode == "column_ordered_sweep" for mode in plan.cleanup_region_modes)),
        "one_pass_transition_region_count": 0,
        "cleanup_serviced_point_count": int(len(np.unique(plan.cleanup_visit_order))) if len(plan.cleanup_visit_order) else 0,
        "all_required_stop_points_explicitly_visited": bool(np.all(hits)) if len(hits) else True,
        "cleanup_points_explicitly_visited": bool(np.all(cleanup_hits[plan.cleanup_visit_order])) if len(plan.cleanup_visit_order) else True,
        "main_path_feasible": main_path_feasible,
        "main_path_min_turn_radius_respected": _path_respects_turn_radius(plan.main_path, plan.config.min_turn_radius),
        "cleanup_path_feasible": cleanup_path_feasible,
        "cleanup_path_min_turn_radius_respected": _path_respects_turn_radius(plan.cleanup_path, plan.config.min_turn_radius),
        "full_path_feasible": full_path_feasible,
        "full_path_min_turn_radius_respected": _path_respects_turn_radius(plan.full_path, plan.config.min_turn_radius),
        "cleanup_region_summaries": cleanup_region_summaries,
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
        ax.add_patch(
            Circle(column, config.routing_keepout_radius, fill=False, color="#dc2626", linestyle=":", linewidth=1.0)
        )
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
    unresolved_label_used = False
    for idx, region_indices in enumerate(plan.cleanup_region_indices):
        if idx >= len(plan.cleanup_region_statuses) or plan.cleanup_region_statuses[idx] == "resolved" or len(region_indices) == 0:
            continue
        unresolved = stop_points[region_indices]
        ax.scatter(
            unresolved[:, 0],
            unresolved[:, 1],
            s=44,
            marker="x",
            color="#dc2626",
            alpha=0.75,
            label=None if unresolved_label_used else "Unresolved cleanup region",
            zorder=7,
        )
        unresolved_label_used = True

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
    cleanup_palette = ["#059669", "#7c3aed", "#dc2626", "#d97706", "#0f766e", "#be185d"]
    for idx, transition in enumerate(plan.cleanup_transition_paths):
        if len(transition) == 0:
            continue
        ax.plot(
            transition[:, 0],
            transition[:, 1],
            color="#fb923c",
            linewidth=1.8,
            linestyle="--",
            label="Cleanup transition" if idx == 0 else None,
            zorder=5,
        )
    for idx, region_path in enumerate(plan.cleanup_region_paths):
        if len(region_path) == 0:
            continue
        color = cleanup_palette[idx % len(cleanup_palette)]
        label = "Cleanup path" if idx == 0 else None
        ax.plot(region_path[:, 0], region_path[:, 1], color=color, linewidth=2.4, label=label, zorder=6)
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
    for summary in validation["cleanup_region_summaries"]:
        print(
            f"Cleanup region {summary['region_index'] + 1}: status={summary['status']}, "
            f"active_columns={summary['active_columns']}, start={summary['start_column']}, "
            f"order={summary['column_order']}, jumps={summary['jump_sizes']}, "
            f"degree_counts={summary['degree_counts']}, failure={summary['failure']}"
        )
    plot_eight_segment_boundary_plan(plan)
