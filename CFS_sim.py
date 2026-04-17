from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# Tank / planner parameters
TANK_RADIUS = 15.0
GRID_SIZE = 5.0 * 0.3048
HALF_GRID = GRID_SIZE / 2.0
PASS_SPACING = 0.31
SCAN_WIDTH = 0.364
COLUMN_RADIUS = 0.3
COLUMN_CLEARANCE = 0.3
ROBOT_WIDTH = 0.25
RESOLUTION = 0.05
MAX_TURN_RATE = np.deg2rad(5.0)
SAMPLE_ARC_LENGTH = 0.06
SMOOTHING_WINDOW = 9
WELD_SPACING = 3.0
WELD_BAND_HALF_WIDTH = 0.06
MIN_TURN_RADIUS = 6.0 * 0.3048
TRIGGER_RADIUS = 5.0
PETAL_COUNT = 3
DEFAULT_START_ANGLE = np.pi
PETAL_REFERENCE_SAMPLES = 80
ROUTING_CLEARANCE_PADDING = 0.15
ROUTE_SAMPLE_STEP = 0.3
PATH_SMOOTHING_PASSES = 12
MAIN_GUIDE_RADIAL_STEP = 0.3
GUIDE_POINT_SPACING = 0.25
FINAL_STAGE_GRID_SIDE = 9
MIN_CLEANUP_COLUMN_JUMP = 3

# Parameterizable column layouts.
# `DEFAULT_COLUMNS` preserves the original drawing-based layout for reuse.
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

# Active default layout for the planner. Leave empty for a no-column tank.
COLUMNS = np.empty((0, 2), dtype=float)

MANWAY = np.array([-14.0, 0.0], dtype=float)


CENTERLINE_WALL_MARGIN = COLUMN_CLEARANCE + ROBOT_WIDTH / 2.0
CENTERLINE_COLUMN_MARGIN = COLUMN_RADIUS + COLUMN_CLEARANCE + ROBOT_WIDTH / 2.0
OUTERMOST_CENTERLINE_RADIUS = TANK_RADIUS - CENTERLINE_WALL_MARGIN
MAX_CENTER_SHIFT = CENTERLINE_COLUMN_MARGIN


@dataclass
class PlannerConfig:
    tank_radius: float = TANK_RADIUS
    grid_size: float = GRID_SIZE
    min_turn_radius: float = MIN_TURN_RADIUS
    trigger_radius: float = TRIGGER_RADIUS
    petal_count: int = PETAL_COUNT
    petal_enabled: bool = True
    start_angle: float = DEFAULT_START_ANGLE
    pass_spacing: float = PASS_SPACING
    sample_arc_length: float = SAMPLE_ARC_LENGTH
    smoothing_window: int = SMOOTHING_WINDOW
    column_radius: float = COLUMN_RADIUS
    column_clearance: float = COLUMN_CLEARANCE
    robot_width: float = ROBOT_WIDTH
    max_center_shift: float = MAX_CENTER_SHIFT
    routing_clearance_padding: float = ROUTING_CLEARANCE_PADDING
    route_sample_step: float = ROUTE_SAMPLE_STEP
    path_smoothing_passes: int = PATH_SMOOTHING_PASSES
    main_guide_radial_step: float = MAIN_GUIDE_RADIAL_STEP
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

    @property
    def spiral_pitch(self) -> float:
        return self.grid_size / (2.0 * np.pi)

    @property
    def radial_sample_step(self) -> float:
        return max(self.grid_size / 4.0, 0.2)

    @property
    def routing_keepout_radius(self) -> float:
        return self.centerline_column_margin + self.routing_clearance_padding

    @property
    def guide_spiral_pitch(self) -> float:
        return self.main_guide_radial_step / (2.0 * np.pi)


@dataclass
class ScanCoverageResult:
    retained_square_centers: np.ndarray
    blocked_square_centers: np.ndarray
    nominal_stop_centers: np.ndarray
    stop_points: np.ndarray
    adjusted_mask: np.ndarray
    discarded_square_centers: np.ndarray


@dataclass
class CoveragePlan:
    config: PlannerConfig
    scan_result: ScanCoverageResult
    main_indices: np.ndarray
    petal_indices: np.ndarray
    visit_order: np.ndarray
    point_modes: np.ndarray
    petal_assignments: np.ndarray
    ordered_stop_points: np.ndarray
    start_path: np.ndarray
    main_path: np.ndarray
    raw_path: np.ndarray
    petal_paths: list[np.ndarray]
    petal_reference_paths: list[np.ndarray]
    full_path: np.ndarray
    petal_active: bool
    petal_axes: np.ndarray


@dataclass
class BoundaryEightSegmentPlan:
    config: PlannerConfig
    scan_result: ScanCoverageResult
    layer_levels: np.ndarray
    layer_segment_indices: list[list[np.ndarray]]
    layer_segment_points: list[list[np.ndarray]]
    main_visited_indices: np.ndarray
    remaining_indices_before_cleanup: np.ndarray
    cleanup_region_indices: list[np.ndarray]
    cleanup_region_modes: list[str]
    cleanup_transition_paths: list[np.ndarray]
    cleanup_region_paths: list[np.ndarray]
    cleanup_column_debug: list[dict[str, object]]
    cleanup_visit_order: np.ndarray
    all_visited_indices: np.ndarray
    entry_path: np.ndarray
    inter_layer_paths: list[np.ndarray]
    layer_paths: list[np.ndarray]
    main_path: np.ndarray
    boundary_path: np.ndarray
    cleanup_path: np.ndarray
    full_path: np.ndarray


@dataclass
class CleanupRegionVariant:
    visit_order: np.ndarray
    path: np.ndarray
    start_heading: float
    end_heading: float
    estimated_length: float
    strip_transitions: int
    turn_radius_ok: bool


@dataclass
class CleanupRegionCandidate:
    indices: np.ndarray
    mode: str
    orientation: str
    forward: CleanupRegionVariant
    reverse: CleanupRegionVariant


@dataclass
class StripInfo:
    strip_id: int
    coordinate: float
    indices_asc: np.ndarray
    along_values: np.ndarray
    min_along: float
    max_along: float
    run_count: int


@dataclass
class ColumnOrderResult:
    active_columns: np.ndarray
    adjacency: dict[int, tuple[int, ...]]
    start_column: int | None
    order: np.ndarray | None
    min_column_jump: int
    max_column_jump: int | None

    @property
    def success(self) -> bool:
        return self.order is not None and len(self.order) == len(self.active_columns)


def _as_point_array(points: list[np.ndarray]) -> np.ndarray:
    if not points:
        return np.empty((0, 2), dtype=float)
    return np.asarray(points, dtype=float)


def _resolve_config(config: PlannerConfig | None = None) -> PlannerConfig:
    return PlannerConfig() if config is None else config


def normalize_angle_positive(angle: np.ndarray | float) -> np.ndarray | float:
    return np.mod(angle, 2.0 * np.pi)


def wrap_to_pi(angle: np.ndarray | float) -> np.ndarray | float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def build_line_segment(start: np.ndarray, end: np.ndarray, spacing: float) -> np.ndarray:
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    distance = np.linalg.norm(end - start)
    steps = max(2, int(np.ceil(distance / max(spacing, 1e-6))) + 1)
    return np.linspace(start, end, steps)


def concatenate_paths(paths: list[np.ndarray], tol: float = 1e-9) -> np.ndarray:
    merged: list[np.ndarray] = []
    last_point: np.ndarray | None = None

    for path in paths:
        if path is None or len(path) == 0:
            continue
        array = np.asarray(path, dtype=float)
        if last_point is not None and np.linalg.norm(array[0] - last_point) <= tol:
            array = array[1:]
        if len(array) == 0:
            continue
        merged.append(array)
        last_point = array[-1]

    if not merged:
        return np.empty((0, 2), dtype=float)
    return np.vstack(merged)


def concatenate_masks(masks: list[np.ndarray]) -> np.ndarray:
    if not masks:
        return np.empty(0, dtype=bool)
    return np.concatenate(masks)


def stack_path_chunks(paths: list[np.ndarray]) -> np.ndarray:
    if not paths:
        return np.empty((0, 2), dtype=float)
    return np.vstack(paths)


def stack_heading_chunks(headings: list[np.ndarray]) -> np.ndarray:
    if not headings:
        return np.empty(0, dtype=float)
    return np.concatenate(headings)


def thin_path_samples(
    points: np.ndarray,
    fixed_mask: np.ndarray,
    min_spacing: float,
) -> tuple[np.ndarray, np.ndarray]:
    if len(points) == 0:
        return np.empty((0, 2), dtype=float), np.empty(0, dtype=bool)

    kept_points = [np.asarray(points[0], dtype=float)]
    kept_mask = [bool(fixed_mask[0])]
    last_kept = np.asarray(points[0], dtype=float)

    for idx in range(1, len(points) - 1):
        point = np.asarray(points[idx], dtype=float)
        is_fixed = bool(fixed_mask[idx])
        if is_fixed or np.linalg.norm(point - last_kept) >= min_spacing:
            kept_points.append(point)
            kept_mask.append(is_fixed)
            last_kept = point

    final_point = np.asarray(points[-1], dtype=float)
    final_fixed = bool(fixed_mask[-1])
    if np.linalg.norm(final_point - last_kept) > 1e-9 or final_fixed:
        kept_points.append(final_point)
        kept_mask.append(final_fixed)

    return np.asarray(kept_points, dtype=float), np.asarray(kept_mask, dtype=bool)


def append_path_chunk(
    path_chunks: list[np.ndarray],
    mask_chunks: list[np.ndarray],
    points: np.ndarray,
    fixed_mask: np.ndarray,
    tol: float = 1e-9,
) -> None:
    points = np.asarray(points, dtype=float)
    fixed_mask = np.asarray(fixed_mask, dtype=bool)
    if len(points) == 0:
        return

    if path_chunks and np.linalg.norm(points[0] - path_chunks[-1][-1]) <= tol:
        points = points[1:]
        fixed_mask = fixed_mask[1:]
    if len(points) == 0:
        return

    path_chunks.append(points)
    mask_chunks.append(fixed_mask)


def append_anchor_chunk(
    point_chunks: list[np.ndarray],
    mask_chunks: list[np.ndarray],
    heading_chunks: list[np.ndarray],
    points: np.ndarray,
    fixed_mask: np.ndarray,
    headings: np.ndarray,
    tol: float = 1e-9,
) -> None:
    points = np.asarray(points, dtype=float)
    fixed_mask = np.asarray(fixed_mask, dtype=bool)
    headings = np.asarray(headings, dtype=float)
    if len(points) == 0:
        return

    if point_chunks and np.linalg.norm(points[0] - point_chunks[-1][-1]) <= tol:
        points = points[1:]
        fixed_mask = fixed_mask[1:]
        headings = headings[1:]
    if len(points) == 0:
        return

    point_chunks.append(points)
    mask_chunks.append(fixed_mask)
    heading_chunks.append(headings)


def path_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


def point_to_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> tuple[float, float]:
    segment = end - start
    length_sq = float(np.dot(segment, segment))
    if length_sq <= 1e-12:
        return float(np.linalg.norm(point - start)), 0.0

    t = float(np.clip(np.dot(point - start, segment) / length_sq, 0.0, 1.0))
    closest = start + t * segment
    return float(np.linalg.norm(point - closest)), t


def segment_hits_circle(
    start: np.ndarray,
    end: np.ndarray,
    center: np.ndarray,
    radius: float,
    tol: float = 1e-6,
) -> tuple[bool, float]:
    distance, t = point_to_segment_distance(center, start, end)
    return distance < radius - tol, t


def find_blocking_columns(
    start: np.ndarray,
    end: np.ndarray,
    config: PlannerConfig,
) -> list[tuple[float, int]]:
    blockers: list[tuple[float, int]] = []

    for idx, column in enumerate(config.columns):
        hits, t = segment_hits_circle(start, end, column, config.routing_keepout_radius)
        if hits:
            blockers.append((t, idx))

    blockers.sort(key=lambda item: item[0])
    return blockers


def sample_arc(
    center: np.ndarray,
    radius: float,
    start_angle: float,
    end_angle: float,
    clockwise: bool,
    spacing: float,
) -> np.ndarray:
    if clockwise:
        if end_angle > start_angle:
            end_angle -= 2.0 * np.pi
    else:
        if end_angle < start_angle:
            end_angle += 2.0 * np.pi

    sweep = abs(end_angle - start_angle)
    steps = max(3, int(np.ceil(radius * sweep / max(spacing, 1e-6))) + 1)
    angles = np.linspace(start_angle, end_angle, steps)
    return np.column_stack((center[0] + radius * np.cos(angles), center[1] + radius * np.sin(angles)))


def build_column_detour_path(
    start: np.ndarray,
    end: np.ndarray,
    column_center: np.ndarray,
    side: int,
    config: PlannerConfig,
    depth: int,
) -> np.ndarray | None:
    clearance_radius = config.routing_keepout_radius
    start_delta = start - column_center
    end_delta = end - column_center
    start_dist = np.linalg.norm(start_delta)
    end_dist = np.linalg.norm(end_delta)

    if start_dist <= clearance_radius + 1e-6 or end_dist <= clearance_radius + 1e-6:
        return None

    start_angle = np.arctan2(start_delta[1], start_delta[0])
    end_angle = np.arctan2(end_delta[1], end_delta[0])
    start_alpha = np.arccos(np.clip(clearance_radius / start_dist, -1.0, 1.0))
    end_alpha = np.arccos(np.clip(clearance_radius / end_dist, -1.0, 1.0))

    tangent_start_angle = start_angle - side * start_alpha
    tangent_end_angle = end_angle + side * end_alpha
    tangent_start = column_center + clearance_radius * np.array(
        [np.cos(tangent_start_angle), np.sin(tangent_start_angle)],
        dtype=float,
    )
    tangent_end = column_center + clearance_radius * np.array(
        [np.cos(tangent_end_angle), np.sin(tangent_end_angle)],
        dtype=float,
    )

    if np.linalg.norm(tangent_start) > config.outermost_centerline_radius + config.routing_clearance_padding:
        return None
    if np.linalg.norm(tangent_end) > config.outermost_centerline_radius + config.routing_clearance_padding:
        return None

    approach = route_between_points(start, tangent_start, config, depth + 1)
    arc = sample_arc(
        column_center,
        clearance_radius,
        tangent_start_angle,
        tangent_end_angle,
        clockwise=side > 0,
        spacing=config.route_sample_step,
    )
    departure = route_between_points(tangent_end, end, config, depth + 1)
    return concatenate_paths([approach, arc, departure])


def route_between_points(
    start: np.ndarray,
    end: np.ndarray,
    config: PlannerConfig,
    depth: int = 0,
) -> np.ndarray:
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    direct = build_line_segment(start, end, config.route_sample_step)

    blockers = find_blocking_columns(start, end, config)
    if not blockers or depth >= 5:
        return direct

    _, blocking_idx = blockers[0]
    blocking_column = config.columns[blocking_idx]
    candidates: list[np.ndarray] = []

    for side in (-1, 1):
        candidate = build_column_detour_path(start, end, blocking_column, side, config, depth)
        if candidate is None or len(candidate) == 0:
            continue
        candidates.append(candidate)

    if not candidates:
        return direct

    return min(candidates, key=path_length)


def smooth_polyline_path(
    points: np.ndarray,
    fixed_mask: np.ndarray,
    config: PlannerConfig,
) -> np.ndarray:
    if len(points) < 3:
        return points.copy()

    smoothed = np.asarray(points, dtype=float).copy()
    fixed_mask = np.asarray(fixed_mask, dtype=bool)

    for _ in range(config.path_smoothing_passes):
        updated = smoothed.copy()
        for idx in range(1, len(smoothed) - 1):
            if fixed_mask[idx]:
                continue
            blended = 0.25 * smoothed[idx - 1] + 0.5 * smoothed[idx] + 0.25 * smoothed[idx + 1]
            updated[idx] = push_out_of_obstacles(blended, config)
        updated[fixed_mask] = points[fixed_mask]
        smoothed = updated

    return smoothed


def mod2pi(angle: float) -> float:
    return float(angle % (2.0 * np.pi))


def compute_path_headings(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.empty(0, dtype=float)
    if len(points) == 1:
        return np.array([0.0], dtype=float)

    deltas = np.gradient(points, axis=0)
    headings = np.arctan2(deltas[:, 1], deltas[:, 0])
    return headings.astype(float)


def _dubins_LSL(alpha: float, beta: float, d: float) -> tuple[float, float, float] | None:
    sin_a, sin_b = np.sin(alpha), np.sin(beta)
    cos_a, cos_b = np.cos(alpha), np.cos(beta)
    cos_ab = np.cos(alpha - beta)
    p_sq = 2.0 + d ** 2 - 2.0 * cos_ab + 2.0 * d * (sin_a - sin_b)
    if p_sq < 0.0:
        return None
    tmp = np.arctan2(cos_b - cos_a, d + sin_a - sin_b)
    t = mod2pi(-alpha + tmp)
    p = float(np.sqrt(p_sq))
    q = mod2pi(beta - tmp)
    return t, p, q


def _dubins_RSR(alpha: float, beta: float, d: float) -> tuple[float, float, float] | None:
    sin_a, sin_b = np.sin(alpha), np.sin(beta)
    cos_a, cos_b = np.cos(alpha), np.cos(beta)
    cos_ab = np.cos(alpha - beta)
    p_sq = 2.0 + d ** 2 - 2.0 * cos_ab + 2.0 * d * (-sin_a + sin_b)
    if p_sq < 0.0:
        return None
    tmp = np.arctan2(cos_a - cos_b, d - sin_a + sin_b)
    t = mod2pi(alpha - tmp)
    p = float(np.sqrt(p_sq))
    q = mod2pi(-beta + tmp)
    return t, p, q


def _dubins_LSR(alpha: float, beta: float, d: float) -> tuple[float, float, float] | None:
    sin_a, sin_b = np.sin(alpha), np.sin(beta)
    cos_a, cos_b = np.cos(alpha), np.cos(beta)
    cos_ab = np.cos(alpha - beta)
    p_sq = -2.0 + d ** 2 + 2.0 * cos_ab + 2.0 * d * (sin_a + sin_b)
    if p_sq < 0.0:
        return None
    p = float(np.sqrt(p_sq))
    tmp = np.arctan2(-cos_a - cos_b, d + sin_a + sin_b) - np.arctan2(-2.0, p)
    t = mod2pi(-alpha + tmp)
    q = mod2pi(-beta + tmp)
    return t, p, q


def _dubins_RSL(alpha: float, beta: float, d: float) -> tuple[float, float, float] | None:
    sin_a, sin_b = np.sin(alpha), np.sin(beta)
    cos_a, cos_b = np.cos(alpha), np.cos(beta)
    cos_ab = np.cos(alpha - beta)
    p_sq = d ** 2 - 2.0 + 2.0 * cos_ab - 2.0 * d * (sin_a + sin_b)
    if p_sq < 0.0:
        return None
    p = float(np.sqrt(p_sq))
    tmp = np.arctan2(cos_a + cos_b, d - sin_a - sin_b) - np.arctan2(2.0, p)
    t = mod2pi(alpha - tmp)
    q = mod2pi(beta - tmp)
    return t, p, q


def _dubins_RLR(alpha: float, beta: float, d: float) -> tuple[float, float, float] | None:
    sin_a, sin_b = np.sin(alpha), np.sin(beta)
    cos_a, cos_b = np.cos(alpha), np.cos(beta)
    cos_ab = np.cos(alpha - beta)
    tmp = (6.0 - d ** 2 + 2.0 * cos_ab + 2.0 * d * (sin_a - sin_b)) / 8.0
    if abs(tmp) > 1.0:
        return None
    p = mod2pi(2.0 * np.pi - np.arccos(tmp))
    t = mod2pi(alpha - np.arctan2(cos_a - cos_b, d - sin_a + sin_b) + p / 2.0)
    q = mod2pi(alpha - beta - t + p)
    return t, p, q


def _dubins_LRL(alpha: float, beta: float, d: float) -> tuple[float, float, float] | None:
    sin_a, sin_b = np.sin(alpha), np.sin(beta)
    cos_a, cos_b = np.cos(alpha), np.cos(beta)
    cos_ab = np.cos(alpha - beta)
    tmp = (6.0 - d ** 2 + 2.0 * cos_ab + 2.0 * d * (-sin_a + sin_b)) / 8.0
    if abs(tmp) > 1.0:
        return None
    p = mod2pi(2.0 * np.pi - np.arccos(tmp))
    t = mod2pi(-alpha - np.arctan2(cos_a - cos_b, d + sin_a - sin_b) + p / 2.0)
    q = mod2pi(mod2pi(beta) - alpha - t + p)
    return t, p, q


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
    alpha = mod2pi(start_heading - theta)
    beta = mod2pi(end_heading - theta)
    d = distance / turn_radius

    candidates: list[tuple[float, list[str], tuple[float, float, float]]] = []
    planners = [
        (["L", "S", "L"], _dubins_LSL),
        (["R", "S", "R"], _dubins_RSR),
        (["L", "S", "R"], _dubins_LSR),
        (["R", "S", "L"], _dubins_RSL),
        (["R", "L", "R"], _dubins_RLR),
        (["L", "R", "L"], _dubins_LRL),
    ]

    for mode, planner in planners:
        result = planner(alpha, beta, d)
        if result is None:
            continue
        cost = sum(result)
        candidates.append((cost, mode, result))

    if not candidates:
        return None

    _, best_mode, best_lengths = min(candidates, key=lambda item: item[0])
    return best_mode, best_lengths


def sample_dubins_path(
    start_point: np.ndarray,
    start_heading: float,
    mode: list[str],
    lengths: tuple[float, float, float],
    turn_radius: float,
    spacing: float,
) -> np.ndarray:
    start_point = np.asarray(start_point, dtype=float)
    x = float(start_point[0])
    y = float(start_point[1])
    heading = float(start_heading)
    points = [np.array([x, y], dtype=float)]

    def append_segment(segment_type: str, segment_length: float) -> None:
        nonlocal x, y, heading
        actual_length = segment_length * turn_radius
        if actual_length <= 1e-9:
            return
        step_count = max(1, int(np.ceil(actual_length / max(spacing, 1e-6))))
        ds = actual_length / step_count

        for _ in range(step_count):
            if segment_type == "S":
                x += ds * np.cos(heading)
                y += ds * np.sin(heading)
            elif segment_type == "L":
                next_heading = heading + ds / turn_radius
                x += turn_radius * (np.sin(next_heading) - np.sin(heading))
                y += -turn_radius * (np.cos(next_heading) - np.cos(heading))
                heading = next_heading
            elif segment_type == "R":
                next_heading = heading - ds / turn_radius
                x += turn_radius * (-np.sin(next_heading) + np.sin(heading))
                y += turn_radius * (np.cos(next_heading) - np.cos(heading))
                heading = next_heading
            points.append(np.array([x, y], dtype=float))

    for segment_type, segment_length in zip(mode, lengths):
        append_segment(segment_type, segment_length)

    return np.asarray(points, dtype=float)


def average_heading(angle_a: float, angle_b: float) -> float:
    return float(np.arctan2(np.sin(angle_a) + np.sin(angle_b), np.cos(angle_a) + np.cos(angle_b)))


def build_curvature_limited_segment(
    start_point: np.ndarray,
    start_heading: float,
    end_point: np.ndarray,
    end_heading: float,
    config: PlannerConfig,
    depth: int = 0,
) -> np.ndarray:
    distance = float(np.linalg.norm(np.asarray(end_point, dtype=float) - np.asarray(start_point, dtype=float)))
    planned = plan_dubins_segment(start_point, start_heading, end_point, end_heading, config.min_turn_radius)

    if planned is None:
        return build_line_segment(start_point, end_point, config.guide_point_spacing)

    mode, lengths = planned
    segment = sample_dubins_path(
        start_point,
        start_heading,
        mode,
        lengths,
        config.min_turn_radius,
        config.guide_point_spacing,
    )
    segment[-1] = np.asarray(end_point, dtype=float)

    if all(is_point_feasible(point, config) for point in segment):
        return segment

    if depth >= 8 or distance <= config.guide_point_spacing * 0.75:
        return np.asarray([push_out_of_obstacles(point, config) for point in segment], dtype=float)

    midpoint = 0.5 * (np.asarray(start_point, dtype=float) + np.asarray(end_point, dtype=float))
    midpoint = push_out_of_obstacles(midpoint, config)
    mid_heading = average_heading(start_heading, end_heading)

    first = build_curvature_limited_segment(
        start_point,
        start_heading,
        midpoint,
        mid_heading,
        config,
        depth + 1,
    )
    second = build_curvature_limited_segment(
        midpoint,
        mid_heading,
        end_point,
        end_heading,
        config,
        depth + 1,
    )
    return concatenate_paths([first, second])


def estimate_local_turn_radii(points: np.ndarray) -> np.ndarray:
    if len(points) < 3:
        return np.empty(0, dtype=float)

    radii: list[float] = []
    for idx in range(1, len(points) - 1):
        a = points[idx - 1]
        b = points[idx]
        c = points[idx + 1]
        ab = np.linalg.norm(b - a)
        bc = np.linalg.norm(c - b)
        ca = np.linalg.norm(a - c)
        area2 = abs(np.cross(b - a, c - a))
        if area2 < 1e-9:
            radii.append(np.inf)
            continue
        radius = ab * bc * ca / (2.0 * area2)
        radii.append(float(radius))
    return np.asarray(radii, dtype=float)


def square_intersects_tank(
    cx: float,
    cy: float,
    tank_radius: float = TANK_RADIUS,
    half_grid: float = HALF_GRID,
) -> bool:
    radius_sq = tank_radius ** 2
    corners = np.array(
        [
            (cx - half_grid, cy - half_grid),
            (cx + half_grid, cy - half_grid),
            (cx - half_grid, cy + half_grid),
            (cx + half_grid, cy + half_grid),
        ],
        dtype=float,
    )

    if np.any(np.sum(corners ** 2, axis=1) <= radius_sq):
        return True

    if abs(cx) <= half_grid and abs(cy) <= half_grid:
        return True

    closest_x = np.clip(0.0, cx - half_grid, cx + half_grid)
    closest_y = np.clip(0.0, cy - half_grid, cy + half_grid)
    return closest_x ** 2 + closest_y ** 2 <= radius_sq


def square_intersects_circle(
    cx: float,
    cy: float,
    circle_center: np.ndarray,
    circle_radius: float,
    half_grid: float,
) -> bool:
    closest_x = np.clip(circle_center[0], cx - half_grid, cx + half_grid)
    closest_y = np.clip(circle_center[1], cy - half_grid, cy + half_grid)
    dx = closest_x - circle_center[0]
    dy = closest_y - circle_center[1]
    return dx ** 2 + dy ** 2 <= circle_radius ** 2


def scan_square_blocked_by_columns(cx: float, cy: float, config: PlannerConfig) -> bool:
    for column in config.columns:
        if square_intersects_circle(
            cx,
            cy,
            column,
            config.scan_keepout_radius,
            config.half_grid,
        ):
            return True
    return False


def is_point_feasible(point: np.ndarray, config: PlannerConfig) -> bool:
    if np.linalg.norm(point) > config.outermost_centerline_radius + 1e-9:
        return False

    for column in config.columns:
        if np.linalg.norm(point - column) < config.centerline_column_margin - 1e-9:
            return False
    return True


def push_out_of_obstacles(point: np.ndarray, config: PlannerConfig | None = None) -> np.ndarray:
    config = _resolve_config(config)
    adjusted = np.asarray(point, dtype=float).copy()

    for _ in range(10):
        push = np.zeros(2, dtype=float)

        radial_distance = np.linalg.norm(adjusted)
        if radial_distance > config.outermost_centerline_radius:
            inward = -adjusted / max(radial_distance, 1e-9)
            push += inward * (radial_distance - config.outermost_centerline_radius + 1e-3)

        for column in config.columns:
            delta = adjusted - column
            distance = np.linalg.norm(delta)
            if distance < config.centerline_column_margin:
                if distance < 1e-9:
                    delta = 1e-6 * np.array([np.cos(config.start_angle), np.sin(config.start_angle)])
                    distance = 1e-6
                away = delta / distance
                push += away * (config.centerline_column_margin - distance + 1e-3)

        if np.linalg.norm(push) < 1e-6:
            break
        adjusted = adjusted + push

    return adjusted


def build_scan_coverage_result(config: PlannerConfig | None = None) -> ScanCoverageResult:
    config = _resolve_config(config)
    n = int(np.ceil(config.tank_radius / config.grid_size)) + 1
    retained_square_centers: list[np.ndarray] = []
    blocked_square_centers: list[np.ndarray] = []
    nominal_stop_centers: list[np.ndarray] = []
    accepted_stop_points: list[np.ndarray] = []
    adjusted_mask: list[bool] = []
    discarded_square_centers: list[np.ndarray] = []

    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            cx = i * config.grid_size
            cy = j * config.grid_size

            if not square_intersects_tank(cx, cy, config.tank_radius, config.half_grid):
                continue

            candidate = np.array([cx, cy], dtype=float)
            retained_square_centers.append(candidate)

            if scan_square_blocked_by_columns(cx, cy, config):
                blocked_square_centers.append(candidate)
                continue

            if is_point_feasible(candidate, config):
                nominal_stop_centers.append(candidate)
                accepted_stop_points.append(candidate)
                adjusted_mask.append(False)
                continue

            # Baseline rule: only keep scan stops whose original cell center is already feasible.
            discarded_square_centers.append(candidate)

    return ScanCoverageResult(
        retained_square_centers=_as_point_array(retained_square_centers),
        blocked_square_centers=_as_point_array(blocked_square_centers),
        nominal_stop_centers=_as_point_array(nominal_stop_centers),
        stop_points=_as_point_array(accepted_stop_points),
        adjusted_mask=np.asarray(adjusted_mask, dtype=bool),
        discarded_square_centers=_as_point_array(discarded_square_centers),
    )


def generate_required_scan_stop_points(config: PlannerConfig | None = None) -> np.ndarray:
    return build_scan_coverage_result(config).stop_points


def moving_average_2d(points: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return points.copy()

    kernel = np.ones(window, dtype=float) / window
    pad = window // 2
    padded = np.pad(points, ((pad, pad), (0, 0)), mode="edge")
    x = np.convolve(padded[:, 0], kernel, mode="valid")
    y = np.convolve(padded[:, 1], kernel, mode="valid")
    return np.column_stack((x, y))


def build_base_spiral(config: PlannerConfig | None = None) -> np.ndarray:
    config = _resolve_config(config)
    start_angle = np.arctan2(MANWAY[1], MANWAY[0])
    radial_pitch = config.pass_spacing / (2.0 * np.pi)

    theta = [start_angle]
    while True:
        current_radius = config.outermost_centerline_radius - radial_pitch * (theta[-1] - start_angle)
        if current_radius <= config.centerline_column_margin * 0.7:
            break
        dtheta = config.sample_arc_length / max(current_radius, 0.35)
        theta.append(theta[-1] + dtheta)

    theta = np.array(theta)
    radius = config.outermost_centerline_radius - radial_pitch * (theta - start_angle)
    radius = np.clip(radius, config.centerline_column_margin * 0.6, None)
    spiral = np.column_stack((radius * np.cos(theta), radius * np.sin(theta)))
    return np.vstack((MANWAY, spiral))


def project_spiral_to_free_space(path: np.ndarray, config: PlannerConfig | None = None) -> np.ndarray:
    config = _resolve_config(config)
    projected = np.zeros_like(path)
    projected[0] = MANWAY

    for idx in range(1, len(path)):
        candidate = push_out_of_obstacles(path[idx], config)
        prev = projected[idx - 1]
        step = candidate - prev
        step_norm = np.linalg.norm(step)

        max_step = config.sample_arc_length * 1.6
        if step_norm > max_step:
            candidate = prev + step * (max_step / step_norm)
            candidate = push_out_of_obstacles(candidate, config)

        projected[idx] = candidate

    projected = moving_average_2d(projected, config.smoothing_window)
    projected[0] = MANWAY
    for idx in range(1, len(projected)):
        projected[idx] = push_out_of_obstacles(projected[idx], config)
    return projected


def compute_tangent_and_inward_normal(path: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tangent = np.gradient(path, axis=0)
    tangent_norm = np.linalg.norm(tangent, axis=1, keepdims=True)
    tangent = tangent / np.clip(tangent_norm, 1e-9, None)

    left_normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    toward_center = -path
    choose_left = np.sum(left_normal * toward_center, axis=1, keepdims=True) >= 0.0
    inward_normal = np.where(choose_left, left_normal, -left_normal)
    inward_norm = np.linalg.norm(inward_normal, axis=1, keepdims=True)
    inward_normal = inward_normal / np.clip(inward_norm, 1e-9, None)
    return tangent, inward_normal


def build_coverage_path(config: PlannerConfig | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base_spiral = build_base_spiral(config)
    path = project_spiral_to_free_space(base_spiral, config)
    tangent, inward_normal = compute_tangent_and_inward_normal(path)
    return path, tangent, inward_normal


def order_points_along_spiral(points: np.ndarray, config: PlannerConfig) -> np.ndarray:
    if len(points) == 0:
        return np.empty(0, dtype=int)

    radii = np.linalg.norm(points, axis=1)
    angles = normalize_angle_positive(np.arctan2(points[:, 1], points[:, 0]) - config.start_angle)

    max_loops = max(1, int(np.ceil((config.outermost_centerline_radius - radii.min()) / config.grid_size)) + 3)
    loops = np.arange(max_loops + 1, dtype=float)
    theta_candidates = angles[:, None] + 2.0 * np.pi * loops[None, :]
    expected_radii = config.outermost_centerline_radius - config.spiral_pitch * theta_candidates
    errors = np.abs(radii[:, None] - expected_radii)
    errors = np.where(expected_radii >= -config.grid_size, errors, np.inf)
    best_loop = np.argmin(errors, axis=1)
    phases = theta_candidates[np.arange(len(points)), best_loop]
    return np.lexsort((angles, -radii, phases))


def build_start_path(config: PlannerConfig) -> np.ndarray:
    start_center = np.array([0.0, 0.0], dtype=float)
    feasible_start = push_out_of_obstacles(start_center, config)
    outer_target = config.outermost_centerline_radius * np.array(
        [np.cos(config.start_angle), np.sin(config.start_angle)],
        dtype=float,
    )
    outer_start = push_out_of_obstacles(outer_target, config)

    if np.linalg.norm(feasible_start - start_center) <= 1e-9:
        origin_segment = start_center[None, :]
    else:
        origin_segment = feasible_start[None, :]
    outward_segment = build_line_segment(feasible_start, outer_start, config.guide_point_spacing)
    return concatenate_paths([origin_segment, outward_segment])


def build_main_reference_path(config: PlannerConfig, inner_radius: float) -> np.ndarray:
    spiral_path, _, _ = build_coverage_path(config)
    if len(spiral_path) == 0:
        return np.empty((0, 2), dtype=float)

    radii = np.linalg.norm(spiral_path, axis=1)
    keep_mask = radii >= inner_radius - config.grid_size * 0.5
    trimmed = spiral_path[keep_mask]

    if len(trimmed) == 0:
        target = inner_radius * np.array([np.cos(config.start_angle), np.sin(config.start_angle)], dtype=float)
        return push_out_of_obstacles(target, config)[None, :]

    last_radius = np.linalg.norm(trimmed[-1])
    if last_radius > inner_radius + config.grid_size * 0.35:
        target = inner_radius * np.array([np.cos(config.start_angle), np.sin(config.start_angle)], dtype=float)
        trimmed = concatenate_paths([trimmed, push_out_of_obstacles(target, config)[None, :]])

    return trimmed


def merge_points_into_reference(
    reference_path: np.ndarray,
    ordered_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(reference_path) == 0 and len(ordered_points) == 0:
        return np.empty((0, 2), dtype=float), np.empty(0, dtype=bool), np.empty(0, dtype=float)
    if len(reference_path) == 0:
        return ordered_points.copy(), np.ones(len(ordered_points), dtype=bool), compute_path_headings(ordered_points)
    if len(ordered_points) == 0:
        return reference_path.copy(), np.zeros(len(reference_path), dtype=bool), compute_path_headings(reference_path)

    distances = np.linalg.norm(ordered_points[:, None, :] - reference_path[None, :, :], axis=2)
    nearest_indices = np.argmin(distances, axis=1)
    nearest_indices = np.maximum.accumulate(nearest_indices)
    reference_headings = compute_path_headings(reference_path)

    anchors: list[np.ndarray] = []
    fixed_flags: list[bool] = []
    anchor_headings: list[float] = []
    point_cursor = 0

    for ref_idx, ref_point in enumerate(reference_path):
        anchors.append(ref_point)
        fixed_flags.append(False)
        anchor_headings.append(float(reference_headings[ref_idx]))

        while point_cursor < len(ordered_points) and nearest_indices[point_cursor] == ref_idx:
            anchors.append(ordered_points[point_cursor])
            fixed_flags.append(True)
            anchor_headings.append(float(reference_headings[ref_idx]))
            point_cursor += 1

    while point_cursor < len(ordered_points):
        anchors.append(ordered_points[point_cursor])
        fixed_flags.append(True)
        anchor_headings.append(float(reference_headings[-1]))
        point_cursor += 1

    return (
        np.asarray(anchors, dtype=float),
        np.asarray(fixed_flags, dtype=bool),
        np.asarray(anchor_headings, dtype=float),
    )


def build_path_from_anchor_sequence(
    anchor_points: np.ndarray,
    fixed_mask: np.ndarray,
    config: PlannerConfig,
    anchor_headings: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if len(anchor_points) == 0:
        return np.empty((0, 2), dtype=float), np.empty(0, dtype=bool)

    path_chunks: list[np.ndarray] = []
    mask_chunks: list[np.ndarray] = []

    first_mask = np.zeros(1, dtype=bool)
    first_mask[0] = fixed_mask[0]
    append_path_chunk(path_chunks, mask_chunks, anchor_points[:1], first_mask)

    for idx in range(len(anchor_points) - 1):
        segment = route_between_points(anchor_points[idx], anchor_points[idx + 1], config)

        segment_mask = np.zeros(len(segment), dtype=bool)
        segment_mask[0] = fixed_mask[idx]
        segment_mask[-1] = fixed_mask[idx + 1]
        append_path_chunk(path_chunks, mask_chunks, segment, segment_mask)

    raw_path = stack_path_chunks(path_chunks)
    raw_mask = concatenate_masks(mask_chunks)
    thinned_path, thinned_mask = thin_path_samples(raw_path, raw_mask, config.route_sample_step * 0.7)
    smoothed_path = smooth_polyline_path(thinned_path, thinned_mask, config)
    return smoothed_path, thinned_mask


def build_single_petal_reference(axis_angle: float, config: PlannerConfig) -> np.ndarray:
    half_span = np.pi / config.petal_count
    arc_length = max(np.pi * config.trigger_radius, 1.0)
    sample_count = max(PETAL_REFERENCE_SAMPLES, int(np.ceil(arc_length / config.guide_point_spacing)) + 1)
    s = np.linspace(0.0, 1.0, sample_count)
    angles = axis_angle - half_span + 2.0 * half_span * s
    radii = config.trigger_radius * (np.cos(np.pi * s) ** 2)
    path = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
    return np.asarray([push_out_of_obstacles(point, config) for point in path], dtype=float)


def assign_points_to_petals(points: np.ndarray, axis_angles: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.empty(0, dtype=int)

    point_angles = np.arctan2(points[:, 1], points[:, 0])
    angle_diffs = np.abs(wrap_to_pi(point_angles[:, None] - axis_angles[None, :]))
    return np.argmin(angle_diffs, axis=1)


def order_points_along_reference(points: np.ndarray, reference_path: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.empty(0, dtype=int)

    distances = np.linalg.norm(points[:, None, :] - reference_path[None, :, :], axis=2)
    nearest = np.argmin(distances, axis=1)
    radii = np.linalg.norm(points, axis=1)
    return np.lexsort((-radii, nearest))


def build_mandatory_stop_order(
    stop_points: np.ndarray,
    config: PlannerConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray], bool, np.ndarray]:
    if len(stop_points) == 0:
        empty_idx = np.empty(0, dtype=int)
        return empty_idx, empty_idx, np.empty(0, dtype=int), [], [], False, np.empty(0, dtype=float)

    radii = np.linalg.norm(stop_points, axis=1)
    inner_mask = radii <= config.trigger_radius + 1e-9

    if not config.petal_enabled or np.count_nonzero(inner_mask) == 0:
        all_order = order_points_along_spiral(stop_points, config)
        return (
            all_order,
            np.empty(0, dtype=int),
            all_order,
            [],
            [],
            False,
            np.empty(0, dtype=float),
        )

    main_indices = np.flatnonzero(~inner_mask)
    petal_indices = np.flatnonzero(inner_mask)
    ordered_main = main_indices[order_points_along_spiral(stop_points[main_indices], config)]

    trigger_angle = config.start_angle + (config.outermost_centerline_radius - config.trigger_radius) / config.spiral_pitch
    first_axis = trigger_angle + np.pi / config.petal_count
    petal_axes = normalize_angle_positive(
        first_axis + np.arange(config.petal_count, dtype=float) * (2.0 * np.pi / config.petal_count)
    )

    petal_reference_paths = [build_single_petal_reference(angle, config) for angle in petal_axes]
    petal_assignments = assign_points_to_petals(stop_points[petal_indices], petal_axes)
    ordered_petals: list[np.ndarray] = []

    for petal_id, reference_path in enumerate(petal_reference_paths):
        local_mask = petal_assignments == petal_id
        local_global_indices = petal_indices[local_mask]
        local_points = stop_points[local_global_indices]
        local_order = order_points_along_reference(local_points, reference_path)
        ordered_petals.append(local_global_indices[local_order])

    ordered_visit = np.concatenate([ordered_main, *ordered_petals]) if ordered_petals else ordered_main
    ordered_petal_indices = np.concatenate(ordered_petals) if ordered_petals else np.empty(0, dtype=int)
    return (
        ordered_main,
        ordered_petal_indices,
        ordered_visit,
        ordered_petals,
        petal_reference_paths,
        True,
        petal_axes,
    )


def build_circular_coverage_plan(config: PlannerConfig | None = None) -> CoveragePlan:
    config = _resolve_config(config)
    scan_result = build_scan_coverage_result(config)
    stop_points = scan_result.stop_points

    (
        main_indices,
        petal_indices,
        visit_order,
        petal_index_groups,
        petal_reference_paths,
        petal_active,
        petal_axes,
    ) = build_mandatory_stop_order(stop_points, config)

    ordered_stop_points = stop_points[visit_order] if len(visit_order) else np.empty((0, 2), dtype=float)
    start_path = build_start_path(config)
    start_mask = np.zeros(len(start_path), dtype=bool)
    if len(start_mask):
        start_mask[0] = True
    start_headings = compute_path_headings(start_path)

    anchor_chunks: list[np.ndarray] = []
    anchor_mask_chunks: list[np.ndarray] = []
    anchor_heading_chunks: list[np.ndarray] = []
    append_anchor_chunk(anchor_chunks, anchor_mask_chunks, anchor_heading_chunks, start_path, start_mask, start_headings)

    petal_paths: list[np.ndarray] = []
    main_path = np.empty((0, 2), dtype=float)

    if len(main_indices):
        main_points = stop_points[main_indices]
        inner_radius = config.trigger_radius if petal_active else float(np.min(np.linalg.norm(main_points, axis=1)))
        main_path = build_main_reference_path(config, inner_radius)
        main_anchors, main_fixed, main_headings = merge_points_into_reference(main_path, main_points)
        append_anchor_chunk(
            anchor_chunks,
            anchor_mask_chunks,
            anchor_heading_chunks,
            main_anchors,
            main_fixed,
            main_headings,
        )

    if petal_active:
        for group, reference_path in zip(petal_index_groups, petal_reference_paths):
            mandatory_points = stop_points[group] if len(group) else np.empty((0, 2), dtype=float)
            petal_anchors, petal_fixed, petal_headings = merge_points_into_reference(reference_path, mandatory_points)
            if len(petal_anchors) == 0:
                petal_paths.append(np.empty((0, 2), dtype=float))
                continue
            append_anchor_chunk(
                anchor_chunks,
                anchor_mask_chunks,
                anchor_heading_chunks,
                petal_anchors,
                petal_fixed,
                petal_headings,
            )
            petal_paths.append(petal_anchors)

    combined_anchors = stack_path_chunks(anchor_chunks)
    combined_anchor_mask = concatenate_masks(anchor_mask_chunks)
    combined_anchor_headings = stack_heading_chunks(anchor_heading_chunks)
    raw_path, raw_mask = build_path_from_anchor_sequence(
        combined_anchors,
        combined_anchor_mask,
        config,
        combined_anchor_headings,
    )
    full_path = raw_path

    point_modes = np.full(len(stop_points), "main", dtype=object)
    petal_assignments = np.full(len(stop_points), -1, dtype=int)
    if petal_active and len(petal_indices):
        point_modes[petal_indices] = "petal"
        for petal_id, group in enumerate(petal_index_groups):
            petal_assignments[group] = petal_id

    return CoveragePlan(
        config=config,
        scan_result=scan_result,
        main_indices=main_indices,
        petal_indices=petal_indices,
        visit_order=visit_order,
        point_modes=point_modes,
        petal_assignments=petal_assignments,
        ordered_stop_points=ordered_stop_points,
        start_path=start_path,
        main_path=main_path,
        raw_path=raw_path,
        petal_paths=petal_paths,
        petal_reference_paths=petal_reference_paths,
        full_path=full_path,
        petal_active=petal_active,
        petal_axes=petal_axes,
    )


def validate_coverage_plan(plan: CoveragePlan) -> dict[str, object]:
    config = plan.config
    stop_points = plan.scan_result.stop_points
    feasible_points = np.array([is_point_feasible(point, config) for point in stop_points], dtype=bool)
    feasible_path = np.array([is_point_feasible(point, config) for point in plan.full_path], dtype=bool)
    turn_radii = estimate_local_turn_radii(plan.full_path)
    finite_turn_radii = turn_radii[np.isfinite(turn_radii)]

    no_center_columns = np.asarray([column for column in config.columns if np.linalg.norm(column) > 1e-9], dtype=float)
    if no_center_columns.size == 0:
        no_center_columns = np.empty((0, 2), dtype=float)
    no_center_plan = build_circular_coverage_plan(
        PlannerConfig(
            tank_radius=config.tank_radius,
            grid_size=config.grid_size,
            min_turn_radius=config.min_turn_radius,
            trigger_radius=config.trigger_radius,
            petal_count=config.petal_count,
            petal_enabled=config.petal_enabled,
            start_angle=config.start_angle,
            pass_spacing=config.pass_spacing,
            sample_arc_length=config.sample_arc_length,
            smoothing_window=config.smoothing_window,
            column_radius=config.column_radius,
            column_clearance=config.column_clearance,
            robot_width=config.robot_width,
            max_center_shift=config.max_center_shift,
            columns=no_center_columns,
        )
    )

    blocking_center_plan = build_circular_coverage_plan(
        PlannerConfig(
            tank_radius=config.tank_radius,
            grid_size=config.grid_size,
            min_turn_radius=config.min_turn_radius,
            trigger_radius=config.trigger_radius,
            petal_count=config.petal_count,
            petal_enabled=config.petal_enabled,
            start_angle=config.start_angle,
            pass_spacing=config.pass_spacing,
            sample_arc_length=config.sample_arc_length,
            smoothing_window=config.smoothing_window,
            column_radius=max(config.column_radius, config.trigger_radius * 0.8),
            column_clearance=config.column_clearance,
            robot_width=config.robot_width,
            max_center_shift=config.max_center_shift,
            columns=np.array([[0.0, 0.0]], dtype=float),
        )
    )

    return {
        "all_scan_points_feasible": bool(np.all(feasible_points)),
        "all_path_samples_feasible": bool(np.all(feasible_path)),
        "min_turn_radius_respected": bool(
            len(finite_turn_radii) == 0 or np.min(finite_turn_radii) >= config.min_turn_radius - 1e-6
        ),
        "petal_mode_needed": bool(np.any(np.linalg.norm(stop_points, axis=1) <= config.trigger_radius + 1e-9)),
        "petal_mode_active": bool(plan.petal_active),
        "visit_order_covers_all_points": bool(
            len(plan.visit_order) == len(stop_points)
            and np.array_equal(np.sort(plan.visit_order), np.arange(len(stop_points)))
        ),
        "no_center_column_petals_collect_inner_points": bool(
            (not np.any(np.linalg.norm(no_center_plan.scan_result.stop_points, axis=1) <= config.trigger_radius + 1e-9))
            or no_center_plan.petal_active
        ),
        "blocking_center_column_can_deactivate_petals": bool(not blocking_center_plan.petal_active),
        "default_inner_point_count": int(np.count_nonzero(np.linalg.norm(stop_points, axis=1) <= config.trigger_radius + 1e-9)),
        "no_center_inner_point_count": int(
            np.count_nonzero(np.linalg.norm(no_center_plan.scan_result.stop_points, axis=1) <= config.trigger_radius + 1e-9)
        ),
        "blocking_center_inner_point_count": int(
            np.count_nonzero(np.linalg.norm(blocking_center_plan.scan_result.stop_points, axis=1) <= config.trigger_radius + 1e-9)
        ),
        "minimum_sampled_turn_radius": float(np.min(finite_turn_radii)) if len(finite_turn_radii) else float("inf"),
    }


def plot_coverage_plan(plan: CoveragePlan) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    config = plan.config
    stop_points = plan.scan_result.stop_points
    fig, ax = plt.subplots(figsize=(10, 10))

    ax.add_patch(Circle((0.0, 0.0), config.tank_radius, fill=False, color="black", linewidth=2.0))
    ax.add_patch(
        Circle(
            (0.0, 0.0),
            config.outermost_centerline_radius,
            fill=False,
            color="#9a3412",
            linestyle="--",
            linewidth=1.5,
        )
    )

    for column in config.columns:
        ax.add_patch(Circle(column, config.scan_keepout_radius, color="#ef4444", alpha=0.18))
        ax.add_patch(
            Circle(
                column,
                config.centerline_column_margin,
                fill=False,
                color="#dc2626",
                linestyle=":",
                linewidth=1.2,
            )
        )

    if len(stop_points):
        main_points = stop_points[plan.main_indices] if len(plan.main_indices) else np.empty((0, 2), dtype=float)
        petal_points = stop_points[plan.petal_indices] if len(plan.petal_indices) else np.empty((0, 2), dtype=float)

        if len(main_points):
            ax.scatter(
                main_points[:, 0],
                main_points[:, 1],
                s=24,
                color="#2563eb",
                label="Main-traversal mandatory stops",
                zorder=5,
            )
        if len(petal_points):
            ax.scatter(
                petal_points[:, 0],
                petal_points[:, 1],
                s=42,
                marker="D",
                color="#059669",
                label="Petal-mode mandatory stops",
                zorder=6,
            )

    if len(plan.start_path):
        ax.plot(
            plan.start_path[:, 0],
            plan.start_path[:, 1],
            color="#9ca3af",
            linewidth=1.2,
            linestyle="--",
            label="Center-to-edge guide",
            zorder=1,
        )

    if len(plan.main_path):
        ax.plot(
            plan.main_path[:, 0],
            plan.main_path[:, 1],
            color="#60a5fa",
            linewidth=1.2,
            alpha=0.7,
            linestyle="--",
            label="Main circular guide",
            zorder=1,
        )

    for idx, petal_path in enumerate(plan.petal_reference_paths):
        ax.plot(
            petal_path[:, 0],
            petal_path[:, 1],
            color="#10b981",
            linewidth=1.0,
            linestyle="--",
            alpha=0.6,
            label="Petal guide geometry" if idx == 0 else None,
            zorder=1,
        )

    if len(plan.full_path) > 1:
        ax.plot(
            plan.full_path[:, 0],
            plan.full_path[:, 1],
            color="#111827",
            linewidth=2.6,
            label="Robot motion path",
            zorder=4,
        )

        deltas = np.diff(plan.full_path, axis=0)
        arrow_step = max(1, len(deltas) // 32)
        arrow_positions = plan.full_path[:-1:arrow_step]
        arrow_vectors = deltas[::arrow_step]
        ax.quiver(
            arrow_positions[:, 0],
            arrow_positions[:, 1],
            arrow_vectors[:, 0],
            arrow_vectors[:, 1],
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.003,
            headlength=6.0,
            headaxislength=5.4,
            headwidth=5.0,
            color="#111827",
            alpha=0.8,
            zorder=7,
        )

    if len(plan.full_path):
        ax.scatter(
            plan.full_path[0, 0],
            plan.full_path[0, 1],
            s=70,
            marker="o",
            color="#f97316",
            edgecolors="black",
            linewidths=0.5,
            label="Actual motion start",
            zorder=8,
        )

    if len(plan.ordered_stop_points):
        ax.scatter(
            plan.ordered_stop_points[0, 0],
            plan.ordered_stop_points[0, 1],
            s=70,
            marker="*",
            color="#f59e0b",
            edgecolors="black",
            linewidths=0.5,
            label="First mandatory stop",
            zorder=8,
        )
        ax.scatter(
            plan.ordered_stop_points[-1, 0],
            plan.ordered_stop_points[-1, 1],
            s=65,
            marker="P",
            color="#7c3aed",
            edgecolors="black",
            linewidths=0.4,
            label="Final mandatory stop",
            zorder=8,
        )

    ax.scatter(0.0, 0.0, s=45, color="black", label="Tank center", zorder=8)
    ax.set_aspect("equal", adjustable="box")
    margin = config.grid_size * 1.5
    ax.set_xlim(-(config.tank_radius + margin), config.tank_radius + margin)
    ax.set_ylim(-(config.tank_radius + margin), config.tank_radius + margin)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Circular tank coverage planner with mandatory scan stops")
    ax.grid(True, alpha=0.22)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


def plot_required_scan_points(config: PlannerConfig | None = None) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    config = _resolve_config(config)
    scan_result = build_scan_coverage_result(config)

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.add_patch(Circle((0.0, 0.0), config.tank_radius, fill=False, color="black", linewidth=2.0))
    ax.add_patch(
        Circle(
            (0.0, 0.0),
            config.outermost_centerline_radius,
            fill=False,
            color="#9a3412",
            linestyle="--",
            linewidth=1.3,
        )
    )

    for column in config.columns:
        ax.add_patch(Circle(column, config.scan_keepout_radius, color="#ef4444", alpha=0.18))
        ax.add_patch(
            Circle(
                column,
                config.centerline_column_margin,
                fill=False,
                color="#dc2626",
                linestyle=":",
                linewidth=1.1,
            )
        )

    if len(scan_result.retained_square_centers):
        ax.scatter(
            scan_result.retained_square_centers[:, 0],
            scan_result.retained_square_centers[:, 1],
            s=14,
            color="#d1d5db",
            alpha=0.45,
            label="Tank-intersecting cell centers",
            zorder=2,
        )

    if len(scan_result.blocked_square_centers):
        ax.scatter(
            scan_result.blocked_square_centers[:, 0],
            scan_result.blocked_square_centers[:, 1],
            s=32,
            marker="x",
            color="#dc2626",
            alpha=0.9,
            label="Blocked by column keep-out",
            zorder=4,
        )

    if len(scan_result.discarded_square_centers):
        ax.scatter(
            scan_result.discarded_square_centers[:, 0],
            scan_result.discarded_square_centers[:, 1],
            s=30,
            marker="x",
            color="#7c3aed",
            alpha=0.9,
            label="Discarded infeasible centers",
            zorder=4,
        )

    if len(scan_result.stop_points):
        ax.scatter(
            scan_result.stop_points[:, 0],
            scan_result.stop_points[:, 1],
            s=38,
            color="#1d4ed8",
            edgecolors="white",
            linewidths=0.4,
            label="Required robot visit points",
            zorder=5,
        )

    ax.scatter(0.0, 0.0, s=42, color="black", label="Tank center", zorder=6)
    margin = config.grid_size * 1.5
    ax.set_xlim(-(config.tank_radius + margin), config.tank_radius + margin)
    ax.set_ylim(-(config.tank_radius + margin), config.tank_radius + margin)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Required robot visit points")
    ax.grid(True, alpha=0.22)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


def _segment_points_clockwise(points: np.ndarray, mask: np.ndarray, x_sign: int, y_sign: int) -> np.ndarray:
    selected = points[mask]
    if len(selected) == 0:
        return np.empty((0, 2), dtype=float)
    order_value = x_sign * selected[:, 0] - y_sign * selected[:, 1]
    return selected[np.argsort(order_value)]


def _ordered_boundary_segments(stop_points: np.ndarray) -> list[np.ndarray]:
    x = stop_points[:, 0]
    y = stop_points[:, 1]
    top_y = float(np.max(y))
    bottom_y = float(np.min(y))
    left_x = float(np.min(x))
    right_x = float(np.max(x))

    top = stop_points[np.isclose(y, top_y)]
    top = top[np.argsort(top[:, 0])]
    top_right = top[-1]

    right = stop_points[np.isclose(x, right_x)]
    right = right[np.argsort(-right[:, 1])]
    right_bottom = right[-1]

    bottom = stop_points[np.isclose(y, bottom_y)]
    bottom = bottom[np.argsort(-bottom[:, 0])]
    bottom_left = bottom[-1]

    left = stop_points[np.isclose(x, left_x)]
    left = left[np.argsort(left[:, 1])]
    left_top = left[-1]

    def select_corner_between(edge_start: np.ndarray, edge_end: np.ndarray) -> np.ndarray:
        min_x = min(edge_start[0], edge_end[0]) + 1e-9
        max_x = max(edge_start[0], edge_end[0]) - 1e-9
        min_y = min(edge_start[1], edge_end[1]) + 1e-9
        max_y = max(edge_start[1], edge_end[1]) - 1e-9
        mask = (x > min_x) & (x < max_x) & (y > min_y) & (y < max_y)
        candidates = stop_points[mask]
        if len(candidates) == 0:
            return np.empty((0, 2), dtype=float)

        delta = edge_end - edge_start
        if abs(delta[0]) <= 1e-9 or abs(delta[1]) <= 1e-9:
            return np.empty((0, 2), dtype=float)

        progress_x = (candidates[:, 0] - edge_start[0]) / delta[0]
        progress_y = (candidates[:, 1] - edge_start[1]) / delta[1]
        diagonal_error = np.abs(progress_x - progress_y)
        radii = np.linalg.norm(candidates, axis=1)
        score_order = np.lexsort((diagonal_error, -radii))
        ranked = candidates[score_order]
        ranked_error = diagonal_error[score_order]

        if len(ranked) == 0:
            return np.empty((0, 2), dtype=float)

        chosen: list[np.ndarray] = []
        for point, error in zip(ranked, ranked_error):
            if error > 0.55:
                continue
            if any(np.linalg.norm(point - existing) <= 1e-9 for existing in chosen):
                continue
            chosen.append(point)
            if len(chosen) == 4:
                break

        if not chosen:
            return np.empty((0, 2), dtype=float)

        chosen_array = np.asarray(chosen, dtype=float)
        progress = 0.5 * (
            (chosen_array[:, 0] - edge_start[0]) / delta[0] + (chosen_array[:, 1] - edge_start[1]) / delta[1]
        )
        return chosen_array[np.argsort(progress)]

    upper_right = select_corner_between(top_right, right[0])
    lower_right = select_corner_between(right_bottom, bottom[0])
    lower_left = select_corner_between(bottom_left, left[0])
    upper_left = select_corner_between(left_top, top[0])

    return [top, upper_right, right, lower_right, bottom, lower_left, left, upper_left]


def _segment_indices_from_points(stop_points: np.ndarray, segment_points: np.ndarray) -> np.ndarray:
    indices: list[int] = []
    for point in segment_points:
        match = np.flatnonzero(np.all(np.isclose(stop_points, point, atol=1e-9), axis=1))
        if len(match):
            indices.append(int(match[0]))
    return np.asarray(indices, dtype=int)


def _dedupe_consecutive_points(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return points
    kept = [points[0]]
    for point in points[1:]:
        if np.linalg.norm(point - kept[-1]) > 1e-9:
            kept.append(point)
    return np.asarray(kept, dtype=float)


def _filter_corner_progression(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return points

    kept = [points[0]]
    for point in points[1:]:
        delta = point - kept[-1]
        if abs(delta[0]) <= 1e-9 or abs(delta[1]) <= 1e-9:
            continue
        kept.append(point)
    return np.asarray(kept, dtype=float)


def _remove_points(source_points: np.ndarray, points_to_remove: np.ndarray) -> np.ndarray:
    if len(source_points) == 0 or len(points_to_remove) == 0:
        return source_points.copy()

    keep_mask = np.ones(len(source_points), dtype=bool)
    for point in points_to_remove:
        matches = np.all(np.isclose(source_points, point, atol=1e-9), axis=1)
        keep_mask &= ~matches
    return source_points[keep_mask]


def _build_polar_connector(start: np.ndarray, end: np.ndarray, spacing: float) -> np.ndarray:
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    if np.linalg.norm(end - start) <= 1e-9:
        return start[None, :]

    start_angle = float(np.arctan2(start[1], start[0]))
    end_angle = float(np.arctan2(end[1], end[0]))
    delta_angle = float(wrap_to_pi(end_angle - start_angle))
    radius_start = float(np.linalg.norm(start))
    radius_end = float(np.linalg.norm(end))
    mean_radius = max(0.5 * (radius_start + radius_end), 1e-6)
    sweep_length = max(abs(delta_angle) * mean_radius, np.linalg.norm(end - start))
    steps = max(3, int(np.ceil(sweep_length / max(spacing, 1e-6))) + 1)

    angles = np.linspace(start_angle, start_angle + delta_angle, steps)
    radii = np.linspace(radius_start, radius_end, steps)
    path = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
    path[0] = start
    path[-1] = end
    return path


def _build_tangent_guided_segment(start: np.ndarray, end: np.ndarray, spacing: float) -> np.ndarray:
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    chord = np.linalg.norm(end - start)
    if chord <= 1e-9:
        return start[None, :]

    start_radius = float(np.linalg.norm(start))
    end_radius = float(np.linalg.norm(end))
    if start_radius <= 1e-9 or end_radius <= 1e-9:
        return build_line_segment(start, end, spacing)

    start_angle = float(np.arctan2(start[1], start[0]))
    end_angle = float(np.arctan2(end[1], end[0]))
    delta_angle = float(wrap_to_pi(end_angle - start_angle))
    if abs(delta_angle) <= 1e-6:
        return build_line_segment(start, end, spacing)

    clockwise = delta_angle < 0.0
    if clockwise:
        start_tangent_dir = np.array([start[1], -start[0]], dtype=float) / start_radius
        end_tangent_dir = np.array([end[1], -end[0]], dtype=float) / end_radius
    else:
        start_tangent_dir = np.array([-start[1], start[0]], dtype=float) / start_radius
        end_tangent_dir = np.array([-end[1], end[0]], dtype=float) / end_radius

    tangent_scale = max(0.5 * chord, 0.5 * (start_radius + end_radius) * abs(delta_angle))
    m0 = start_tangent_dir * tangent_scale
    m1 = end_tangent_dir * tangent_scale

    estimated_length = max(chord, 0.5 * (start_radius + end_radius) * abs(delta_angle))
    steps = max(4, int(np.ceil(estimated_length / max(spacing, 1e-6))) + 1)
    t = np.linspace(0.0, 1.0, steps)
    h00 = 2.0 * t**3 - 3.0 * t**2 + 1.0
    h10 = t**3 - 2.0 * t**2 + t
    h01 = -2.0 * t**3 + 3.0 * t**2
    h11 = t**3 - t**2
    path = (
        h00[:, None] * start
        + h10[:, None] * m0
        + h01[:, None] * end
        + h11[:, None] * m1
    )
    path[0] = start
    path[-1] = end
    return path


def _heading_between_points(start: np.ndarray, end: np.ndarray) -> float:
    delta = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
    if np.linalg.norm(delta) <= 1e-9:
        return 0.0
    return float(np.arctan2(delta[1], delta[0]))


def _segment_start_heading(segment: np.ndarray, fallback_target: np.ndarray | None = None) -> float:
    segment = np.asarray(segment, dtype=float)
    if len(segment) >= 2:
        return _heading_between_points(segment[0], segment[1])
    if len(segment) == 1 and fallback_target is not None:
        return _heading_between_points(segment[0], fallback_target)
    return 0.0


def _segment_end_heading(segment: np.ndarray, fallback_source: np.ndarray | None = None) -> float:
    segment = np.asarray(segment, dtype=float)
    if len(segment) >= 2:
        return _heading_between_points(segment[-2], segment[-1])
    if len(segment) == 1 and fallback_source is not None:
        return _heading_between_points(fallback_source, segment[-1])
    return 0.0


def _connector_is_straight(
    start: np.ndarray,
    start_heading: float,
    end: np.ndarray,
    end_heading: float,
    angle_tolerance: float = np.deg2rad(5.0),
) -> bool:
    chord = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
    if np.linalg.norm(chord) <= 1e-9:
        return True

    chord_heading = float(np.arctan2(chord[1], chord[0]))
    return (
        abs(float(wrap_to_pi(start_heading - chord_heading))) <= angle_tolerance
        and abs(float(wrap_to_pi(end_heading - chord_heading))) <= angle_tolerance
    )


def _trim_short_terminal_steps(path: np.ndarray, min_step: float) -> np.ndarray:
    path = np.asarray(path, dtype=float)
    if len(path) <= 2:
        return path

    trimmed = path.copy()
    while len(trimmed) > 2 and np.linalg.norm(trimmed[1] - trimmed[0]) < min_step:
        trimmed = np.delete(trimmed, 1, axis=0)
    while len(trimmed) > 2 and np.linalg.norm(trimmed[-1] - trimmed[-2]) < min_step:
        trimmed = np.delete(trimmed, -2, axis=0)
    return trimmed


def _thin_connector_samples(path: np.ndarray, min_step: float) -> np.ndarray:
    path = np.asarray(path, dtype=float)
    if len(path) <= 2:
        return path
    fixed_mask = np.zeros(len(path), dtype=bool)
    fixed_mask[0] = True
    fixed_mask[-1] = True
    thinned, _ = thin_path_samples(path, fixed_mask, min_step)
    return thinned


def _build_structured_connector(
    start: np.ndarray,
    start_heading: float,
    end: np.ndarray,
    end_heading: float,
    config: PlannerConfig,
) -> np.ndarray:
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    if np.linalg.norm(end - start) <= 1e-9:
        return start[None, :]

    if _connector_is_straight(start, start_heading, end, end_heading):
        return build_line_segment(start, end, config.guide_point_spacing)

    connector = build_curvature_limited_segment(
        start,
        start_heading,
        end,
        end_heading,
        config,
    )
    connector[0] = start
    connector[-1] = end
    connector = _trim_short_terminal_steps(connector, 0.4 * config.guide_point_spacing)
    connector = _thin_connector_samples(connector, 0.4 * config.guide_point_spacing)
    connector[0] = start
    connector[-1] = end
    return connector


def _build_layer_path(layer_segments: list[np.ndarray], config: PlannerConfig) -> np.ndarray:
    nonempty_segments = [segment for segment in layer_segments if len(segment)]
    if not nonempty_segments:
        return np.empty((0, 2), dtype=float)

    path_chunks: list[np.ndarray] = []
    first_segment = nonempty_segments[0]
    path_chunks.append(first_segment[:1])

    previous_end = first_segment[0]
    for point in first_segment[1:]:
        path_chunks.append(build_line_segment(previous_end, point, config.guide_point_spacing))
        previous_end = point

    for segment_idx in range(1, len(nonempty_segments)):
        previous_segment = nonempty_segments[segment_idx - 1]
        segment = nonempty_segments[segment_idx]
        if np.linalg.norm(segment[0] - previous_end) > 1e-9:
            exit_heading = _segment_end_heading(previous_segment, segment[0])
            entry_heading = _segment_start_heading(segment, previous_end)
            path_chunks.append(
                _build_structured_connector(
                    previous_end,
                    exit_heading,
                    segment[0],
                    entry_heading,
                    config,
                )
            )

        previous_end = segment[0]
        for point in segment[1:]:
            path_chunks.append(build_line_segment(previous_end, point, config.guide_point_spacing))
            previous_end = point

    return concatenate_paths(path_chunks)


def _layer_start_info(layer_segments: list[np.ndarray]) -> tuple[np.ndarray, float] | None:
    nonempty_segments = [segment for segment in layer_segments if len(segment)]
    if not nonempty_segments:
        return None
    first_segment = nonempty_segments[0]
    next_point = nonempty_segments[1][0] if len(nonempty_segments) > 1 else None
    return first_segment[0], _segment_start_heading(first_segment, next_point)


def _layer_end_info(layer_segments: list[np.ndarray]) -> tuple[np.ndarray, float] | None:
    nonempty_segments = [segment for segment in layer_segments if len(segment)]
    if not nonempty_segments:
        return None
    last_segment = nonempty_segments[-1]
    previous_point = nonempty_segments[-2][-1] if len(nonempty_segments) > 1 else None
    return last_segment[-1], _segment_end_heading(last_segment, previous_point)


def _layer_has_valid_progression(layer_segments: list[np.ndarray]) -> bool:
    edge_indices = (0, 2, 4, 6)
    corner_indices = (1, 3, 5, 7)

    if any(len(layer_segments[idx]) == 0 for idx in edge_indices):
        return False

    return any(len(layer_segments[idx]) > 0 for idx in corner_indices)


def _layer_has_required_edges(layer_segments: list[np.ndarray]) -> bool:
    edge_indices = (0, 2, 4, 6)
    return not any(len(layer_segments[idx]) == 0 for idx in edge_indices)


def _extract_boundary_layers(
    stop_points: np.ndarray,
    max_layers: int,
) -> tuple[np.ndarray, list[list[np.ndarray]], list[list[np.ndarray]]]:
    remaining_points = np.asarray(stop_points, dtype=float)
    layer_levels: list[float] = []
    layer_segment_points: list[list[np.ndarray]] = []
    layer_segment_indices: list[list[np.ndarray]] = []

    for layer_idx in range(max_layers):
        if len(remaining_points) == 0:
            break

        raw_segments = _ordered_boundary_segments(remaining_points)
        segments = [_dedupe_consecutive_points(segment) for segment in raw_segments]
        for corner_idx in (1, 3, 5, 7):
            segments[corner_idx] = _filter_corner_progression(segments[corner_idx])
        assigned_points = (
            np.vstack([segment for segment in segments if len(segment)])
            if any(len(segment) for segment in segments)
            else np.empty((0, 2), dtype=float)
        )
        assigned_points = _dedupe_consecutive_points(assigned_points)

        allow_rectangular_terminal_layer = True
        layer_is_valid = (
            _layer_has_valid_progression(segments)
            or (allow_rectangular_terminal_layer and _layer_has_required_edges(segments))
        )

        if len(assigned_points) < 8 or not layer_is_valid:
            break

        layer_levels.append(float(np.max(np.linalg.norm(assigned_points, axis=1))))
        layer_segment_points.append(segments)
        layer_segment_indices.append(
            [_segment_indices_from_points(stop_points, segment) for segment in segments]
        )
        remaining_points = _remove_points(remaining_points, assigned_points)

        if len(remaining_points):
            remaining_x_count = len(np.unique(np.round(remaining_points[:, 0] / GRID_SIZE).astype(int)))
            remaining_y_count = len(np.unique(np.round(remaining_points[:, 1] / GRID_SIZE).astype(int)))
            if remaining_x_count <= FINAL_STAGE_GRID_SIDE and remaining_y_count <= FINAL_STAGE_GRID_SIDE:
                break

    return (
        np.asarray(layer_levels, dtype=float),
        layer_segment_indices,
        layer_segment_points,
    )


def _flatten_unique_indices(index_groups: list[list[np.ndarray]]) -> np.ndarray:
    flattened: list[int] = []
    for group in index_groups:
        for indices in group:
            if len(indices):
                flattened.extend(np.asarray(indices, dtype=int).tolist())

    if not flattened:
        return np.empty(0, dtype=int)
    return np.asarray(sorted(set(flattened)), dtype=int)


def _flatten_ordered_indices(index_groups: list[list[np.ndarray]]) -> np.ndarray:
    ordered: list[int] = []
    seen: set[int] = set()
    for group in index_groups:
        for indices in group:
            for idx in np.asarray(indices, dtype=int).tolist():
                if idx in seen:
                    continue
                seen.add(idx)
                ordered.append(int(idx))
    return np.asarray(ordered, dtype=int) if ordered else np.empty(0, dtype=int)


def _point_grid_key(point: np.ndarray, grid_size: float) -> tuple[int, int]:
    scaled = np.rint(np.asarray(point, dtype=float) / grid_size).astype(int)
    return int(scaled[0]), int(scaled[1])


def _extract_cleanup_regions(
    stop_points: np.ndarray,
    remaining_indices: np.ndarray,
    grid_size: float,
) -> list[np.ndarray]:
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


def _ordered_waypoint_headings(points: np.ndarray) -> np.ndarray:
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

        if np.linalg.norm(delta) <= 1e-9:
            headings[idx] = headings[idx - 1] if idx > 0 else 0.0
        else:
            headings[idx] = float(np.arctan2(delta[1], delta[0]))
    return headings


def _path_samples_feasible(points: np.ndarray, config: PlannerConfig) -> bool:
    return bool(np.all([is_point_feasible(point, config) for point in np.asarray(points, dtype=float)]))


def _path_respects_min_turn_radius(points: np.ndarray, min_turn_radius: float) -> bool:
    radii = estimate_local_turn_radii(np.asarray(points, dtype=float))
    finite = radii[np.isfinite(radii)]
    return bool(len(finite) == 0 or np.min(finite) >= min_turn_radius - 1e-6)


def _fixed_point_mask(path: np.ndarray, fixed_points: np.ndarray, atol: float = 1e-9) -> np.ndarray:
    path = np.asarray(path, dtype=float)
    fixed_points = np.asarray(fixed_points, dtype=float)
    if len(path) == 0:
        return np.empty(0, dtype=bool)
    if len(fixed_points) == 0:
        return np.zeros(len(path), dtype=bool)
    return np.array(
        [np.any(np.all(np.isclose(fixed_points, point, atol=atol), axis=1)) for point in path],
        dtype=bool,
    )


def _build_exact_waypoint_path(points: np.ndarray, config: PlannerConfig) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        return np.empty((0, 2), dtype=float)
    if len(points) == 1:
        return points.copy()

    headings = _ordered_waypoint_headings(points)
    path_chunks: list[np.ndarray] = []
    mask_chunks: list[np.ndarray] = []

    append_path_chunk(path_chunks, mask_chunks, points[:1], np.array([True], dtype=bool))
    for idx in range(len(points) - 1):
        segment = build_curvature_limited_segment(
            points[idx],
            headings[idx],
            points[idx + 1],
            headings[idx + 1],
            config,
        )
        segment_mask = np.zeros(len(segment), dtype=bool)
        segment_mask[0] = True
        segment_mask[-1] = True
        append_path_chunk(path_chunks, mask_chunks, segment, segment_mask)

    raw_path = stack_path_chunks(path_chunks)
    raw_mask = concatenate_masks(mask_chunks)
    thinned_path, thinned_mask = thin_path_samples(raw_path, raw_mask, config.route_sample_step * 0.7)
    smoothed_path = smooth_polyline_path(thinned_path, thinned_mask, config)

    if _path_samples_feasible(smoothed_path, config) and _path_respects_min_turn_radius(smoothed_path, config.min_turn_radius):
        return smoothed_path
    if _path_samples_feasible(thinned_path, config) and _path_respects_min_turn_radius(thinned_path, config.min_turn_radius):
        return thinned_path
    return raw_path


def _build_region_variant(
    visit_order: np.ndarray,
    stop_points: np.ndarray,
    config: PlannerConfig,
    strip_transitions: int,
) -> CleanupRegionVariant | None:
    visit_order = np.asarray(visit_order, dtype=int)
    if len(visit_order) == 0:
        return None

    waypoints = stop_points[visit_order]
    path = _build_exact_waypoint_path(waypoints, config)
    if len(path) == 0 or not _path_samples_feasible(path, config):
        return None

    headings = compute_path_headings(path)
    start_heading = float(headings[0]) if len(headings) else 0.0
    end_heading = float(headings[-1]) if len(headings) else start_heading
    return CleanupRegionVariant(
        visit_order=visit_order,
        path=path,
        start_heading=start_heading,
        end_heading=end_heading,
        estimated_length=path_length(path),
        strip_transitions=int(strip_transitions),
        turn_radius_ok=bool(_path_respects_min_turn_radius(path, config.min_turn_radius)),
    )


def _build_region_strip_infos(
    region_indices: np.ndarray,
    stop_points: np.ndarray,
    axis: str,
    grid_size: float,
) -> list[StripInfo]:
    if len(region_indices) == 0:
        return []

    axis_idx = 0 if axis == "vertical" else 1
    along_idx = 1 - axis_idx
    grouped: dict[int, list[int]] = {}

    for global_idx in np.asarray(region_indices, dtype=int):
        strip_id = int(np.rint(stop_points[global_idx, axis_idx] / grid_size))
        grouped.setdefault(strip_id, []).append(int(global_idx))

    strip_infos: list[StripInfo] = []
    for strip_id in sorted(grouped):
        strip_indices = np.asarray(grouped[strip_id], dtype=int)
        strip_points = stop_points[strip_indices]
        order = np.argsort(strip_points[:, along_idx])
        indices_asc = strip_indices[order]
        along_values = stop_points[indices_asc, along_idx]
        if len(along_values) <= 1:
            run_count = 1
        else:
            run_count = 1 + int(np.count_nonzero(np.diff(along_values) > grid_size + 1e-6))

        strip_infos.append(
            StripInfo(
                strip_id=int(strip_id),
                coordinate=float(np.mean(stop_points[indices_asc, axis_idx])),
                indices_asc=indices_asc,
                along_values=along_values,
                min_along=float(along_values[0]),
                max_along=float(along_values[-1]),
                run_count=int(run_count),
            )
        )

    return strip_infos


def extract_active_cleanup_columns(strip_infos: list[StripInfo]) -> np.ndarray:
    if not strip_infos:
        return np.empty(0, dtype=int)
    return np.asarray(sorted(info.strip_id for info in strip_infos), dtype=int)


def build_feasible_column_graph(
    active_columns: np.ndarray,
    min_column_jump: int = MIN_CLEANUP_COLUMN_JUMP,
    max_column_jump: int | None = None,
) -> dict[int, tuple[int, ...]]:
    columns = np.asarray(active_columns, dtype=int)
    graph: dict[int, tuple[int, ...]] = {}
    for column in columns:
        neighbors: list[int] = []
        for other in columns:
            if int(other) == int(column):
                continue
            jump = abs(int(other) - int(column))
            if jump < int(min_column_jump):
                continue
            if max_column_jump is not None and jump > int(max_column_jump):
                continue
            neighbors.append(int(other))
        graph[int(column)] = tuple(sorted(neighbors, key=lambda other: (abs(other - int(column)), other)))
    return graph


def remaining_columns_still_reachable(
    graph: dict[int, tuple[int, ...]],
    remaining_columns: set[int],
) -> bool:
    if len(remaining_columns) <= 1:
        return True

    start = next(iter(remaining_columns))
    stack = [start]
    visited: set[int] = set()

    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        for neighbor in graph.get(current, ()):
            if neighbor in remaining_columns and neighbor not in visited:
                stack.append(neighbor)

    return len(visited) == len(remaining_columns)


def search_column_order(
    current: int,
    graph: dict[int, tuple[int, ...]],
    remaining_columns: set[int],
    path: list[int],
) -> list[int] | None:
    if not remaining_columns:
        return path.copy()

    candidates = [neighbor for neighbor in graph.get(current, ()) if neighbor in remaining_columns]
    candidates.sort(
        key=lambda neighbor: (
            abs(neighbor - current),
            sum(
                1
                for next_neighbor in graph.get(neighbor, ())
                if next_neighbor in remaining_columns and next_neighbor != neighbor
            ),
            abs(neighbor),
            neighbor,
        )
    )

    for neighbor in candidates:
        next_remaining = set(remaining_columns)
        next_remaining.remove(neighbor)
        if next_remaining and not remaining_columns_still_reachable(graph, next_remaining):
            continue
        path.append(neighbor)
        result = search_column_order(neighbor, graph, next_remaining, path)
        if result is not None:
            return result
        path.pop()

    return None


def find_column_order(
    active_columns: np.ndarray,
    min_column_jump: int = MIN_CLEANUP_COLUMN_JUMP,
    max_column_jump: int | None = None,
) -> ColumnOrderResult:
    columns = np.asarray(sorted(np.asarray(active_columns, dtype=int).tolist()), dtype=int)
    graph = build_feasible_column_graph(columns, min_column_jump, max_column_jump)
    if len(columns) == 0:
        return ColumnOrderResult(columns, graph, None, np.empty(0, dtype=int), min_column_jump, max_column_jump)

    start_candidates = sorted(
        columns.tolist(),
        key=lambda column: (len(graph.get(int(column), ())), abs(int(column)), int(column)),
    )

    for start in start_candidates:
        remaining = set(int(column) for column in columns.tolist())
        remaining.remove(int(start))
        if remaining and not remaining_columns_still_reachable(graph, remaining):
            continue
        result = search_column_order(int(start), graph, remaining, [int(start)])
        if result is not None:
            return ColumnOrderResult(
                active_columns=columns,
                adjacency=graph,
                start_column=int(start),
                order=np.asarray(result, dtype=int),
                min_column_jump=int(min_column_jump),
                max_column_jump=max_column_jump,
            )

    return ColumnOrderResult(
        active_columns=columns,
        adjacency=graph,
        start_column=int(start_candidates[0]) if start_candidates else None,
        order=None,
        min_column_jump=int(min_column_jump),
        max_column_jump=max_column_jump,
    )


def _strip_direction_heading(axis: str, ascending: bool) -> float:
    if axis == "vertical":
        return float(np.pi / 2.0 if ascending else -np.pi / 2.0)
    return float(0.0 if ascending else np.pi)


def _estimate_connector_cost(
    start_point: np.ndarray,
    start_heading: float,
    end_point: np.ndarray,
    end_heading: float,
    config: PlannerConfig,
) -> tuple[float, np.ndarray | None, bool]:
    candidates: list[tuple[int, float, np.ndarray, bool]] = []

    for connector in (
        build_curvature_limited_segment(start_point, start_heading, end_point, end_heading, config),
        route_between_points(start_point, end_point, config),
    ):
        if len(connector) == 0 or not _path_samples_feasible(connector, config):
            continue
        turn_ok = _path_respects_min_turn_radius(connector, config.min_turn_radius)
        candidates.append((0 if turn_ok else 1, path_length(connector), connector, turn_ok))

    if not candidates:
        return float("inf"), None, False

    _, cost, connector, turn_ok = min(candidates, key=lambda item: (item[0], item[1]))
    return cost, connector, turn_ok


def _build_sweep_candidate(
    region_indices: np.ndarray,
    stop_points: np.ndarray,
    axis: str,
    config: PlannerConfig,
) -> CleanupRegionCandidate | None:
    strip_infos = _build_region_strip_infos(region_indices, stop_points, axis, config.grid_size)
    if len(strip_infos) < 2:
        return None

    if any(info.run_count != 1 for info in strip_infos):
        return None

    active_columns = extract_active_cleanup_columns(strip_infos)
    column_order_result = find_column_order(active_columns, MIN_CLEANUP_COLUMN_JUMP)
    if not column_order_result.success or column_order_result.order is None:
        return None

    strip_lookup = {info.strip_id: info for info in strip_infos}
    visit_orders: list[np.ndarray] = []
    for start_ascending in (True, False):
        ordered_chunks: list[np.ndarray] = []
        current_ascending = start_ascending
        feasible = True
        for column_id in column_order_result.order.tolist():
            info = strip_lookup.get(int(column_id))
            if info is None:
                feasible = False
                break
            ordered_chunks.append(info.indices_asc if current_ascending else info.indices_asc[::-1])
            current_ascending = not current_ascending
        if feasible and ordered_chunks:
            visit_orders.append(np.concatenate(ordered_chunks))

    variants: list[tuple[np.ndarray, CleanupRegionVariant, CleanupRegionVariant]] = []
    strip_transitions = max(0, len(column_order_result.order) - 1)
    for visit_order in visit_orders:
        forward = _build_region_variant(visit_order, stop_points, config, strip_transitions)
        reverse = _build_region_variant(visit_order[::-1], stop_points, config, strip_transitions)
        if forward is not None and reverse is not None:
            variants.append((visit_order, forward, reverse))

    if not variants:
        return None

    _, best_forward, best_reverse = min(
        variants,
        key=lambda item: (
            0 if (item[1].turn_radius_ok or item[2].turn_radius_ok) else 1,
            min(item[1].estimated_length, item[2].estimated_length),
        ),
    )

    return CleanupRegionCandidate(
        indices=np.asarray(region_indices, dtype=int),
        mode="sweep",
        orientation=axis,
        forward=best_forward,
        reverse=best_reverse,
    )


def _build_transition_candidate(
    region_indices: np.ndarray,
    stop_points: np.ndarray,
    axis: str,
    config: PlannerConfig,
) -> CleanupRegionCandidate | None:
    strip_infos = _build_region_strip_infos(region_indices, stop_points, axis, config.grid_size)
    if len(strip_infos) == 0:
        return None

    ordered_chunks: list[np.ndarray] = []
    current_endpoint: np.ndarray | None = None
    current_heading: float | None = None

    for strip_idx, info in enumerate(strip_infos):
        asc_indices = np.asarray(info.indices_asc, dtype=int)
        desc_indices = asc_indices[::-1]

        if strip_idx == 0:
            chosen = asc_indices
            ascending = True
        else:
            assert current_endpoint is not None
            assert current_heading is not None

            asc_start = stop_points[asc_indices[0]]
            desc_start = stop_points[desc_indices[0]]
            asc_heading = _strip_direction_heading(axis, True)
            desc_heading = _strip_direction_heading(axis, False)

            asc_cost, _, asc_turn_ok = _estimate_connector_cost(current_endpoint, current_heading, asc_start, asc_heading, config)
            desc_cost, _, desc_turn_ok = _estimate_connector_cost(current_endpoint, current_heading, desc_start, desc_heading, config)

            if (0 if desc_turn_ok else 1, desc_cost) < (0 if asc_turn_ok else 1, asc_cost):
                chosen = desc_indices
                ascending = False
            else:
                chosen = asc_indices
                ascending = True

        ordered_chunks.append(chosen)
        current_endpoint = stop_points[chosen[-1]]
        current_heading = _strip_direction_heading(axis, ascending)

    visit_order = np.concatenate(ordered_chunks) if ordered_chunks else np.empty(0, dtype=int)
    strip_transitions = max(0, len(strip_infos) - 1)
    forward = _build_region_variant(visit_order, stop_points, config, strip_transitions)
    reverse = _build_region_variant(visit_order[::-1], stop_points, config, strip_transitions)
    if forward is None or reverse is None:
        return None

    return CleanupRegionCandidate(
        indices=np.asarray(region_indices, dtype=int),
        mode="transition",
        orientation=axis,
        forward=forward,
        reverse=reverse,
    )


def _candidate_length(candidate: CleanupRegionCandidate) -> float:
    return min(candidate.forward.estimated_length, candidate.reverse.estimated_length)


def _candidate_has_turn_clean_variant(candidate: CleanupRegionCandidate) -> bool:
    return bool(candidate.forward.turn_radius_ok or candidate.reverse.turn_radius_ok)


def _choose_cleanup_variant(
    candidate: CleanupRegionCandidate,
    current_point: np.ndarray,
    current_heading: float,
    stop_points: np.ndarray,
    config: PlannerConfig,
) -> tuple[CleanupRegionVariant, np.ndarray]:
    options: list[tuple[int, float, CleanupRegionVariant, np.ndarray]] = []
    for variant in (candidate.forward, candidate.reverse):
        if len(variant.visit_order) == 0:
            continue
        region_start = stop_points[variant.visit_order[0]]
        entry_cost, connector, turn_ok = _estimate_connector_cost(
            current_point,
            current_heading,
            region_start,
            variant.start_heading,
            config,
        )
        if connector is None:
            continue
        options.append((0 if turn_ok else 1, entry_cost, variant, connector))

    if not options:
        fallback_variant = candidate.forward
        fallback_connector = build_line_segment(
            current_point,
            stop_points[fallback_variant.visit_order[0]],
            config.guide_point_spacing,
        )
        return fallback_variant, fallback_connector

    _, _, variant, connector = min(options, key=lambda item: (item[0], item[1], item[2].estimated_length))
    return variant, connector


def _build_cleanup_column_debug(
    region_indices: np.ndarray,
    stop_points: np.ndarray,
    candidate: CleanupRegionCandidate,
    grid_size: float,
    min_column_jump: int = MIN_CLEANUP_COLUMN_JUMP,
) -> dict[str, object]:
    strip_infos = _build_region_strip_infos(region_indices, stop_points, candidate.orientation, grid_size)
    active_columns = extract_active_cleanup_columns(strip_infos)
    column_order = find_column_order(active_columns, min_column_jump)
    jumps = (
        np.abs(np.diff(column_order.order)).astype(int).tolist()
        if column_order.success and column_order.order is not None and len(column_order.order) >= 2
        else []
    )
    return {
        "orientation": candidate.orientation,
        "mode": candidate.mode,
        "active_columns": active_columns.tolist(),
        "feasible_jump_counts": {
            int(column): len(column_order.adjacency.get(int(column), ()))
            for column in active_columns.tolist()
        },
        "chosen_start_column": column_order.start_column,
        "column_order": column_order.order.tolist() if column_order.order is not None else None,
        "jump_sizes": jumps,
        "order_found": bool(column_order.success),
        "order_valid": bool(
            column_order.success
            and column_order.order is not None
            and len(np.unique(column_order.order)) == len(active_columns)
            and (len(jumps) == 0 or all(jump >= min_column_jump for jump in jumps))
        ),
    }


def _select_cleanup_candidate(
    region_indices: np.ndarray,
    stop_points: np.ndarray,
    config: PlannerConfig,
) -> CleanupRegionCandidate:
    sweep_candidates = [
        candidate
        for candidate in (
            _build_sweep_candidate(region_indices, stop_points, "vertical", config),
            _build_sweep_candidate(region_indices, stop_points, "horizontal", config),
        )
        if candidate is not None
    ]
    if sweep_candidates:
        preferred = [candidate for candidate in sweep_candidates if _candidate_has_turn_clean_variant(candidate)]
        pool = preferred if preferred else sweep_candidates
        return min(pool, key=lambda candidate: (candidate.forward.strip_transitions, _candidate_length(candidate)))

    transition_candidates = [
        candidate
        for candidate in (
            _build_transition_candidate(region_indices, stop_points, "vertical", config),
            _build_transition_candidate(region_indices, stop_points, "horizontal", config),
        )
        if candidate is not None
    ]
    if transition_candidates:
        preferred = [candidate for candidate in transition_candidates if _candidate_has_turn_clean_variant(candidate)]
        pool = preferred if preferred else transition_candidates
        return min(
            pool,
            key=lambda candidate: (candidate.forward.strip_transitions, _candidate_length(candidate)),
        )

    fallback = _build_region_variant(np.asarray(region_indices, dtype=int), stop_points, config, 0)
    if fallback is None:
        fallback = CleanupRegionVariant(
            visit_order=np.asarray(region_indices, dtype=int),
            path=stop_points[np.asarray(region_indices, dtype=int)],
            start_heading=0.0,
            end_heading=0.0,
            estimated_length=0.0,
            strip_transitions=0,
            turn_radius_ok=False,
        )
    return CleanupRegionCandidate(
        indices=np.asarray(region_indices, dtype=int),
        mode="transition",
        orientation="vertical",
        forward=fallback,
        reverse=fallback,
    )


def _explicit_stop_hits(path: np.ndarray, stop_points: np.ndarray) -> np.ndarray:
    if len(stop_points) == 0:
        return np.empty(0, dtype=bool)
    if len(path) == 0:
        return np.zeros(len(stop_points), dtype=bool)

    return np.asarray(
        [np.any(np.all(np.isclose(path, point, atol=1e-9), axis=1)) for point in stop_points],
        dtype=bool,
    )


def build_eight_segment_boundary_plan(config: PlannerConfig | None = None) -> BoundaryEightSegmentPlan:
    config = _resolve_config(config)
    scan_result = build_scan_coverage_result(config)
    stop_points = scan_result.stop_points

    if len(stop_points) == 0:
        empty = np.empty((0, 2), dtype=float)
        empty_idx = np.empty(0, dtype=int)
        return BoundaryEightSegmentPlan(
            config=config,
            scan_result=scan_result,
            layer_levels=np.empty(0, dtype=float),
            layer_segment_indices=[],
            layer_segment_points=[],
            main_visited_indices=empty_idx,
            remaining_indices_before_cleanup=empty_idx,
            cleanup_region_indices=[],
            cleanup_region_modes=[],
            cleanup_transition_paths=[],
            cleanup_region_paths=[],
            cleanup_column_debug=[],
            cleanup_visit_order=empty_idx,
            all_visited_indices=empty_idx,
            entry_path=empty,
            inter_layer_paths=[],
            layer_paths=[],
            main_path=empty,
            boundary_path=empty,
            cleanup_path=empty,
            full_path=empty,
        )

    max_layers = 16
    layer_levels, layer_segment_indices, layer_segment_points = _extract_boundary_layers(
        stop_points,
        max_layers=max_layers,
    )

    outer_top_segment = layer_segment_points[0][0] if layer_segment_points else np.empty((0, 2), dtype=float)
    structured_visit_order = _flatten_ordered_indices(layer_segment_indices)
    feasible_start = push_out_of_obstacles(np.zeros(2, dtype=float), config)
    entry_path = (
        _build_exact_waypoint_path(np.vstack([feasible_start, stop_points[structured_visit_order[0]]]), config)
        if len(structured_visit_order)
        else feasible_start[None, :]
    )

    layer_paths = [_build_layer_path(layer_segments, config) for layer_segments in layer_segment_points]
    inter_layer_paths: list[np.ndarray] = []
    for layer_idx in range(len(layer_paths) - 1):
        current_path = layer_paths[layer_idx]
        next_path = layer_paths[layer_idx + 1]
        if len(current_path) == 0 or len(next_path) == 0:
            inter_layer_paths.append(np.empty((0, 2), dtype=float))
            continue
        current_end_info = _layer_end_info(layer_segment_points[layer_idx])
        next_start_info = _layer_start_info(layer_segment_points[layer_idx + 1])
        if current_end_info is None or next_start_info is None:
            inter_layer_paths.append(np.empty((0, 2), dtype=float))
            continue
        current_end, current_heading = current_end_info
        next_start, next_heading = next_start_info
        inter_layer_paths.append(
            _build_structured_connector(
                current_end,
                current_heading,
                next_start,
                next_heading,
                config,
            )
        )

    stage1_chunks: list[np.ndarray] = [entry_path]
    for layer_idx, layer_path in enumerate(layer_paths):
        stage1_chunks.append(layer_path)
        if layer_idx < len(inter_layer_paths):
            stage1_chunks.append(inter_layer_paths[layer_idx])

    main_visited_indices = _flatten_unique_indices(layer_segment_indices)
    boundary_path = concatenate_paths(stage1_chunks[1:])
    raw_main_path = concatenate_paths(stage1_chunks)
    structured_fixed_points = np.vstack([feasible_start[None, :], stop_points[main_visited_indices]])
    raw_fixed_mask = _fixed_point_mask(raw_main_path, structured_fixed_points)
    main_path, _ = thin_path_samples(raw_main_path, raw_fixed_mask, 0.5 * config.guide_point_spacing)
    all_indices = np.arange(len(stop_points), dtype=int)
    remaining_indices_before_cleanup = np.setdiff1d(all_indices, main_visited_indices, assume_unique=False)
    cleanup_region_indices = _extract_cleanup_regions(stop_points, remaining_indices_before_cleanup, config.grid_size)
    cleanup_region_modes: list[str] = []
    cleanup_transition_paths: list[np.ndarray] = []
    cleanup_region_paths: list[np.ndarray] = []
    cleanup_column_debug: list[dict[str, object]] = []
    cleanup_visit_order_chunks: list[np.ndarray] = []
    cleanup_chunks: list[np.ndarray] = []
    current_point = main_path[-1] if len(main_path) else feasible_start
    current_heading = float(compute_path_headings(main_path)[-1]) if len(main_path) >= 2 else 0.0

    unplanned_regions = list(cleanup_region_indices)
    while unplanned_regions:
        region_options: list[tuple[int, float, np.ndarray, CleanupRegionCandidate, CleanupRegionVariant, np.ndarray]] = []
        for region_indices in unplanned_regions:
            candidate = _select_cleanup_candidate(region_indices, stop_points, config)
            variant, connector = _choose_cleanup_variant(candidate, current_point, current_heading, stop_points, config)
            connector_turn_ok = _path_respects_min_turn_radius(connector, config.min_turn_radius) if len(connector) else True
            region_options.append(
                (
                    0 if connector_turn_ok else 1,
                    path_length(connector),
                    np.asarray(region_indices, dtype=int),
                    candidate,
                    variant,
                    connector,
                )
            )

        _, _, chosen_region_indices, chosen_candidate, chosen_variant, chosen_connector = min(
            region_options,
            key=lambda item: (item[0], item[1], len(item[2])),
        )
        cleanup_region_modes.append(chosen_candidate.mode)
        cleanup_transition_paths.append(chosen_connector)
        cleanup_region_paths.append(chosen_variant.path)
        cleanup_column_debug.append(
            _build_cleanup_column_debug(
                chosen_region_indices,
                stop_points,
                chosen_candidate,
                config.grid_size,
            )
        )
        cleanup_visit_order_chunks.append(chosen_variant.visit_order)
        cleanup_chunks.extend([chosen_connector, chosen_variant.path])

        if len(chosen_variant.path):
            current_point = chosen_variant.path[-1]
            headings = compute_path_headings(chosen_variant.path)
            current_heading = float(headings[-1]) if len(headings) else current_heading

        matched_region_idx = next(
            idx for idx, region in enumerate(unplanned_regions)
            if np.array_equal(region, chosen_region_indices)
        )
        unplanned_regions.pop(matched_region_idx)

    cleanup_visit_order = (
        np.concatenate(cleanup_visit_order_chunks)
        if cleanup_visit_order_chunks
        else np.empty(0, dtype=int)
    )
    cleanup_path = concatenate_paths(cleanup_chunks)
    full_path = concatenate_paths([main_path, cleanup_path])
    all_visited_indices = (
        np.unique(np.concatenate([main_visited_indices, cleanup_visit_order]))
        if len(cleanup_visit_order)
        else main_visited_indices.copy()
    )

    return BoundaryEightSegmentPlan(
        config=config,
        scan_result=scan_result,
        layer_levels=layer_levels,
        layer_segment_indices=layer_segment_indices,
        layer_segment_points=layer_segment_points,
        main_visited_indices=main_visited_indices,
        remaining_indices_before_cleanup=remaining_indices_before_cleanup,
        cleanup_region_indices=cleanup_region_indices,
        cleanup_region_modes=cleanup_region_modes,
        cleanup_transition_paths=cleanup_transition_paths,
        cleanup_region_paths=cleanup_region_paths,
        cleanup_column_debug=cleanup_column_debug,
        cleanup_visit_order=cleanup_visit_order,
        all_visited_indices=all_visited_indices,
        entry_path=entry_path,
        inter_layer_paths=inter_layer_paths,
        layer_paths=layer_paths,
        main_path=main_path,
        boundary_path=boundary_path,
        cleanup_path=cleanup_path,
        full_path=full_path,
    )


def validate_eight_segment_boundary_plan(plan: BoundaryEightSegmentPlan) -> dict[str, object]:
    stop_points = plan.scan_result.stop_points
    full_hits = _explicit_stop_hits(plan.full_path, stop_points)
    main_hits = _explicit_stop_hits(plan.main_path, stop_points)
    cleanup_serviced_unique = np.unique(plan.cleanup_visit_order) if len(plan.cleanup_visit_order) else np.empty(0, dtype=int)
    sweep_region_count = int(sum(mode == "sweep" for mode in plan.cleanup_region_modes))
    transition_region_count = int(sum(mode == "transition" for mode in plan.cleanup_region_modes))
    main_path_feasible = bool(_path_samples_feasible(plan.main_path, plan.config)) if len(plan.main_path) else True
    main_turn_ok = bool(_path_respects_min_turn_radius(plan.main_path, plan.config.min_turn_radius)) if len(plan.main_path) else True
    cleanup_path_feasible = bool(_path_samples_feasible(plan.cleanup_path, plan.config)) if len(plan.cleanup_path) else True
    cleanup_turn_ok = bool(_path_respects_min_turn_radius(plan.cleanup_path, plan.config.min_turn_radius)) if len(plan.cleanup_path) else True
    full_path_feasible = bool(_path_samples_feasible(plan.full_path, plan.config)) if len(plan.full_path) else True
    full_turn_ok = bool(_path_respects_min_turn_radius(plan.full_path, plan.config.min_turn_radius)) if len(plan.full_path) else True
    first_cleanup_debug = plan.cleanup_column_debug[0] if plan.cleanup_column_debug else {}
    cleanup_column_orders_valid = bool(
        all(
            debug.get("order_valid", False)
            for debug, mode in zip(plan.cleanup_column_debug, plan.cleanup_region_modes)
            if mode == "sweep"
        )
    )

    return {
        "total_required_stop_points": int(len(stop_points)),
        "points_visited_by_main_structured_coverage": int(len(plan.main_visited_indices)),
        "remaining_unvisited_points_before_cleanup": int(len(plan.remaining_indices_before_cleanup)),
        "cleanup_region_count": int(len(plan.cleanup_region_indices)),
        "sweep_region_count": sweep_region_count,
        "one_pass_transition_region_count": transition_region_count,
        "cleanup_serviced_point_count": int(len(cleanup_serviced_unique)),
        "all_required_stop_points_explicitly_visited": bool(np.all(full_hits)) if len(full_hits) else True,
        "missing_required_point_count": int(np.count_nonzero(~full_hits)) if len(full_hits) else 0,
        "main_structured_points_explicitly_hit": bool(np.all(main_hits[plan.main_visited_indices])) if len(plan.main_visited_indices) else True,
        "segment_count": int(
            sum(1 for layer_segments in plan.layer_segment_points for segment in layer_segments if len(segment))
        ),
        "layer_count": int(len(plan.layer_segment_points)),
        "main_path_feasible": main_path_feasible,
        "main_path_min_turn_radius_respected": main_turn_ok,
        "cleanup_path_feasible": cleanup_path_feasible,
        "cleanup_path_min_turn_radius_respected": cleanup_turn_ok,
        "full_path_feasible": full_path_feasible,
        "full_path_min_turn_radius_respected": full_turn_ok,
        "cleanup_column_orders_valid": cleanup_column_orders_valid,
        "cleanup_column_debug": plan.cleanup_column_debug,
        "first_cleanup_active_columns": first_cleanup_debug.get("active_columns", []),
        "first_cleanup_column_order": first_cleanup_debug.get("column_order"),
        "first_cleanup_jump_sizes": first_cleanup_debug.get("jump_sizes", []),
        "first_cleanup_order_found": bool(first_cleanup_debug.get("order_found", False)),
    }


def plot_eight_segment_boundary_plan(plan: BoundaryEightSegmentPlan) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    config = plan.config
    stop_points = plan.scan_result.stop_points
    fig, ax = plt.subplots(figsize=(10, 10))

    ax.add_patch(Circle((0.0, 0.0), config.tank_radius, fill=False, color="black", linewidth=2.0))
    ax.add_patch(
        Circle(
            (0.0, 0.0),
            config.outermost_centerline_radius,
            fill=False,
            color="#9a3412",
            linestyle="--",
            linewidth=1.2,
        )
    )

    for column in config.columns:
        ax.add_patch(Circle(column, config.scan_keepout_radius, color="#ef4444", alpha=0.15))
        ax.add_patch(
            Circle(
                column,
                config.centerline_column_margin,
                fill=False,
                color="#dc2626",
                linestyle=":",
                linewidth=1.0,
            )
        )

    if len(stop_points):
        ax.scatter(
            stop_points[:, 0],
            stop_points[:, 1],
            s=16,
            color="#9ca3af",
            alpha=0.4,
            label="All required visit points",
            zorder=1,
        )

    if len(plan.main_visited_indices):
        main_points = stop_points[plan.main_visited_indices]
        ax.scatter(
            main_points[:, 0],
            main_points[:, 1],
            s=24,
            color="#1d4ed8",
            alpha=0.85,
            label="Main structured points",
            zorder=5,
        )

    if len(plan.remaining_indices_before_cleanup):
        remaining_points = stop_points[plan.remaining_indices_before_cleanup]
        ax.scatter(
            remaining_points[:, 0],
            remaining_points[:, 1],
            s=44,
            facecolors="none",
            edgecolors="#f97316",
            linewidths=1.0,
            label="Remaining points before cleanup",
            zorder=4,
        )

    layer_path_colors = ["#111827", "#2563eb", "#059669", "#dc2626", "#7c3aed"]

    for layer_idx, layer_path in enumerate(plan.layer_paths):
        if len(layer_path) == 0:
            continue
        ax.plot(
            layer_path[:, 0],
            layer_path[:, 1],
            color=layer_path_colors[layer_idx % len(layer_path_colors)],
            linewidth=2.1,
            label=f"Loop {layer_idx + 1}",
            zorder=2,
        )

    for connector_idx, connector in enumerate(plan.inter_layer_paths):
        if len(connector) == 0:
            continue
        ax.plot(
            connector[:, 0],
            connector[:, 1],
            color="#f97316",
            linewidth=1.8,
            linestyle="--",
            label="Inward connector" if connector_idx == 0 else None,
            zorder=3,
        )

    if len(plan.entry_path):
        ax.plot(
            plan.entry_path[:, 0],
            plan.entry_path[:, 1],
            color="#f97316",
            linewidth=2.2,
            label="Center to top entry",
            zorder=3,
        )

    cleanup_palette = ["#059669", "#7c3aed", "#dc2626", "#d97706", "#0f766e", "#be185d"]
    cleanup_colors = [cleanup_palette[idx % len(cleanup_palette)] for idx in range(max(1, len(plan.cleanup_region_indices)))]
    for region_idx, region_indices in enumerate(plan.cleanup_region_indices):
        if len(region_indices) == 0:
            continue
        region_points = stop_points[region_indices]
        color = cleanup_colors[region_idx % len(cleanup_colors)]
        ax.scatter(
            region_points[:, 0],
            region_points[:, 1],
            s=34,
            color=color,
            alpha=0.95,
            label="Cleanup region points" if region_idx == 0 else None,
            zorder=7,
        )

    for region_idx, transition in enumerate(plan.cleanup_transition_paths):
        if len(transition) == 0:
            continue
        ax.plot(
            transition[:, 0],
            transition[:, 1],
            color="#fb923c",
            linewidth=2.0,
            linestyle="--",
            label="Cleanup transition" if region_idx == 0 else None,
            zorder=4,
        )

    for region_idx, region_path in enumerate(plan.cleanup_region_paths):
        if len(region_path) == 0:
            continue
        color = cleanup_colors[region_idx % len(cleanup_colors)]
        mode = plan.cleanup_region_modes[region_idx] if region_idx < len(plan.cleanup_region_modes) else "cleanup"
        ax.plot(
            region_path[:, 0],
            region_path[:, 1],
            color=color,
            linewidth=2.6,
            label=f"{mode.title()} cleanup path" if region_idx == 0 else None,
            zorder=5,
        )

    if len(plan.full_path):
        ax.plot(
            plan.full_path[:, 0],
            plan.full_path[:, 1],
            color="#374151",
            linewidth=0.9,
            alpha=0.12,
            label="Final combined path",
            zorder=1,
        )

    if len(plan.full_path) > 1:
        arrow_indices = np.arange(0, len(plan.full_path) - 1, max(1, len(plan.full_path) // 80))
        deltas = plan.full_path[arrow_indices + 1] - plan.full_path[arrow_indices]
        ax.quiver(
            plan.full_path[arrow_indices, 0],
            plan.full_path[arrow_indices, 1],
            deltas[:, 0],
            deltas[:, 1],
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.003,
            color="#f97316",
            zorder=8,
            label="Travel direction",
        )

    if plan.cleanup_column_debug:
        first_debug = plan.cleanup_column_debug[0]
        debug_text = (
            f"Cleanup cols: {first_debug.get('active_columns', [])}\n"
            f"Order: {first_debug.get('column_order')}\n"
            f"Jumps: {first_debug.get('jump_sizes', [])}"
        )
        ax.text(
            0.02,
            0.02,
            debug_text,
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="bottom",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.78, "edgecolor": "#9ca3af"},
            zorder=10,
        )

    ax.scatter(0.0, 0.0, s=42, color="black", label="Tank center", zorder=9)
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


def animate_coverage_plan(
    plan: CoveragePlan,
    interval_ms: int = 25,
    tail_points: int = 80,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.patches import Circle

    config = plan.config
    stop_points = plan.scan_result.stop_points
    fig, ax = plt.subplots(figsize=(10, 10))

    ax.add_patch(Circle((0.0, 0.0), config.tank_radius, fill=False, color="black", linewidth=2.0))
    ax.add_patch(
        Circle(
            (0.0, 0.0),
            config.outermost_centerline_radius,
            fill=False,
            color="#9a3412",
            linestyle="--",
            linewidth=1.5,
        )
    )

    for column in config.columns:
        ax.add_patch(Circle(column, config.scan_keepout_radius, color="#ef4444", alpha=0.18))
        ax.add_patch(
            Circle(
                column,
                config.centerline_column_margin,
                fill=False,
                color="#dc2626",
                linestyle=":",
                linewidth=1.2,
            )
        )

    if len(plan.main_path):
        ax.plot(
            plan.main_path[:, 0],
            plan.main_path[:, 1],
            color="#93c5fd",
            linewidth=1.0,
            linestyle="--",
            alpha=0.6,
            zorder=1,
        )

    for reference_path in plan.petal_reference_paths:
        ax.plot(
            reference_path[:, 0],
            reference_path[:, 1],
            color="#86efac",
            linewidth=0.9,
            linestyle="--",
            alpha=0.5,
            zorder=1,
        )

    unvisited_main = stop_points[plan.main_indices] if len(plan.main_indices) else np.empty((0, 2), dtype=float)
    unvisited_petal = stop_points[plan.petal_indices] if len(plan.petal_indices) else np.empty((0, 2), dtype=float)

    main_unvisited_artist = ax.scatter(
        unvisited_main[:, 0] if len(unvisited_main) else [],
        unvisited_main[:, 1] if len(unvisited_main) else [],
        s=24,
        color="#93c5fd",
        alpha=0.75,
        label="Unvisited main stops",
        zorder=3,
    )
    petal_unvisited_artist = ax.scatter(
        unvisited_petal[:, 0] if len(unvisited_petal) else [],
        unvisited_petal[:, 1] if len(unvisited_petal) else [],
        s=34,
        marker="D",
        color="#6ee7b7",
        alpha=0.75,
        label="Unvisited petal stops",
        zorder=3,
    )

    visited_artist = ax.scatter(
        [],
        [],
        s=28,
        color="#111827",
        label="Visited mandatory stops",
        zorder=5,
    )
    robot_artist = ax.scatter(
        [],
        [],
        s=85,
        color="#f97316",
        edgecolors="black",
        linewidths=0.6,
        label="Robot",
        zorder=6,
    )
    path_artist, = ax.plot(
        [],
        [],
        color="#111827",
        linewidth=2.6,
        zorder=4,
        label="Driven path",
    )
    tail_artist, = ax.plot(
        [],
        [],
        color="#f97316",
        linewidth=3.0,
        alpha=0.85,
        zorder=5,
        label="Recent motion",
    )
    status_text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#d1d5db"},
    )

    path_points = plan.full_path
    ordered_stops = plan.ordered_stop_points
    stop_progress_indices: list[int] = []
    path_idx = 0
    for stop in ordered_stops:
        while path_idx < len(path_points):
            if np.linalg.norm(path_points[path_idx] - stop) < 1e-9:
                stop_progress_indices.append(path_idx)
                path_idx += 1
                break
            path_idx += 1

    stop_progress_indices_array = np.asarray(stop_progress_indices, dtype=int)
    stop_modes_in_order = plan.point_modes[plan.visit_order] if len(plan.visit_order) else np.empty(0, dtype=object)

    ax.scatter(0.0, 0.0, s=45, color="black", label="Tank center", zorder=6)
    ax.set_aspect("equal", adjustable="box")
    margin = config.grid_size * 1.5
    ax.set_xlim(-(config.tank_radius + margin), config.tank_radius + margin)
    ax.set_ylim(-(config.tank_radius + margin), config.tank_radius + margin)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Animated tank coverage path")
    ax.grid(True, alpha=0.22)
    ax.legend(loc="upper right")
    plt.tight_layout()

    def update(frame: int):
        current_path = path_points[: frame + 1]
        path_artist.set_data(current_path[:, 0], current_path[:, 1])

        tail_start = max(0, frame - tail_points)
        tail = path_points[tail_start : frame + 1]
        tail_artist.set_data(tail[:, 0], tail[:, 1])

        robot_point = path_points[frame]
        robot_artist.set_offsets(robot_point[None, :])

        visited_count = int(np.searchsorted(stop_progress_indices_array, frame, side="right"))
        if visited_count > 0:
            visited_points = ordered_stops[:visited_count]
            visited_artist.set_offsets(visited_points)
        else:
            visited_artist.set_offsets(np.empty((0, 2), dtype=float))

        remaining_main = ordered_stops[visited_count:][stop_modes_in_order[visited_count:] == "main"]
        remaining_petal = ordered_stops[visited_count:][stop_modes_in_order[visited_count:] == "petal"]
        main_unvisited_artist.set_offsets(remaining_main if len(remaining_main) else np.empty((0, 2), dtype=float))
        petal_unvisited_artist.set_offsets(remaining_petal if len(remaining_petal) else np.empty((0, 2), dtype=float))

        mode = "petal" if visited_count > 0 and visited_count <= len(stop_modes_in_order) and stop_modes_in_order[max(0, visited_count - 1)] == "petal" else "main"
        status_text.set_text(
            f"path sample: {frame + 1}/{len(path_points)}\n"
            f"visited stops: {visited_count}/{len(ordered_stops)}\n"
            f"current mode: {mode}"
        )

        return (
            path_artist,
            tail_artist,
            robot_artist,
            visited_artist,
            main_unvisited_artist,
            petal_unvisited_artist,
            status_text,
        )

    animation = FuncAnimation(
        fig,
        update,
        frames=len(path_points),
        interval=interval_ms,
        blit=False,
        repeat=True,
    )
    fig._coverage_animation = animation
    plt.show()


if __name__ == "__main__":
    boundary_plan = build_eight_segment_boundary_plan()
    validation = validate_eight_segment_boundary_plan(boundary_plan)

    print(f"Total required stop points: {validation['total_required_stop_points']}")
    print(f"Points visited by main structured coverage: {validation['points_visited_by_main_structured_coverage']}")
    print(f"Remaining unvisited points before cleanup: {validation['remaining_unvisited_points_before_cleanup']}")
    print(f"Number of cleanup regions found: {validation['cleanup_region_count']}")
    print(f"Number of sweep regions: {validation['sweep_region_count']}")
    print(f"Number of one-pass transition regions: {validation['one_pass_transition_region_count']}")
    print(f"Total remaining points serviced by cleanup: {validation['cleanup_serviced_point_count']}")
    print(f"Whether all required stop points are now explicitly visited: {validation['all_required_stop_points_explicitly_visited']}")
    print(f"Main structured path feasible: {validation['main_path_feasible']}")
    print(f"Main structured path min turn radius respected: {validation['main_path_min_turn_radius_respected']}")
    print(f"Cleanup path feasible: {validation['cleanup_path_feasible']}")
    print(f"Cleanup path min turn radius respected: {validation['cleanup_path_min_turn_radius_respected']}")
    print(f"Full path feasible: {validation['full_path_feasible']}")
    print(f"Full path min turn radius respected: {validation['full_path_min_turn_radius_respected']}")
    print(f"Cleanup column orders valid: {validation['cleanup_column_orders_valid']}")
    print(f"First cleanup active columns: {validation['first_cleanup_active_columns']}")
    print(f"First cleanup column order: {validation['first_cleanup_column_order']}")
    print(f"First cleanup jump sizes: {validation['first_cleanup_jump_sizes']}")
    print(f"First cleanup order found: {validation['first_cleanup_order_found']}")

    plot_eight_segment_boundary_plan(boundary_plan)
