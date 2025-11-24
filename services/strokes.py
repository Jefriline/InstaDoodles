from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

try:
    from numba import jit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

from .image_processing import compute_orientation_field


@dataclass
class Stroke:
    points: List[Tuple[float, float]]
    stroke_type: str
    weight: float


def _smooth_polyline(points: np.ndarray, iterations: int) -> np.ndarray:
    poly = points.astype(np.float32)
    for _ in range(iterations):
        if len(poly) < 3:
            break
        new_points = []
        for i in range(len(poly) - 1):
            p0 = poly[i]
            p1 = poly[i + 1]
            q = 0.75 * p0 + 0.25 * p1
            r = 0.25 * p0 + 0.75 * p1
            new_points.extend([q, r])
        poly = np.array(new_points, dtype=np.float32)
    return poly


@jit(nopython=True, cache=True)
def _bezier_cubic_point(t: float, p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
    mt = 1.0 - t
    mt2 = mt * mt
    mt3 = mt2 * mt
    t2 = t * t
    t3 = t2 * t
    
    result = mt3 * p0 + 3.0 * mt2 * t * p1 + 3.0 * mt * t2 * p2 + t3 * p3
    return result


@jit(nopython=True, cache=True)
def _bezier_quadratic_point(t: float, p0: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    mt = 1.0 - t
    mt2 = mt * mt
    t2 = t * t
    
    result = mt2 * p0 + 2.0 * mt * t * p1 + t2 * p2
    return result


def _smooth_with_bezier(points: np.ndarray, num_samples: int = 50) -> np.ndarray:
    if len(points) < 2:
        return points
    
    if len(points) == 2:
        return points
    
    points = points.astype(np.float32)
    smoothed_points = []
    
    if len(points) == 3:
        p0, p1, p2 = points[0], points[1], points[2]
        t_values = np.linspace(0.0, 1.0, num_samples)
        for t in t_values:
            point = _bezier_quadratic_point(float(t), p0, p1, p2)
            smoothed_points.append(point)
        return np.array(smoothed_points, dtype=np.float32)
    
    num_segments = max(1, (len(points) - 1) // 3)
    samples_per_segment = max(10, num_samples // num_segments)
    
    for i in range(0, len(points) - 1, 3):
        if i + 3 < len(points):
            p0 = points[i]
            p1 = points[i + 1]
            p2 = points[i + 2]
            p3 = points[i + 3]
            
            t_values = np.linspace(0.0, 1.0, samples_per_segment)
            for j, t in enumerate(t_values):
                if i == 0 and j == 0:
                    smoothed_points.append(p0)
                elif j > 0:
                    point = _bezier_cubic_point(float(t), p0, p1, p2, p3)
                    smoothed_points.append(point)
        else:
            remaining = points[i:]
            if len(remaining) >= 3:
                p0, p1, p2 = remaining[0], remaining[1], remaining[-1]
                t_values = np.linspace(0.0, 1.0, samples_per_segment)
                for j, t in enumerate(t_values):
                    if j > 0:
                        point = _bezier_quadratic_point(float(t), p0, p1, p2)
                        smoothed_points.append(point)
            else:
                if len(smoothed_points) == 0 or not np.array_equal(smoothed_points[-1], remaining[0]):
                    smoothed_points.append(remaining[0])
                smoothed_points.extend(remaining[1:])
    
    if not smoothed_points:
        return _smooth_polyline(points, iterations=2)
    
    return np.array(smoothed_points, dtype=np.float32)


def _contour_to_stroke(contour: np.ndarray, detail_level: int, base_weight: float) -> Stroke:
    perimeter = float(cv2.arcLength(contour, True))
    detail_norm = (max(1, min(10, detail_level)) - 1) / 9.0
    if perimeter <= 0:
        raw_points = contour.reshape(-1, 2)
    else:
        max_ratio = 0.02
        min_ratio = 0.002
        epsilon_ratio = max_ratio - detail_norm * (max_ratio - min_ratio)
        epsilon = max(0.35, perimeter * epsilon_ratio)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        raw_points = approx.reshape(-1, 2) if approx.shape[0] >= 2 else contour.reshape(-1, 2)

    max_smoothing_iterations = 4
    min_smoothing_iterations = 0
    smoothing_iterations = int(
        round(
            max_smoothing_iterations
            - detail_norm * (max_smoothing_iterations - min_smoothing_iterations)
        )
    )
    smoothing_iterations = max(0, smoothing_iterations)
    
    if smoothing_iterations > 0:
        smoothed_points = _smooth_with_bezier(raw_points, num_samples=max(20, int(perimeter / 5)))
    else:
        smoothed_points = raw_points.astype(np.float32)
    
    points = [(float(x), float(y)) for x, y in smoothed_points]
    return Stroke(points=points, stroke_type="outline", weight=base_weight)


def build_stroke_plan(bgr_image: np.ndarray, detail_level: int, threshold_offset: int = 8) -> List[Stroke]:
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    median = np.median(blur)
    lower = int(max(0, (1.0 - 0.33) * median))
    upper = int(min(255, (1.0 + 0.33) * median))
    edges = cv2.Canny(blur, lower, upper, apertureSize=3, L2gradient=True)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    outlines: List[Stroke] = []
    outline_weight = 1.4 + 0.15 * (detail_level / 10.0)
    for contour in contours:
        if cv2.arcLength(contour, True) < 25:
            continue
        outlines.append(_contour_to_stroke(contour, detail_level, outline_weight))

    block = 15 + detail_level * 2
    block = block + 1 if block % 2 == 0 else block
    offset = int(np.clip(threshold_offset, -20, 20))
    binary = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block,
        offset,
    )
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    binary = cv2.medianBlur(binary, 3)

    orientation_field, coherence_field = compute_orientation_field(blur, smoothing_sigma=2.5)
    streamlines = generate_stroke_streamlines(
        binary,
        orientation_field,
        coherence_field,
        detail_level=detail_level,
    )

    intensity = 1.0 - gray.astype(np.float32) / 255.0
    texture_strokes: List[Stroke] = []
    for path in streamlines:
        if len(path) < 5:
            continue
        path_array = np.array([[x, y] for x, y in path], dtype=np.float32)
        if len(path) >= 3:
            smooth = _smooth_with_bezier(path_array, num_samples=max(15, len(path) * 2))
        else:
            smooth = _smooth_polyline(path_array, max(1, int(detail_level / 4)))
        points = [(float(x), float(y)) for x, y in smooth]

        sample_vals = []
        for px, py in points:
            xi = int(np.clip(round(px), 0, bgr_image.shape[1] - 1))
            yi = int(np.clip(round(py), 0, bgr_image.shape[0] - 1))
            sample_vals.append(intensity[yi, xi])
        if not sample_vals:
            continue
        avg_intensity = float(np.mean(sample_vals))
        weight = 0.5 + avg_intensity * 1.2
        texture_strokes.append(
            Stroke(
                points=points,
                stroke_type="texture",
                weight=weight,
            )
        )

    return outlines + texture_strokes


def generate_stroke_streamlines(
    binary_mask: np.ndarray,
    orientation_field: np.ndarray,
    coherence_field: np.ndarray,
    detail_level: int,
    max_strokes: int | None = None,
) -> List[List[Tuple[int, int]]]:
    mask = binary_mask > 0
    height, width = mask.shape
    if height == 0 or width == 0:
        return []

    distance = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 3)
    distance_norm = distance / (distance.max() + 1e-6)
    coherence_norm = coherence_field / (coherence_field.max() + 1e-6)

    detail = max(1, min(10, int(detail_level)))
    seed_spacing = max(3, int(16 - 1.2 * detail))
    min_coherence = 0.15 + 0.05 * (detail / 10.0)
    base_step = 1.0
    max_steps = int(80 + detail * 35)

    candidates = []
    for y in range(0, height, seed_spacing):
        for x in range(0, width, seed_spacing):
            if not mask[y, x]:
                continue
            score = coherence_norm[y, x] * (0.5 + distance_norm[y, x])
            if score < min_coherence:
                continue
            candidates.append((score, y, x))

    if not candidates:
        return []

    candidates.sort(reverse=True)
    if max_strokes is None:
        max_strokes = 120 + detail * 50

    visited = np.zeros(mask.shape, dtype=np.uint8)
    strokes: List[List[Tuple[int, int]]] = []

    def _walk(start_y: float, start_x: float, direction: int) -> List[Tuple[int, int]]:
        path: List[Tuple[int, int]] = []
        y = start_y
        x = start_x
        for _ in range(max_steps):
            yi = int(round(y))
            xi = int(round(x))
            if yi < 0 or yi >= height or xi < 0 or xi >= width:
                break
            if not mask[yi, xi]:
                break
            if coherence_norm[yi, xi] < min_coherence:
                break

            angle = orientation_field[yi, xi]
            dx = np.cos(angle) * direction
            dy = np.sin(angle) * direction
            if abs(dx) < 1e-4 and abs(dy) < 1e-4:
                break

            y += dy * base_step
            x += dx * base_step

            yi2 = int(round(y))
            xi2 = int(round(x))
            if yi2 < 0 or yi2 >= height or xi2 < 0 or xi2 >= width:
                break
            if not mask[yi2, xi2]:
                break

            path.append((xi2, yi2))
        return path

    for score, sy, sx in candidates:
        if len(strokes) >= max_strokes:
            break
        if visited[sy, sx]:
            continue

        forward = _walk(float(sy), float(sx), 1)
        backward = _walk(float(sy), float(sx), -1)
        backward.reverse()
        stroke_points = backward + [(sx, sy)] + forward
        if len(stroke_points) < 6:
            continue

        for px, py in stroke_points:
            if 0 <= py < height and 0 <= px < width:
                visited[py, px] = 1
        strokes.append(stroke_points)

    return strokes
