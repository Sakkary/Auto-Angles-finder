#!/usr/bin/env python3
"""Detect connection angles around a circular node in a plan drawing."""

import argparse
from typing import List, Tuple

import cv2
import numpy as np

# Circle selection constraints. These can be tuned for specific datasets.
MIN_RADIUS = 10
MAX_RADIUS = 200
HOUGH_DP = 1.2
HOUGH_MINDIST = 20
HOUGH_PARAM1 = 100
HOUGH_PARAM2 = 30

# Connection tolerance: how close the closest point on a segment should be to
# the circle radius to be considered a touching pipe connection.
CONNECTION_TOLERANCE_RATIO = 0.10
# Angle merge tolerance: treat nearly-collinear segments as the same connection.
ANGLE_MERGE_TOLERANCE_DEG = 3.0


def detect_circle(
    blurred_gray: np.ndarray, min_radius: int | None = None, max_radius: int | None = None
) -> Tuple[int, int, int]:
    """Detect the most likely node circle in the drawing.

    If multiple circles are detected, we:
    1. Filter circles within a radius range [min_radius, max_radius].
    2. Pick the remaining circle with the largest radius.
    """
    if min_radius is None:
        min_radius = MIN_RADIUS
    if max_radius is None:
        max_radius = MAX_RADIUS
    circles = cv2.HoughCircles(
        blurred_gray,
        cv2.HOUGH_GRADIENT,
        dp=HOUGH_DP,
        minDist=HOUGH_MINDIST,
        param1=HOUGH_PARAM1,
        param2=HOUGH_PARAM2,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    if circles is None:
        raise RuntimeError("No circles detected; try adjusting HoughCircles settings.")

    circles = np.round(circles[0, :]).astype(int)
    filtered = [c for c in circles if min_radius <= c[2] <= max_radius]
    candidates = filtered if filtered else circles
    print(
        f"Detected {len(circles)} circle(s), using {len(candidates)} candidate(s) "
        f"within radius bounds."
    )

    # Choose the largest radius circle as the most likely node.
    cx, cy, r = max(candidates, key=lambda c: c[2])
    return int(cx), int(cy), int(r)


def _closest_point_on_segment(
    cx: float, cy: float, x1: float, y1: float, x2: float, y2: float
) -> Tuple[float, float]:
    """Return the closest point on the segment (x1,y1)-(x2,y2) to (cx,cy)."""
    vx, vy = x2 - x1, y2 - y1
    denom = vx * vx + vy * vy
    if denom == 0:
        return float(x1), float(y1)
    t = ((cx - x1) * vx + (cy - y1) * vy) / denom
    t = max(0.0, min(1.0, t))
    px = x1 + t * vx
    py = y1 + t * vy
    return px, py


def detect_connections(
    edges: np.ndarray,
    center: Tuple[int, int],
    radius: int,
    tolerance_ratio: float,
) -> List[Tuple[int, int, int, int]]:
    """Detect line segments that connect to the circle node."""
    cx, cy = center
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=20,
        maxLineGap=10,
    )

    if lines is None:
        return []

    tolerance = max(2.0, tolerance_ratio * radius)
    connections = []

    for (x1, y1, x2, y2) in lines[:, 0]:
        px, py = _closest_point_on_segment(cx, cy, x1, y1, x2, y2)
        d = np.hypot(px - cx, py - cy)

        # A segment is a connection if the closest point is near the radius.
        if abs(d - radius) < tolerance:
            connections.append((x1, y1, x2, y2))

    return connections


def compute_angles(
    center: Tuple[int, int],
    segments: List[Tuple[int, int, int, int]],
    angle_merge_tol: float,
) -> Tuple[List[Tuple[Tuple[int, int, int, int], float]], List[float], List[float]]:
    """Compute absolute connection angles and sector angles.

    Segments with nearly identical angles are merged to avoid counting the same
    connection multiple times when HoughLinesP returns overlapping segments.
    Angles are derived from the closest point on each segment to the center.
    """
    cx, cy = center
    segment_angles = []

    for x1, y1, x2, y2 in segments:
        px, py = _closest_point_on_segment(cx, cy, x1, y1, x2, y2)

        # Invert y to convert to a standard Cartesian coordinate system.
        dx = px - cx
        dy = cy - py
        angle = np.degrees(np.arctan2(dy, dx))
        # Normalize to [0, 360) degrees.
        angle = angle % 360
        distance = np.hypot(px - cx, py - cy)
        segment_angles.append(((x1, y1, x2, y2), angle, distance))

    # Merge segments with angles within angle_merge_tol degrees.
    merged_segments = []
    for segment, angle, distance in segment_angles:
        matched = False
        for idx, (stored_segment, stored_angle, stored_distance) in enumerate(
            merged_segments
        ):
            diff = abs(angle - stored_angle)
            diff = min(diff, 360 - diff)
            if diff <= angle_merge_tol:
                matched = True
                if distance < stored_distance:
                    merged_segments[idx] = (segment, angle, distance)
                break
        if not matched:
            merged_segments.append((segment, angle, distance))

    sorted_angles = sorted(angle for _, angle, _ in merged_segments)

    sector_angles = []
    if sorted_angles:
        for i in range(len(sorted_angles) - 1):
            sector_angles.append(sorted_angles[i + 1] - sorted_angles[i])
        sector_angles.append((sorted_angles[0] + 360) - sorted_angles[-1])

    annotated_segments = [(segment, angle) for segment, angle, _ in merged_segments]
    return annotated_segments, sorted_angles, sector_angles


def annotate_image(
    image: np.ndarray,
    center: Tuple[int, int],
    radius: int,
    segment_angles: List[Tuple[Tuple[int, int, int, int], float]],
) -> np.ndarray:
    """Draw the detected circle and connection angles on the image."""
    annotated = image.copy()
    cx, cy = center

    cv2.circle(annotated, (cx, cy), radius, (0, 255, 0), 2)
    cv2.circle(annotated, (cx, cy), 3, (255, 0, 0), -1)

    for (x1, y1, x2, y2), angle in segment_angles:
        px, py = _closest_point_on_segment(cx, cy, x1, y1, x2, y2)
        x_end, y_end = int(round(px)), int(round(py))

        cv2.line(annotated, (x1, y1), (x2, y2), (0, 165, 255), 2)
        label = f"{angle:.1f}°"
        cv2.putText(
            annotated,
            label,
            (x_end + 5, y_end - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    return annotated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure angles between pipe connections around a circular node."
    )
    parser.add_argument("image_path", help="Path to the input image file.")
    parser.add_argument(
        "--min-radius",
        type=int,
        default=MIN_RADIUS,
        help="Minimum circle radius to consider (pixels).",
    )
    parser.add_argument(
        "--max-radius",
        type=int,
        default=MAX_RADIUS,
        help="Maximum circle radius to consider (pixels).",
    )
    parser.add_argument(
        "--angle-merge-tol",
        type=float,
        default=ANGLE_MERGE_TOLERANCE_DEG,
        help="Angle tolerance (degrees) to merge near-duplicate connections.",
    )
    parser.add_argument(
        "--connection-tol-ratio",
        type=float,
        default=CONNECTION_TOLERANCE_RATIO,
        help="Tolerance ratio for how close a segment must be to the circle radius.",
    )
    return parser.parse_args()


def analyze_image_array(image: np.ndarray):
    """Run the full pipeline on an image array.

    Returns a dict with numeric results and the annotated image.
    """
    if image is None:
        raise ValueError("image is None in analyze_image_array")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
    edges = cv2.Canny(blurred, 50, 150)

    cx, cy, r = detect_circle(blurred)
    connections = detect_connections(edges, (cx, cy), r, CONNECTION_TOLERANCE_RATIO)
    segment_angles, angles, sector_angles = compute_angles(
        (cx, cy), connections, ANGLE_MERGE_TOLERANCE_DEG
    )
    annotated = annotate_image(image, (cx, cy), r, segment_angles)

    return {
        "center": (int(cx), int(cy)),
        "radius": int(r),
        "angles": [float(a) for a in angles],
        "sector_angles": [float(a) for a in sector_angles],
        "annotated": annotated,
    }


def main() -> None:
    args = parse_args()
    image = cv2.imread(args.image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {args.image_path}")

    global MIN_RADIUS
    global MAX_RADIUS
    global ANGLE_MERGE_TOLERANCE_DEG
    global CONNECTION_TOLERANCE_RATIO
    MIN_RADIUS = args.min_radius
    MAX_RADIUS = args.max_radius
    ANGLE_MERGE_TOLERANCE_DEG = args.angle_merge_tol
    CONNECTION_TOLERANCE_RATIO = args.connection_tol_ratio

    result = analyze_image_array(image)

    cx, cy = result["center"]
    r = result["radius"]
    angles = result["angles"]
    sector_angles = result["sector_angles"]
    annotated = result["annotated"]

    print(f"Detected circle center: ({cx}, {cy}), radius: {r}")
    print("Connection angles (degrees):", [round(a, 2) for a in angles])
    print(
        "Sector angles (degrees):",
        [round(a, 2) for a in sector_angles],
        "sum:",
        round(sum(sector_angles), 2),
    )
    if not angles:
        print("Warning: no connections detected for this circle.")

    output_path = "annotated_output.png"
    cv2.imwrite(output_path, annotated)
    print(f"Annotated output saved to {output_path}")


if __name__ == "__main__":
    main()
