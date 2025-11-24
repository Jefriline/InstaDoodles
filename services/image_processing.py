import math
import cv2
import numpy as np
try:
    from numba import jit, prange
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range

try:
    from cv2 import ximgproc
    _HAS_XIMGPROC = True
except Exception:
    _HAS_XIMGPROC = False


def load_bgr_image(image_path: str) -> "np.ndarray":
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    return image


def convert_bgr_to_rgb(bgr_image: "np.ndarray") -> "np.ndarray":
    return cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)


def convert_bgr_to_rgba_with_alpha(bgr_image: "np.ndarray", alpha_percent: int) -> "np.ndarray":
    rgb_image = convert_bgr_to_rgb(bgr_image)
    alpha_value = max(0, min(255, int(255 * alpha_percent / 100)))
    height, width = rgb_image.shape[:2]
    alpha_channel = np.full((height, width), alpha_value, dtype=np.uint8)
    rgba_image = np.dstack((rgb_image, alpha_channel))
    return rgba_image


def convert_bgr_to_gray(bgr_image: "np.ndarray") -> "np.ndarray":
    return cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)


def gaussian_blur_gray(gray_image: "np.ndarray", kernel_size: int = 5) -> "np.ndarray":
    if kernel_size % 2 == 0:
        kernel_size = kernel_size + 1
    return cv2.GaussianBlur(gray_image, (kernel_size, kernel_size), 0)


def threshold_binary_inv(gray_image: "np.ndarray", threshold_value: int | None = None) -> "np.ndarray":
    if threshold_value is None:
        _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return binary
    _, binary = cv2.threshold(gray_image, int(threshold_value), 255, cv2.THRESH_BINARY_INV)
    return binary


def find_external_contours(binary_image: "np.ndarray") -> "list[np.ndarray]":
    image_height, image_width = binary_image.shape
    image_area = image_width * image_height

    contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None or len(contours) == 0:
        return []

    filtered_contours: list[np.ndarray] = []
    min_contour_area = 20
    border_margin = 5

    for contour in contours:
        if len(contour) < 3:
            continue
        area = cv2.contourArea(contour)
        if area < min_contour_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        touches_all_borders = (
            x <= border_margin and
            y <= border_margin and
            (x + w) >= (image_width - border_margin) and
            (y + h) >= (image_height - border_margin)
        )
        if touches_all_borders:
            area_ratio = area / image_area
            if area_ratio > 0.90:
                continue

        filtered_contours.append(contour)

    filtered_contours.sort(key=cv2.contourArea, reverse=True)
    return filtered_contours

def scale_bgr_image(bgr_image: "np.ndarray", percent: int) -> "np.ndarray":
    height, width = bgr_image.shape[:2]
    new_width = max(1, int(width * percent / 100))
    new_height = max(1, int(height * percent / 100))
    interpolation = cv2.INTER_AREA if percent < 100 else cv2.INTER_CUBIC
    return cv2.resize(bgr_image, (new_width, new_height), interpolation=interpolation)


def compute_canny_edges(gray_image: "np.ndarray", low_threshold: int = 100, high_threshold: int = 200) -> "np.ndarray":
    blurred = cv2.GaussianBlur(gray_image, (3, 3), 0)
    
    median = np.median(blurred)
    sigma = 0.33
    
    edges_fine = None
    edges_medium = None
    edges_coarse = None
    
    lower_fine = int(max(5, (1.0 - sigma * 1.5) * median - 40))
    upper_fine = int(min(255, (1.0 + sigma * 0.5) * median))
    edges_fine = cv2.Canny(blurred, lower_fine, upper_fine, apertureSize=3, L2gradient=True)
    
    lower_medium = int(max(10, (1.0 - sigma) * median - 20))
    upper_medium = int(min(255, (1.0 + sigma) * median + 20))
    edges_medium = cv2.Canny(blurred, lower_medium, upper_medium, apertureSize=3, L2gradient=True)
    
    if low_threshold != 100 or high_threshold != 200:
        lower_coarse = low_threshold
        upper_coarse = high_threshold
    else:
        lower_coarse = int(max(20, (1.0 - sigma * 0.5) * median))
        upper_coarse = int(min(255, (1.0 + sigma * 1.5) * median + 40))
    edges_coarse = cv2.Canny(blurred, lower_coarse, upper_coarse, apertureSize=3, L2gradient=True)
    
    edges = cv2.bitwise_or(edges_fine, edges_medium)
    edges = cv2.bitwise_or(edges, edges_coarse)
    
    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    return edges


def morphology_clean(binary_image: "np.ndarray", close_kernel_size: int = 3, open_kernel_size: int = 3, iterations: int = 1) -> "np.ndarray":
    close_kernel_size = max(1, int(close_kernel_size))
    open_kernel_size = max(1, int(open_kernel_size))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_kernel_size, close_kernel_size))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (open_kernel_size, open_kernel_size))
    cleaned = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, close_kernel, iterations=iterations)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, open_kernel, iterations=iterations)
    return cleaned


def thinning(binary_image: "np.ndarray") -> "np.ndarray":
    if binary_image.dtype != np.uint8:
        binary_image = binary_image.astype(np.uint8)
    _, bin_img = cv2.threshold(binary_image, 127, 255, cv2.THRESH_BINARY)
    if _HAS_XIMGPROC:
        return ximgproc.thinning(bin_img)
    img = bin_img.copy()
    size = np.size(img)
    skeleton = np.zeros(img.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skeleton = cv2.bitwise_or(skeleton, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break
    return skeleton


def compute_fill_params(image_height: int, image_width: int, detail_level: int) -> tuple[int, int, int]:
    detail = max(1, min(10, int(detail_level)))
    base = max(1, min(6, int(round(6 - detail * 0.4))))
    scale_factor = max(1.0, (image_width + image_height) / 800.0)
    spacing = int(max(1, round(base * scale_factor * 0.9)))
    jitter = int(max(1, round(1 + (11 - detail) / 3)))
    step = int(max(3, round(2 * jitter + 4)))
    return spacing, jitter, step


def generate_nested_contours(binary_image: "np.ndarray", detail_level: int) -> list[list["np.ndarray"]]:
    if binary_image.dtype != np.uint8:
        bin_img = binary_image.astype(np.uint8)
    else:
        bin_img = binary_image.copy()

    detail = max(1, min(10, int(detail_level)))
    kernel_size = 3 if detail >= 7 else 5
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    max_layers = max(4, 4 + detail)

    current = bin_img.copy()
    layers: list[list[np.ndarray]] = []

    for _ in range(max_layers):
        contours = find_external_contours(current)
        if not contours:
            break
        layers.append(contours)
        current = cv2.erode(current, kernel, iterations=1)
        if cv2.countNonZero(current) == 0:
            break

    return layers


@jit(nopython=True, cache=True)
def _compute_coherence_fast(ix2: np.ndarray, iy2: np.ndarray, ixy: np.ndarray) -> np.ndarray:
    height, width = ix2.shape
    coherence = np.zeros((height, width), dtype=np.float32)
    
    for y in prange(height):
        for x in prange(width):
            num = (ix2[y, x] - iy2[y, x]) ** 2 + (2.0 * ixy[y, x]) ** 2
            denom = ix2[y, x] + iy2[y, x] + 1e-8
            coherence[y, x] = min(1.0, max(0.0, np.sqrt(num) / denom))
    
    return coherence


def compute_orientation_field(gray_image: "np.ndarray", smoothing_sigma: float = 3.0) -> tuple["np.ndarray", "np.ndarray"]:
    gray_f = gray_image.astype(np.float32) / 255.0
    ix = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    iy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)

    ix2 = ix * ix
    iy2 = iy * iy
    ixy = ix * iy

    if smoothing_sigma > 0:
        ksize = int(max(3, (int(smoothing_sigma * 6) // 2) * 2 + 1))
        ix2 = cv2.GaussianBlur(ix2, (ksize, ksize), smoothing_sigma)
        iy2 = cv2.GaussianBlur(iy2, (ksize, ksize), smoothing_sigma)
        ixy = cv2.GaussianBlur(ixy, (ksize, ksize), smoothing_sigma)

    denom = ix2 - iy2
    orientation = 0.5 * np.arctan2(2.0 * ixy, denom + 1e-8)

    if _HAS_NUMBA:
        coherence = _compute_coherence_fast(ix2, iy2, ixy)
    else:
        coherence = np.sqrt((ix2 - iy2) ** 2 + (2 * ixy) ** 2)
        norm = ix2 + iy2 + 1e-8
        coherence = np.clip(coherence / norm, 0.0, 1.0)

    return orientation.astype(np.float32), coherence.astype(np.float32)


def generate_stroke_streamlines(
    binary_mask: "np.ndarray",
    skeleton: "np.ndarray",
    orientation_field: "np.ndarray",
    coherence_field: "np.ndarray",
    detail_level: int,
    max_strokes: int | None = None,
) -> list[list[tuple[int, int]]]:
    mask = binary_mask > 0
    sk_points = np.column_stack(np.nonzero(skeleton))
    if sk_points.size == 0:
        return []

    detail = max(1, min(10, int(detail_level)))
    if max_strokes is None:
        max_strokes = 60 + detail * 40

    coherence_norm = coherence_field / (coherence_field.max() + 1e-6)
    seed_coherence = coherence_norm[sk_points[:, 0], sk_points[:, 1]]
    order = np.argsort(seed_coherence)[::-1]
    sk_points = sk_points[order]

    visited = np.zeros(mask.shape, dtype=np.uint8)
    strokes: list[list[tuple[int, int]]] = []

    height, width = mask.shape
    base_step = 1.1
    max_steps = int(80 + detail * 30)
    min_coherence = 0.18 + 0.05 * (detail / 10.0)

    def _walk(start_y: float, start_x: float, direction: int) -> list[tuple[int, int]]:
        path: list[tuple[int, int]] = []
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
            dx = math.cos(angle) * direction
            dy = math.sin(angle) * direction
            if dx == 0 and dy == 0:
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

    for (sy, sx) in sk_points:
        if len(strokes) >= max_strokes:
            break
        if visited[sy, sx]:
            continue
        if coherence_norm[sy, sx] < min_coherence:
            continue

        forward = _walk(float(sy), float(sx), 1)
        backward = _walk(float(sy), float(sx), -1)
        backward = backward[::-1]

        stroke_points = backward + [(sx, sy)] + forward
        if len(stroke_points) < 6:
            continue

        for px, py in stroke_points:
            if 0 <= py < height and 0 <= px < width:
                visited[py, px] = 1

        strokes.append(stroke_points)

    return strokes


def apply_floyd_steinberg_dithering(gray_image: "np.ndarray") -> "np.ndarray":
    img = gray_image.astype(np.float32)
    height, width = img.shape
    
    output = np.zeros((height, width), dtype=np.uint8)
    
    for y in range(height):
        for x in range(width):
            old_pixel = img[y, x]
            
            new_pixel = 255.0 if old_pixel > 127.5 else 0.0
            output[y, x] = int(new_pixel)
            
            quant_error = old_pixel - new_pixel
            
            if x + 1 < width:
                img[y, x + 1] += quant_error * 7.0 / 16.0
            if y + 1 < height:
                if x > 0:
                    img[y + 1, x - 1] += quant_error * 3.0 / 16.0
                img[y + 1, x] += quant_error * 5.0 / 16.0
                if x + 1 < width:
                    img[y + 1, x + 1] += quant_error * 1.0 / 16.0
    
    output = 255 - output
    
    return output
