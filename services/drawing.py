import threading
import pyautogui
import numpy as np
import cv2
import random


def draw_edges_with_pyautogui(edges_image: "np.ndarray", origin_x: int, origin_y: int, cancel_event: "threading.Event") -> None:
    image_height, image_width = edges_image.shape

    for pixel_y in range(image_height):
        if cancel_event.is_set():
            break
        for pixel_x in range(image_width):
            if cancel_event.is_set():
                break
            if edges_image[pixel_y, pixel_x] != 0:
                pyautogui.click(origin_x + pixel_x, origin_y + pixel_y)


def draw_contours_with_pyautogui(contours: "list[np.ndarray]", origin_x: int, origin_y: int, cancel_event: "threading.Event", approx_epsilon_px: float = 3.0) -> None:
    for contour in contours:
        if cancel_event.is_set():
            break
        if contour.shape[0] < 2:
            continue
        peri = cv2.arcLength(contour, True)
        epsilon = max(1.0, approx_epsilon_px)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        points = approx.reshape(-1, 2)
        start_x = origin_x + int(points[0][0])
        start_y = origin_y + int(points[0][1])
        pyautogui.moveTo(start_x, start_y)
        pyautogui.mouseDown()
        for px, py in points[1:]:
            if cancel_event.is_set():
                break
            pyautogui.dragTo(origin_x + int(px), origin_y + int(py), duration=0)
        pyautogui.mouseUp()


def draw_scanline_fill_with_pyautogui(binary_image: "np.ndarray", origin_x: int, origin_y: int, cancel_event: "threading.Event", line_spacing_px: int = 2) -> None:
    height, width = binary_image.shape
    spacing = max(1, int(line_spacing_px))
    
    memory_cell_size = max(2, spacing)
    memory_height = (height + memory_cell_size - 1) // memory_cell_size
    memory_width = (width + memory_cell_size - 1) // memory_cell_size
    drawn_areas = np.zeros((memory_height, memory_width), dtype=np.uint8)
    
    lines_drawn = 0
    total_segments = 0
    
    for y in range(0, height, spacing):
        if cancel_event.is_set():
            break
        
        row = binary_image[y]
        memory_y = y // memory_cell_size
        
        x = 0
        while x < width:
            if cancel_event.is_set():
                break
            
            if row[x] == 255:
                start = x
                while x < width and row[x] == 255:
                    x += 1
                end = x - 1
                
                memory_start_x = start // memory_cell_size
                memory_end_x = end // memory_cell_size
                
                already_drawn = False
                for mem_x in range(memory_start_x, min(memory_end_x + 1, memory_width)):
                    if drawn_areas[memory_y, mem_x] > 0:
                        already_drawn = True
                        break
                
                if not already_drawn:
                    pyautogui.moveTo(origin_x + start, origin_y + y)
                    pyautogui.mouseDown()
                    pyautogui.dragTo(origin_x + end, origin_y + y, duration=0)
                    pyautogui.mouseUp()
                    
                    for mem_x in range(memory_start_x, min(memory_end_x + 1, memory_width)):
                        drawn_areas[memory_y, mem_x] = 1
                    
                    lines_drawn += 1
                    total_segments += 1
            else:
                x += 1
    
    if spacing > 1 and not cancel_event.is_set():
        fine_spacing = 1
        for y in range(0, height, fine_spacing):
            if cancel_event.is_set():
                break
            
            if y % spacing == 0:
                continue
            
            row = binary_image[y]
            memory_y = y // memory_cell_size
            
            x = 0
            while x < width:
                if cancel_event.is_set():
                    break
                
                if row[x] == 255:
                    start = x
                    while x < width and row[x] == 255:
                        x += 1
                    end = x - 1
                    
                    memory_start_x = start // memory_cell_size
                    memory_end_x = end // memory_cell_size
                    
                    needs_drawing = False
                    for mem_x in range(memory_start_x, min(memory_end_x + 1, memory_width)):
                        if drawn_areas[memory_y, mem_x] == 0:
                            needs_drawing = True
                            break
                    
                    if needs_drawing:
                        pyautogui.moveTo(origin_x + start, origin_y + y)
                        pyautogui.mouseDown()
                        pyautogui.dragTo(origin_x + end, origin_y + y, duration=0)
                        pyautogui.mouseUp()
                        
                        for mem_x in range(memory_start_x, min(memory_end_x + 1, memory_width)):
                            drawn_areas[memory_y, mem_x] = 1
                        
                        total_segments += 1
                else:
                    x += 1


def draw_stroke_paths_with_pyautogui(
    stroke_paths: "list[list[tuple[int, int]]]",
    origin_x: int,
    origin_y: int,
    cancel_event: "threading.Event",
) -> None:
    for path in stroke_paths:
        if cancel_event.is_set():
            break
        if len(path) < 2:
            continue
        start_x, start_y = path[0]
        pyautogui.moveTo(origin_x + int(start_x), origin_y + int(start_y))
        pyautogui.mouseDown()
        for px, py in path[1:]:
            if cancel_event.is_set():
                break
            pyautogui.dragTo(origin_x + int(px), origin_y + int(py), duration=0)
        pyautogui.mouseUp()


def draw_scribble_fill_with_pyautogui(
    binary_image: "np.ndarray",
    origin_x: int,
    origin_y: int,
    cancel_event: "threading.Event",
    line_spacing_px: int = 3,
    jitter_px: int = 1,
    segment_step_px: int = 5,
) -> None:
    height, width = binary_image.shape
    spacing = max(1, int(line_spacing_px))
    jitter = max(0, int(jitter_px))
    base_step = max(2, int(segment_step_px))

    for y in range(0, height, spacing):
        if cancel_event.is_set():
            break
        row = binary_image[y]
        x = 0
        while x < width:
            if cancel_event.is_set():
                break
            if row[x] == 255:
                start = x
                while x < width and row[x] == 255:
                    x += 1
                end = x - 1
                if end <= start:
                    continue
                points = []
                cur = start
                points.append((origin_x + cur, origin_y + y + random.randint(-jitter, jitter)))
                while cur < end:
                    step = base_step + random.randint(-2, 2)
                    step = max(2, step)
                    cur = min(end, cur + step)
                    jy = y + random.randint(-jitter, jitter)
                    points.append((origin_x + cur, origin_y + jy))
                if len(points) >= 2:
                    pyautogui.moveTo(points[0][0], points[0][1])
                    pyautogui.mouseDown()
                    for px, py in points[1:]:
                        if cancel_event.is_set():
                            break
                        pyautogui.dragTo(px, py, duration=0)
                    pyautogui.mouseUp()
            else:
                x += 1


def draw_serpentine_scribble_fill_with_pyautogui(
    binary_image: "np.ndarray",
    origin_x: int,
    origin_y: int,
    cancel_event: "threading.Event",
    line_spacing_px: int = 3,
    jitter_px: int = 1,
    segment_step_px: int = 6,
) -> None:
    height, width = binary_image.shape
    spacing = max(1, int(line_spacing_px))
    jitter = max(0, int(jitter_px))
    base_step = max(2, int(segment_step_px))

    left_to_right = True
    for row_y in range(0, height, spacing):
        if cancel_event.is_set():
            break
        row = binary_image[row_y]
        segments: list[tuple[int, int]] = []
        x = 0
        while x < width:
            if row[x] == 255:
                start = x
                while x < width and row[x] == 255:
                    x += 1
                end = x - 1
                segments.append((start, end))
            x += 1
        if not left_to_right:
            segments.reverse()

        for start, end in segments:
            if cancel_event.is_set():
                break
            points = []
            cur = start if left_to_right else end
            points.append((origin_x + cur, origin_y + row_y + random.randint(-jitter, jitter)))
            while (cur < end if left_to_right else cur > start):
                step = base_step + random.randint(-2, 2)
                step = max(2, step)
                cur = min(end, cur + step) if left_to_right else max(start, cur - step)
                jy = row_y + random.randint(-jitter, jitter)
                points.append((origin_x + cur, origin_y + jy))
            if len(points) >= 2:
                pyautogui.moveTo(points[0][0], points[0][1])
                pyautogui.mouseDown()
                for px, py in points[1:]:
                    if cancel_event.is_set():
                        break
                    pyautogui.dragTo(px, py, duration=0)
                pyautogui.mouseUp()

        left_to_right = not left_to_right
