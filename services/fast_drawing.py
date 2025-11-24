import ctypes
import ctypes.wintypes
import threading
import time
import math
import numpy as np
from typing import List, Tuple


class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.wintypes.LONG),
        ("dy", ctypes.wintypes.LONG),
        ("mouseData", ctypes.wintypes.DWORD),
        ("dwFlags", ctypes.wintypes.DWORD),
        ("time", ctypes.wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.wintypes.ULONG)),
    ]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.wintypes.WORD),
        ("wScan", ctypes.wintypes.WORD),
        ("dwFlags", ctypes.wintypes.DWORD),
        ("time", ctypes.wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.wintypes.ULONG)),
    ]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", ctypes.wintypes.DWORD),
        ("wParamL", ctypes.wintypes.WORD),
        ("wParamH", ctypes.wintypes.WORD),
    ]


class INPUT(ctypes.Structure):
    class _INPUT(ctypes.Union):
        _fields_ = [("ki", KEYBDINPUT), ("mi", MOUSEINPUT), ("hi", HARDWAREINPUT)]

    _anonymous_ = ("_input",)
    _fields_ = [("type", ctypes.wintypes.DWORD), ("_input", _INPUT)]


MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_ABSOLUTE = 0x8000
INPUT_MOUSE = 0

user32 = ctypes.windll.user32
SendInput = user32.SendInput
SetCursorPos = user32.SetCursorPos
GetSystemMetrics = user32.GetSystemMetrics

SM_CXSCREEN = 0
SM_CYSCREEN = 1


def fast_move_to(x: int, y: int) -> None:
    SetCursorPos(int(x), int(y))


def fast_click(x: int, y: int) -> None:
    fast_move_to(x, y)
    time.sleep(0.001)
    
    mouse_down = INPUT()
    mouse_down.type = INPUT_MOUSE
    mouse_down.mi.dwFlags = MOUSEEVENTF_LEFTDOWN
    SendInput(1, ctypes.byref(mouse_down), ctypes.sizeof(INPUT))
    
    time.sleep(0.001)
    
    mouse_up = INPUT()
    mouse_up.type = INPUT_MOUSE
    mouse_up.mi.dwFlags = MOUSEEVENTF_LEFTUP
    SendInput(1, ctypes.byref(mouse_up), ctypes.sizeof(INPUT))


def fast_mouse_down() -> None:
    mouse_down = INPUT()
    mouse_down.type = INPUT_MOUSE
    mouse_down.mi.dwFlags = MOUSEEVENTF_LEFTDOWN
    SendInput(1, ctypes.byref(mouse_down), ctypes.sizeof(INPUT))


def fast_mouse_up() -> None:
    mouse_up = INPUT()
    mouse_up.type = INPUT_MOUSE
    mouse_up.mi.dwFlags = MOUSEEVENTF_LEFTUP
    SendInput(1, ctypes.byref(mouse_up), ctypes.sizeof(INPUT))


def calculate_curvature(points: List[Tuple[float, float]], idx: int, window: int = 3) -> float:
    if idx < window or idx >= len(points) - window:
        return 0.0
    
    p1 = points[idx - window]
    p2 = points[idx]
    p3 = points[idx + window]
    
    angle1 = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
    angle2 = math.atan2(p3[1] - p2[1], p3[0] - p2[0])
    
    curvature = abs(angle2 - angle1)
    if curvature > math.pi:
        curvature = 2 * math.pi - curvature
    
    return curvature


def fast_drag_smooth(points: List[Tuple[int, int]], cancel_event: threading.Event) -> None:
    if not points or len(points) < 2:
        return
    
    fast_move_to(points[0][0], points[0][1])
    time.sleep(0.001)
    
    fast_mouse_down()
    time.sleep(0.001)
    
    for i in range(1, len(points)):
        if cancel_event.is_set():
            break
        
        x, y = points[i]
        
        curvature = calculate_curvature(points, i)
        
        base_delay = 0.0005
        curvature_delay = curvature * 0.002
        delay = base_delay + curvature_delay
        
        fast_move_to(int(x), int(y))
        time.sleep(delay)
    
    fast_mouse_up()


def fast_drag_instant(points: List[Tuple[int, int]], cancel_event: threading.Event) -> None:
    if not points or len(points) < 2:
        return
    
    fast_move_to(points[0][0], points[0][1])
    time.sleep(0.001)
    fast_mouse_down()
    
    for x, y in points[1:]:
        if cancel_event.is_set():
            break
        fast_move_to(int(x), int(y))
    
    fast_mouse_up()


def draw_edges_fast(edges_image: np.ndarray, origin_x: int, origin_y: int, cancel_event: threading.Event) -> None:
    image_height, image_width = edges_image.shape
    
    for pixel_y in range(image_height):
        if cancel_event.is_set():
            break
        for pixel_x in range(image_width):
            if cancel_event.is_set():
                break
            if edges_image[pixel_y, pixel_x] != 0:
                fast_click(origin_x + pixel_x, origin_y + pixel_y)


def draw_scanline_fill_fast(
    binary_image: np.ndarray,
    origin_x: int,
    origin_y: int,
    cancel_event: threading.Event,
    line_spacing_px: int = 2,
) -> None:
    height, width = binary_image.shape
    spacing = max(1, int(line_spacing_px))
    memory_cell_size = max(1, spacing)
    memory_height = (height + memory_cell_size - 1) // memory_cell_size
    memory_width = (width + memory_cell_size - 1) // memory_cell_size
    drawn_areas = np.zeros((memory_height, memory_width), dtype=np.uint8)
    
    lines_drawn = 0
    
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
                    points = [(origin_x + start, origin_y + y), (origin_x + end, origin_y + y)]
                    fast_drag_instant(points, cancel_event)
                    
                    for mem_x in range(memory_start_x, min(memory_end_x + 1, memory_width)):
                        drawn_areas[memory_y, mem_x] = 1
                    
                    lines_drawn += 1
            else:
                x += 1
    
    if spacing > 1 and not cancel_event.is_set():
        for y in range(0, height, 1):
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
                        points = [(origin_x + start, origin_y + y), (origin_x + end, origin_y + y)]
                        fast_drag_instant(points, cancel_event)
                        
                        for mem_x in range(memory_start_x, min(memory_end_x + 1, memory_width)):
                            drawn_areas[memory_y, mem_x] = 1
                else:
                    x += 1


def draw_stroke_paths_fast(
    stroke_paths: List[List[Tuple[int, int]]],
    origin_x: int,
    origin_y: int,
    cancel_event: threading.Event,
) -> None:
    for path in stroke_paths:
        if cancel_event.is_set():
            break
        if len(path) < 2:
            continue
        
        absolute_path = [(origin_x + int(px), origin_y + int(py)) for px, py in path]
        
        fast_drag_smooth(absolute_path, cancel_event)
