import cv2
import numpy as np
import pyautogui
import time

time.sleep(5)
img = cv2.imread("sample.jpg")
if img is None:
    print("Image not found!")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
h, w = gray.shape
scale = 300 / max(h, w)
gray = cv2.resize(gray, (int(w*scale), int(h*scale)))
edges = cv2.Canny(gray, 100, 200)
screen_w, screen_h = pyautogui.size()
start_x = screen_w // 4
start_y = screen_h // 4
draw_w, draw_h = gray.shape[1], gray.shape[0]

for y in range(draw_h):
    for x in range(draw_w):
        if edges[y, x] != 0:
            px = start_x + x
            py = start_y + y
            pyautogui.click(px, py)
