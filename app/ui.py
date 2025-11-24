import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw
import pyautogui

try:
    import customtkinter as ctk
    _HAS_CTK = True

    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
except ImportError:
    _HAS_CTK = False

    ctk = None

import config
from services.image_processing import (
    load_bgr_image,
    scale_bgr_image,
    convert_bgr_to_rgba_with_alpha,
    convert_bgr_to_gray,
    compute_canny_edges,
    compute_fill_params,
    thinning,
)
from services.strokes import Stroke, build_stroke_plan
from services.drawing import (
    draw_stroke_paths_with_pyautogui,
    draw_edges_with_pyautogui,
    draw_scanline_fill_with_pyautogui,
)
from services.fast_drawing import (
    draw_edges_fast,
    draw_scanline_fill_fast,
    draw_stroke_paths_fast,
)
from services.hotkey_listener import GlobalHotkeyListener


TRANSPARENT_COLOR = "#010203"
PANEL_BG = "#1e1e1e"
LABEL_FG = "#f5f5f5"
BUTTON_BG = "#2c2c2c"
BUTTON_FG = "#f8f8f8"


class InstaDoodlesApp:
    def __init__(self, root) -> None:
        self.root = root
        


        if not hasattr(root, 'title') or not root.title():
            self.root.title("InstaDoodles - Align and Draw")
        

        self.root.configure(bg=TRANSPARENT_COLOR)
        try:

            self.root.wm_attributes("-transparentcolor", TRANSPARENT_COLOR)
        except (tk.TclError, AttributeError):

            self.root.configure(bg=PANEL_BG)

        self.current_image_path = config.IMAGE_PATH
        try:
            self.original_bgr_image = load_bgr_image(self.current_image_path)
        except FileNotFoundError:
            self.original_bgr_image = None
            self.current_image_path = None

        self.scale_percent_var = tk.IntVar(value=config.DEFAULT_SCALE_PERCENT)
        self.alpha_percent_var = tk.IntVar(value=config.DEFAULT_ALPHA_PERCENT)
        self.is_topmost_var = tk.BooleanVar(value=bool(config.TOPMOST_DEFAULT))
        self.mode_var = tk.StringVar(value="fill")
        self.threshold_var = tk.IntVar(value=0)
        self.detail_var = tk.IntVar(value=6)



        self.image_canvas = tk.Canvas(
            self.root,
            bd=0,
            highlightthickness=0,
            bg=TRANSPARENT_COLOR
        )
        self.image_canvas.pack(side="top", anchor="nw", fill="both", expand=True)
        self.image_canvas.bind("<ButtonPress-1>", self.on_drag_start)
        self.image_canvas.bind("<B1-Motion>", self.on_drag_motion)
        self.image_canvas_image_id = None


        if _HAS_CTK:
            main_controls_frame = ctk.CTkFrame(self.root)
        else:
            main_controls_frame = tk.Frame(self.root, bg=PANEL_BG)
        main_controls_frame.pack(fill="x", padx=4, pady=4)


        if _HAS_CTK:
            row1_frame = ctk.CTkFrame(main_controls_frame)
        else:
            row1_frame = tk.Frame(main_controls_frame, bg=PANEL_BG)
        row1_frame.pack(fill="x", pady=(0, 4))
        
        mode_frame = row1_frame
        if _HAS_CTK:
            ctk.CTkLabel(mode_frame, text="Mode:").pack(side="left", padx=(10, 5))
            self.fill_radio = ctk.CTkRadioButton(mode_frame, text="Fill", variable=self.mode_var, value="fill", command=self._on_mode_change)
            self.fill_radio.pack(side="left", padx=5)
            # self.outline_radio = ctk.CTkRadioButton(mode_frame, text="Contorno", variable=self.mode_var, value="outline", command=self._on_mode_change)
            # self.outline_radio.pack(side="left", padx=5)
            # self.both_radio = ctk.CTkRadioButton(mode_frame, text="Ambos", variable=self.mode_var, value="both", command=self._on_mode_change)
            # self.both_radio.pack(side="left", padx=5)
            
            self.topmost_check = ctk.CTkCheckBox(row1_frame, text="Always on top", variable=self.is_topmost_var, command=self.apply_topmost)
            self.topmost_check.pack(side="left", padx=(20, 0))
            
            button_frame = row1_frame
            self.prepare_button = ctk.CTkButton(button_frame, text="Prepare Drawing", command=self.prepare_drawing, width=140)
            self.prepare_button.pack(side="right", padx=5)
            self.start_button = ctk.CTkButton(button_frame, text="Start Drawing", command=self.start_drawing, state="disabled", width=140)
            self.start_button.pack(side="right", padx=5)
            self.select_image_button = ctk.CTkButton(button_frame, text="Select Image", command=self._select_image_file, width=120, fg_color="blue")
            self.select_image_button.pack(side="right", padx=5)
            self.cancel_button = ctk.CTkButton(button_frame, text="Cancel", command=self.cancel_drawing, state="disabled", width=100, fg_color="#8a1f1f", hover_color="#a62b2b")
            self.cancel_button.pack(side="right", padx=5)
        else:
            mode_frame = tk.Frame(row1_frame, bg=PANEL_BG)
            mode_frame.pack(side="left", padx=(0, 10))
            tk.Label(mode_frame, text="Mode:", bg=PANEL_BG, fg=LABEL_FG).pack(side="left", padx=(0, 4))
            self.fill_radio = tk.Radiobutton(mode_frame, text="Fill", variable=self.mode_var, value="fill", command=self._on_mode_change, bg=PANEL_BG, fg=LABEL_FG, selectcolor="#444444", activebackground=PANEL_BG, activeforeground=LABEL_FG)
            self.fill_radio.pack(side="left", padx=2)
            # self.outline_radio = tk.Radiobutton(mode_frame, text="Contorno", variable=self.mode_var, value="outline", command=self._on_mode_change, bg=PANEL_BG, fg=LABEL_FG, selectcolor="#444444", activebackground=PANEL_BG, activeforeground=LABEL_FG)
            # self.outline_radio.pack(side="left", padx=2)
            # self.both_radio = tk.Radiobutton(mode_frame, text="Ambos", variable=self.mode_var, value="both", command=self._on_mode_change, bg=PANEL_BG, fg=LABEL_FG, selectcolor="#444444", activebackground=PANEL_BG, activeforeground=LABEL_FG)
            # self.both_radio.pack(side="left", padx=2)

            self.topmost_check = tk.Checkbutton(row1_frame, text="Always on top", variable=self.is_topmost_var, command=self.apply_topmost, bg=PANEL_BG, fg=LABEL_FG, selectcolor="#444444", activebackground=PANEL_BG, activeforeground=LABEL_FG)
            self.topmost_check.pack(side="left", padx=(10, 0))

            button_frame = tk.Frame(row1_frame, bg=PANEL_BG)
            button_frame.pack(side="right", padx=(10, 0))
            self.select_image_button = tk.Button(button_frame, text="Select Image", command=self._select_image_file, width=14, bg="blue", fg="white", activebackground="#4a4a4a", activeforeground="white", relief=tk.FLAT)
            self.select_image_button.pack(side="left", padx=2)
            self.prepare_button = tk.Button(button_frame, text="Prepare Drawing", command=self.prepare_drawing, width=14, bg=BUTTON_BG, fg=BUTTON_FG, activebackground="#3a3a3a", activeforeground=BUTTON_FG, relief=tk.FLAT)
            self.prepare_button.pack(side="left", padx=2)
            self.start_button = tk.Button(button_frame, text="Start Drawing", command=self.start_drawing, state="disabled", width=14, bg=BUTTON_BG, fg=BUTTON_FG, activebackground="#3a3a3a", activeforeground=BUTTON_FG, relief=tk.FLAT)
            self.start_button.pack(side="left", padx=2)
            self.cancel_button = tk.Button(button_frame, text="Cancel", command=self.cancel_drawing, state="disabled", width=10, bg="#8a1f1f", fg="#ffffff", activebackground="#a62b2b", activeforeground="#ffffff", relief=tk.FLAT)
            self.cancel_button.pack(side="left", padx=2)


        if _HAS_CTK:
            row2_frame = ctk.CTkFrame(main_controls_frame)
        else:
            row2_frame = tk.Frame(main_controls_frame, bg=PANEL_BG)
        row2_frame.pack(fill="x", pady=(0, 4))


        if _HAS_CTK:
            ctk.CTkLabel(row2_frame, text="Size (%):").pack(side="left", padx=(10, 5))
            self.scale_slider = ctk.CTkSlider(row2_frame, from_=10, to=200, variable=self.scale_percent_var, command=lambda v: self.on_scale_change(str(int(v))))
            self.scale_slider.pack(side="left", fill="x", expand=True, padx=5)
            self.scale_label = ctk.CTkLabel(row2_frame, textvariable=self.scale_percent_var, width=50)
            self.scale_label.pack(side="left", padx=5)
        else:
            size_frame = tk.Frame(row2_frame, bg=PANEL_BG)
            size_frame.pack(side="left", padx=(0, 8), fill="x", expand=True)
            tk.Label(size_frame, text="Size (%)", width=12, anchor="w", bg=PANEL_BG, fg=LABEL_FG).pack(side="left")
            self.scale_slider = tk.Scale(size_frame, from_=10, to=200, orient="horizontal", variable=self.scale_percent_var, command=self.on_scale_change, bg=PANEL_BG, fg=LABEL_FG, highlightthickness=0, troughcolor="#3d3d3d")
            self.scale_slider.pack(side="left", fill="x", expand=True, padx=(4, 0))


        if _HAS_CTK:
            ctk.CTkLabel(row2_frame, text="Transparency (%):").pack(side="left", padx=(20, 5))
            self.alpha_slider = ctk.CTkSlider(row2_frame, from_=30, to=100, variable=self.alpha_percent_var, command=lambda v: self.on_alpha_change(str(int(v))))
            self.alpha_slider.pack(side="left", fill="x", expand=True, padx=5)
            self.alpha_label = ctk.CTkLabel(row2_frame, textvariable=self.alpha_percent_var, width=50)
            self.alpha_label.pack(side="left", padx=5)
        else:
            alpha_frame = tk.Frame(row2_frame, bg=PANEL_BG)
            alpha_frame.pack(side="left", padx=(0, 8), fill="x", expand=True)
            tk.Label(alpha_frame, text="Transparency (%)", width=15, anchor="w", bg=PANEL_BG, fg=LABEL_FG).pack(side="left")
            self.alpha_slider = tk.Scale(alpha_frame, from_=30, to=100, orient="horizontal", variable=self.alpha_percent_var, command=self.on_alpha_change, bg=PANEL_BG, fg=LABEL_FG, highlightthickness=0, troughcolor="#3d3d3d")
            self.alpha_slider.pack(side="left", fill="x", expand=True, padx=(4, 0))


        if _HAS_CTK:
            row3_frame = ctk.CTkFrame(main_controls_frame)
        else:
            row3_frame = tk.Frame(main_controls_frame, bg=PANEL_BG)
        row3_frame.pack(fill="x")

        if _HAS_CTK:
            ctk.CTkLabel(row3_frame, text="Detail (1-10):").pack(side="left", padx=(10, 5))
            self.detail_slider = ctk.CTkSlider(row3_frame, from_=1, to=10, variable=self.detail_var, number_of_steps=9)
            self.detail_slider.pack(side="left", fill="x", expand=True, padx=5)
            self.detail_label = ctk.CTkLabel(row3_frame, textvariable=self.detail_var, width=50)
            self.detail_label.pack(side="left", padx=5)
        else:
            detail_frame = tk.Frame(row3_frame, bg=PANEL_BG)
            detail_frame.pack(side="left", padx=(0, 8), fill="x", expand=True)
            tk.Label(detail_frame, text="Detail (1-10)", width=12, anchor="w", bg=PANEL_BG, fg=LABEL_FG).pack(side="left")
            self.detail_slider = tk.Scale(detail_frame, from_=1, to=10, orient="horizontal", variable=self.detail_var, bg=PANEL_BG, fg=LABEL_FG, highlightthickness=0, troughcolor="#3d3d3d")
            self.detail_slider.pack(side="left", fill="x", expand=True, padx=(4, 0))

        self.photo_image_ref = None
        self.preview_image_ref = None
        self.drag_offset_x = 0
        self.drag_offset_y = 0

        self.cancel_event = threading.Event()
        self.drawing_thread: Optional["threading.Thread"] = None
        self.stroke_plan: Optional[list[Stroke]] = None
        self.edges_image: Optional["np.ndarray"] = None
        self.use_simple_edges: bool = False
        self.fill_mask: Optional["np.ndarray"] = None
        self.fill_params: Optional[tuple[int, int, int]] = None
        self.countdown_label: Optional["tk.Label"] = None
        self.hotkey_listener: Optional["GlobalHotkeyListener"] = None
        self.cancel_hint_window: Optional["tk.Toplevel"] = None

        try:
            pyautogui.PAUSE = 0
            if hasattr(pyautogui, "MINIMUM_DURATION"):
                pyautogui.MINIMUM_DURATION = 0
            if hasattr(pyautogui, "MINIMUM_SLEEP"):
                pyautogui.MINIMUM_SLEEP = 0
        except Exception:
            pass

        self.apply_topmost()

        if self.original_bgr_image is not None:
            self.update_image_display()
        else:
            if _HAS_CTK:
                self.preview_label.configure(text="Select an image to begin", fg_color="white", text_color="#555555")
            else:
                self.preview_label.config(text="Select an image to begin", fg="#555555")

        if _HAS_CTK:
            preview_frame = ctk.CTkFrame(self.root)
            preview_frame.pack(fill="both", expand=True, padx=4, pady=(0, 6))
            ctk.CTkLabel(preview_frame, text="Preview", font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", padx=10, pady=(10, 5))
            self.preview_label = ctk.CTkLabel(preview_frame, text="Prepare the drawing to see the preview", fg_color="white", text_color="#555555", corner_radius=5)
            self.preview_label.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        else:
            preview_frame = tk.Frame(self.root, bg=PANEL_BG)
            preview_frame.pack(fill="both", expand=True, padx=4, pady=(0, 6))
            tk.Label(preview_frame, text="Preview", bg=PANEL_BG, fg=LABEL_FG).pack(anchor="w")
            self.preview_label = tk.Label(preview_frame, bg="white", bd=1, relief=tk.SOLID)
            self.preview_label.pack(fill="both", expand=True)
            self.preview_label.config(text="Prepare the drawing to see the preview", fg="#555555")

        self.root.bind("<Escape>", self._on_escape_pressed)
        self.root.bind("<Configure>", self._on_window_configure)

    def _set_controls_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        
        if _HAS_CTK:

            self.scale_slider.configure(state=state)
            self.alpha_slider.configure(state=state)
            self.detail_slider.configure(state=state)
            self.topmost_check.configure(state=state)
            self.prepare_button.configure(state=state)
            self.outline_radio.configure(state=state)
            self.fill_radio.configure(state=state)
            self.both_radio.configure(state=state)
            
            if enabled:
                self.start_button.configure(state="normal" if self.stroke_plan else "disabled")
            else:
                self.start_button.configure(state="disabled")
        else:

            self.scale_slider.config(state=state)
            self.alpha_slider.config(state=state)
            self.detail_slider.config(state=state)
            self.topmost_check.config(state=state)
            self.prepare_button.config(state=state)
            self.outline_radio.config(state=state)
            self.fill_radio.config(state=state)
            self.both_radio.config(state=state)
            
            if enabled:
                self.start_button.config(state="normal" if self.stroke_plan else "disabled")
            else:
                self.start_button.config(state="disabled")

    def _on_escape_pressed(self, _: "tk.Event") -> None:
        self.cancel_drawing()

    def on_scale_change(self, _: str) -> None:
        self.update_image_display()
        self.stroke_plan = None
        self.edges_image = None
        self.use_simple_edges = False
        self.fill_mask = None
        self.fill_params = None
        if _HAS_CTK:
            self.start_button.configure(state="disabled")
            self.prepare_button.configure(text="Prepare Drawing")
            if hasattr(self, 'scale_label'):
                self.scale_label.configure(text=str(self.scale_percent_var.get()))
        else:
            self.start_button.config(state="disabled")
            self.prepare_button.config(text="Prepare Drawing")

    def _on_mode_change(self) -> None:
        self.stroke_plan = None
        self.edges_image = None
        self.use_simple_edges = False
        self.fill_mask = None
        self.fill_params = None
        if _HAS_CTK:
            self.start_button.configure(state="disabled")
            self.prepare_button.configure(text="Prepare Drawing")
        else:
            self.start_button.config(state="disabled")
            self.prepare_button.config(text="Prepare Drawing")

    def _on_window_configure(self, _: "tk.Event") -> None:
        pass

    def _select_image_file(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Images", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff"),
                ("JPEG", "*.jpg *.jpeg"),
                ("PNG", "*.png"),
                ("All files", "*.*")
            ],
            initialdir="."
        )
        
        if file_path:
            try:
                self.original_bgr_image = load_bgr_image(file_path)
                self.current_image_path = file_path
                self.update_image_display()
                self.stroke_plan = None
                self.edges_image = None
                self.use_simple_edges = False
                self.fill_mask = None
                self.fill_params = None
                if _HAS_CTK:
                    self.start_button.configure(state="disabled")
                    self.prepare_button.configure(text="Prepare Drawing")
                else:
                    self.start_button.config(state="disabled")
                    self.prepare_button.config(text="Prepare Drawing")
            except Exception as e:
                messagebox.showerror("Error", f"Could not load image:\n{str(e)}")

    def apply_topmost(self) -> None:
        self.root.wm_attributes("-topmost", bool(self.is_topmost_var.get()))

    def _pil_from_bgr_with_alpha(self, bgr_image: "cv2.Mat", alpha_percent: int) -> "Image.Image":
        rgba_image = convert_bgr_to_rgba_with_alpha(bgr_image, 100)
        gray = convert_bgr_to_gray(bgr_image)
        
        h, w = gray.shape
        border_pixels = np.concatenate([
            gray[0, :],
            gray[-1, :],
            gray[:, 0],
            gray[:, -1]
        ])
        
        background_color = np.median(border_pixels)
        
        if background_color > 200:
            threshold_value = max(180, background_color - 30)
        elif background_color > 150:
            threshold_value = max(140, background_color - 40)
        else:
            threshold_value = max(120, background_color - 50)
        
        alpha_scale = np.clip(alpha_percent, 0, 100) / 100.0
        
        alpha_channel = np.zeros(gray.shape, dtype=np.uint8)
        
        dark_mask = gray < threshold_value
        alpha_channel[dark_mask] = (255 * alpha_scale).astype(np.uint8)
        
        light_mask = gray >= threshold_value
        alpha_channel[light_mask] = 0
        
        rgba_image[:, :, 3] = alpha_channel
        pil_image = Image.fromarray(rgba_image, "RGBA")
        return pil_image

    def _get_scaled_image(self) -> "cv2.Mat":
        if self.original_bgr_image is None:
            raise ValueError("No image loaded")
        percent = int(self.scale_percent_var.get())
        return scale_bgr_image(self.original_bgr_image, percent)

    def update_image_display(self) -> None:
        if self.original_bgr_image is None:
            return
        scaled_bgr = self._get_scaled_image()
        alpha_percent = int(self.alpha_percent_var.get())
        overlay = self._pil_from_bgr_with_alpha(scaled_bgr, alpha_percent)
        
        max_display_width = 800
        max_display_height = 600
        img_width = overlay.width
        img_height = overlay.height
        
        if img_width > max_display_width or img_height > max_display_height:
            scale_w = max_display_width / img_width
            scale_h = max_display_height / img_height
            scale = min(scale_w, scale_h)
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            overlay = overlay.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        self.photo_image_ref = ImageTk.PhotoImage(overlay)
        
        if self.image_canvas_image_id is not None:
            self.image_canvas.delete(self.image_canvas_image_id)
        
        self.image_canvas_image_id = self.image_canvas.create_image(
            0, 0, anchor="nw", image=self.photo_image_ref
        )
        self.image_canvas.config(
            width=self.photo_image_ref.width(),
            height=self.photo_image_ref.height()
        )

    def on_drag_start(self, event: "tk.Event") -> None:
        self.drag_offset_x = int(event.x)
        self.drag_offset_y = int(event.y)

    def on_drag_motion(self, event: "tk.Event") -> None:
        delta_x = int(event.x_root - self.root.winfo_rootx() - self.drag_offset_x)
        delta_y = int(event.y_root - self.root.winfo_rooty() - self.drag_offset_y)
        new_x = self.root.winfo_x() + delta_x
        new_y = self.root.winfo_y() + delta_y
        self.root.geometry(f"+{new_x}+{new_y}")
    
    def prepare_drawing(self) -> None:
        if self.original_bgr_image is None:
            messagebox.showwarning("No Image", "Please select an image first.")
            return
        
        scaled_bgr = self._get_scaled_image()
        mode = self.mode_var.get()
        detail = max(1, min(10, int(self.detail_var.get())))
        
        self.fill_mask = None
        self.fill_params = None
        self.stroke_plan = None
        self.use_simple_edges = False
        self.edges_image = None
        

        gray = convert_bgr_to_gray(scaled_bgr)
        


        if detail >= 9:


            _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
            
            if mode == "fill":

                self.fill_mask = binary

                self.fill_params = (1,)
            
            elif mode == "outline":



                skeleton = thinning(binary)
                self.fill_mask = skeleton
                self.fill_params = (1,)
                self.use_simple_edges = True
                self.edges_image = skeleton
            
            elif mode == "both":

                skeleton = thinning(binary)
                self.fill_mask = binary
                self.fill_params = (1,)
                self.use_simple_edges = True
                self.edges_image = skeleton
        
        else:

            if mode == "outline":

                self.edges_image = compute_canny_edges(gray)
                self.use_simple_edges = True
            else:

                threshold_offset = 8
                full_plan = build_stroke_plan(scaled_bgr, detail, threshold_offset=threshold_offset)
                
                if mode == "fill":

                    self.stroke_plan = [stroke for stroke in full_plan if stroke.stroke_type != "outline"]
                elif mode == "both":

                    self.stroke_plan = full_plan
                
                if mode in ("fill", "both"):

                    self.fill_mask = self._generate_fill_mask(gray, 0)
                    if self.fill_mask is not None:

                        spacing = max(2, 5 - int(detail / 2))
                        self.fill_params = (spacing,)


        if _HAS_CTK:
            self.prepare_button.configure(text="Re-prepare Drawing")
        else:
            self.prepare_button.config(text="Re-prepare Drawing")


        if not self.stroke_plan and self.fill_mask is None and not self.use_simple_edges:
            if _HAS_CTK:
                self.start_button.configure(state="disabled")
                self.preview_label.configure(image="", text="No strokes found. Adjust the detail.")
            else:
                self.start_button.config(state="disabled")
                self.preview_label.config(image="", text="No strokes found. Adjust the detail.")
            return


        if self.use_simple_edges and self.edges_image is not None:

            preview = Image.new("RGB", (self.edges_image.shape[1], self.edges_image.shape[0]), "black")
            edges_rgb = np.stack([self.edges_image, self.edges_image, self.edges_image], axis=2)
            preview = Image.fromarray(edges_rgb, mode="RGB")
        elif self.fill_mask is not None:

            preview = Image.fromarray(255 - self.fill_mask, mode="L").convert("RGB")
        else:

            preview = self._render_preview(
                self.stroke_plan or [],
                scaled_bgr.shape[1],
                scaled_bgr.shape[0],
                fill_mask=self.fill_mask if mode in ("fill", "both") else None,
            )
        
        self.preview_image_ref = ImageTk.PhotoImage(preview)
        if _HAS_CTK:
            self.preview_label.configure(image=self.preview_image_ref, text="")
            self.preview_label.image = self.preview_image_ref
            self.start_button.configure(state="normal")
        else:
            self.preview_label.config(image=self.preview_image_ref, text="")
            self.preview_label.image = self.preview_image_ref
            self.start_button.config(state="normal")

    def _generate_fill_mask(self, gray_image: "np.ndarray", threshold_value: int) -> Optional["np.ndarray"]:
        detail_val = self.detail_var.get()
        

        if detail_val >= 9:


            _, mask = cv2.threshold(gray_image, 180, 255, cv2.THRESH_BINARY_INV)
            


            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            min_area = 2
            clean_mask = np.zeros_like(mask)
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] >= min_area:
                    clean_mask[labels == i] = 255
            
            if cv2.countNonZero(clean_mask) == 0:
                return None
            return clean_mask
        

        from services.image_processing import apply_floyd_steinberg_dithering
        

        kernel_sharpen = np.array([[-1, -1, -1], 
                                   [-1,  9, -1], 
                                   [-1, -1, -1]])
        sharpened = cv2.filter2D(gray_image, -1, kernel_sharpen)
        

        if threshold_value == 0:

            thresh_val, mask = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            _, mask = cv2.threshold(sharpened, int(thresh_val * 0.9), 255, cv2.THRESH_BINARY_INV)
        else:
            _, mask = cv2.threshold(sharpened, int(threshold_value), 255, cv2.THRESH_BINARY_INV)
        

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        min_area = 2
        clean_mask = np.zeros_like(mask)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                clean_mask[labels == i] = 255
        
        if cv2.countNonZero(clean_mask) == 0:
            return None
        
        return clean_mask

    def _render_preview(
        self,
        strokes: list[Stroke],
        width: int,
        height: int,
        fill_mask: Optional["np.ndarray"] = None,
    ) -> "Image.Image":
        preview = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(preview)


        if fill_mask is not None:
            spacing = 2
            for y in range(0, height, spacing):
                row = fill_mask[y]
                x = 0
                while x < width:
                    if row[x] == 255:
                        start = x
                        while x < width and row[x] == 255:
                            x += 1
                        end = x - 1
                        if end > start:
                            draw.line([(start, y), (end, y)], fill="black", width=1)
                    else:
                        x += 1


        for stroke in strokes:
            if len(stroke.points) < 2:
                continue
            base_width = 1.5 if stroke.stroke_type == "outline" else 0.8
            line_width = max(1, int(round(base_width * stroke.weight)))
            draw.line(stroke.points, fill="black", width=line_width, joint="curve")
        
        return preview

    def start_drawing(self) -> None:
        if self.drawing_thread and self.drawing_thread.is_alive():
            return

        if (
            not self.stroke_plan
            and not self.use_simple_edges
            and self.fill_mask is None
        ):
            return

        self.cancel_event.clear()
        self._set_controls_enabled(False)
        if _HAS_CTK:
            self.cancel_button.configure(state="normal")
        else:
            self.cancel_button.config(state="normal")

        self._show_countdown_and_start()

    def _show_countdown_and_start(self) -> None:
        if self.countdown_label is None:
            if _HAS_CTK:
                countdown_frame = ctk.CTkToplevel(self.root)
                countdown_frame.title("Starting drawing...")
                countdown_frame.attributes("-topmost", True)
                countdown_frame.geometry("300x150")
                countdown_frame.resizable(False, False)

                center_x = self.root.winfo_x() + self.root.winfo_width() // 2 - 150
                center_y = self.root.winfo_y() + self.root.winfo_height() // 2 - 75
                countdown_frame.geometry(f"300x150+{center_x}+{center_y}")

                self.countdown_label = ctk.CTkLabel(
                    countdown_frame,
                    text="",
                    font=ctk.CTkFont(size=24, weight="bold"),
                    text_color="red"
                )
                self.countdown_label.pack(expand=True)

                cancel_countdown_button = ctk.CTkButton(
                    countdown_frame,
                    text="Cancel",
                    command=self._cancel_countdown
                )
                cancel_countdown_button.pack(pady=10)
            else:
                countdown_frame = tk.Toplevel(self.root)
                countdown_frame.title("Starting drawing...")
                countdown_frame.attributes("-topmost", True)
                countdown_frame.geometry("300x150")
                countdown_frame.resizable(False, False)

                center_x = self.root.winfo_x() + self.root.winfo_width() // 2 - 150
                center_y = self.root.winfo_y() + self.root.winfo_height() // 2 - 75
                countdown_frame.geometry(f"300x150+{center_x}+{center_y}")

                self.countdown_label = tk.Label(
                    countdown_frame,
                    text="",
                    font=("Arial", 24, "bold"),
                    fg="red"
                )
                self.countdown_label.pack(expand=True)

                cancel_countdown_button = tk.Button(
                    countdown_frame,
                    text="Cancel",
                    command=self._cancel_countdown
                )
                cancel_countdown_button.pack(pady=10)

        self._run_countdown(5)

    def _cancel_countdown(self) -> None:
        if self.countdown_label is not None:
            countdown_window = self.countdown_label.winfo_toplevel()
            countdown_window.destroy()
            self.countdown_label = None
        self.cancel_event.set()
        self._on_drawing_finished()

    def _run_countdown(self, remaining_seconds: int) -> None:
        if self.cancel_event.is_set():
            if self.countdown_label is not None:
                countdown_window = self.countdown_label.winfo_toplevel()
                countdown_window.destroy()
                self.countdown_label = None
            return

        if remaining_seconds > 0:
            if self.countdown_label is not None:
                if _HAS_CTK:
                    self.countdown_label.configure(text=f"{remaining_seconds}")
                else:
                    self.countdown_label.config(text=f"{remaining_seconds}")
                countdown_window = self.countdown_label.winfo_toplevel()
                countdown_window.update()
            next_seconds = remaining_seconds - 1
            self.root.after(1000, lambda s=next_seconds: self._run_countdown(s))
        else:
            if self.countdown_label is not None:
                if _HAS_CTK:
                    self.countdown_label.configure(text="¡Comenzando!")
                else:
                    self.countdown_label.config(text="¡Comenzando!")
                countdown_window = self.countdown_label.winfo_toplevel()
                countdown_window.update()
            self.root.after(500, self._begin_drawing_after_countdown)

    def _begin_drawing_after_countdown(self) -> None:
        if self.countdown_label is not None:
            countdown_window = self.countdown_label.winfo_toplevel()
            countdown_window.destroy()
            self.countdown_label = None

        if self.cancel_event.is_set():
            self._on_drawing_finished()
            return

        origin_x = int(self.image_canvas.winfo_rootx())
        origin_y = int(self.image_canvas.winfo_rooty())

        self._start_hotkey_listener()
        
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        self.root.geometry(f"+{screen_width + 100}+{screen_height + 100}")
        time.sleep(0.2)

        if self.use_simple_edges and self.edges_image is not None:
            self.drawing_thread = threading.Thread(
                target=self._run_simple_edges_drawing,
                args=(origin_x, origin_y),
                daemon=True
            )
        else:
            self.drawing_thread = threading.Thread(
                target=self._run_complex_drawing,
                args=(origin_x, origin_y),
                daemon=True
            )
        self.drawing_thread.start()

    def _run_simple_edges_drawing(self, origin_x: int, origin_y: int) -> None:
        try:
            if self.edges_image is None:
                return
            
            draw_edges_fast(self.edges_image, origin_x, origin_y, self.cancel_event)
        finally:
            self.root.after(0, self._on_drawing_finished)

    def _run_complex_drawing(self, origin_x: int, origin_y: int) -> None:
        try:
            if (
                self.fill_mask is not None
                and self.fill_params is not None
                and self.mode_var.get() in ("fill", "both")
            ):
                (spacing,) = self.fill_params
                draw_scanline_fill_fast(
                    self.fill_mask,
                    origin_x,
                    origin_y,
                    self.cancel_event,
                    line_spacing_px=spacing,
                )
            if self.stroke_plan:
                stroke_paths = []
                for stroke in self.stroke_plan:
                    if len(stroke.points) >= 2:
                        path = [(int(x), int(y)) for x, y in stroke.points]
                        stroke_paths.append(path)
                
                if stroke_paths:
                    draw_stroke_paths_fast(stroke_paths, origin_x, origin_y, self.cancel_event)
        finally:
            self.root.after(0, self._on_drawing_finished)
    
    def _draw_fill_with_updates(
        self,
        fill_mask: "np.ndarray",
        origin_x: int,
        origin_y: int,
        spacing: int,
    ) -> None:
        height, width = fill_mask.shape
        memory_cell_size = max(1, spacing)
        memory_height = (height + memory_cell_size - 1) // memory_cell_size
        memory_width = (width + memory_cell_size - 1) // memory_cell_size
        drawn_areas = np.zeros((memory_height, memory_width), dtype=np.uint8)
        
        lines_drawn = 0
        update_interval = 3
        
        for y in range(0, height, spacing):
            if self.cancel_event.is_set():
                break
            
            row = fill_mask[y]
            memory_y = y // memory_cell_size
            
            x = 0
            while x < width:
                if self.cancel_event.is_set():
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
                        

                        if lines_drawn % update_interval == 0:
                            try:
                                self.root.after(0, self._update_ui)
                            except Exception:
                                pass
                else:
                    x += 1
        

        if spacing > 1 and not self.cancel_event.is_set():
            for y in range(0, height, 1):
                if self.cancel_event.is_set():
                    break
                if y % spacing == 0:
                    continue
                
                row = fill_mask[y]
                memory_y = y // memory_cell_size
                
                x = 0
                while x < width:
                    if self.cancel_event.is_set():
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
                            
                            lines_drawn += 1
                            if lines_drawn % update_interval == 0:
                                try:
                                    self.root.after(0, self._update_ui)
                                except Exception:
                                    pass
                    else:
                        x += 1
    
    def _update_ui(self) -> None:
        try:
            self.root.update_idletasks()
        except Exception:
            pass

    def _draw_strokes(self, strokes: list[Stroke], origin_x: int, origin_y: int) -> None:
        strokes_drawn = 0
        update_interval = 5
        
        for stroke in strokes:
            if self.cancel_event.is_set():
                break
            repetitions = max(1, min(4, int(round(stroke.weight * 2))))
            for rep in range(repetitions):
                if self.cancel_event.is_set():
                    break
                offset = (rep - (repetitions - 1) / 2.0) * 0.4
                offset_path = [(x, y + offset) for x, y in stroke.points]
                draw_stroke_paths_with_pyautogui([offset_path], origin_x, origin_y, self.cancel_event)
                
                strokes_drawn += 1

                if strokes_drawn % update_interval == 0:
                    try:
                        self.root.after(0, self._update_ui)
                    except Exception:
                        pass

    def _start_hotkey_listener(self) -> None:
        try:
            self.hotkey_listener = GlobalHotkeyListener(self.cancel_event, hotkey="<ctrl>+c")
            self.hotkey_listener.start_listening()
            self._show_cancel_hint()
        except ImportError:
            self._show_cancel_hint(show_hotkey=False)
    
    def _show_cancel_hint(self, show_hotkey: bool = True) -> None:
        if self.cancel_hint_window is not None:
            return
        
        try:
            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()
        except:
            screen_width = 1920
            screen_height = 1080
        
        hint_x = screen_width - 270
        hint_y = 20
        
        self.cancel_hint_window = tk.Toplevel(self.root)
        self.cancel_hint_window.title("Cancel Drawing")
        self.cancel_hint_window.attributes("-topmost", True)
        self.cancel_hint_window.geometry(f"250x100+{hint_x}+{hint_y}")
        self.cancel_hint_window.resizable(False, False)
        self.cancel_hint_window.overrideredirect(True)
        self.cancel_hint_window.wm_attributes("-topmost", True)
        self.cancel_hint_window.transient(self.root)
        
        hint_frame = tk.Frame(self.cancel_hint_window, bg="yellow", bd=2, relief="solid")
        hint_frame.pack(fill="both", expand=True)
        
        if show_hotkey:
            hint_text = "Press Ctrl+C to\ncancel the drawing"
        else:
            hint_text = "Use the Cancel button\nto stop the drawing"
        
        hint_label = tk.Label(
            hint_frame,
            text=hint_text,
            font=("Arial", 10, "bold"),
            bg="yellow",
            fg="black",
            justify="center"
        )
        hint_label.pack(expand=True)
        
        close_hint_button = tk.Button(
            hint_frame,
            text="Close",
            command=self._close_cancel_hint,
            bg="lightgray"
        )
        close_hint_button.pack(pady=5)
        
        self.cancel_hint_window.update_idletasks()
        self.cancel_hint_window.update()
    
    def _close_cancel_hint(self) -> None:
        if self.cancel_hint_window is not None:
            self.cancel_hint_window.destroy()
            self.cancel_hint_window = None
    
    def _stop_hotkey_listener(self) -> None:
        if self.hotkey_listener is not None:
            try:
                self.hotkey_listener.stop_listening()
            except Exception:
                pass
            self.hotkey_listener = None
        self._close_cancel_hint()

    def _on_drawing_finished(self) -> None:
        self._stop_hotkey_listener()
        self.root.deiconify()
        self.apply_topmost()
        self._set_controls_enabled(True)
        if _HAS_CTK:
            self.cancel_button.configure(state="disabled")
        else:
            self.cancel_button.config(state="disabled")
        
        messagebox.showinfo("Drawing Complete", "The drawing has been completed successfully.\nThe program will now close.")
        self.root.quit()
        self.root.destroy()

    def cancel_drawing(self) -> None:
        if self.countdown_label is not None:
            self._cancel_countdown()
        if self.drawing_thread and self.drawing_thread.is_alive():
            self.cancel_event.set()
        self._stop_hotkey_listener()