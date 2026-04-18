import cv2
import os
import time
import customtkinter as ctk
from PIL import Image
from ultralytics import YOLO
from datetime import datetime

# Initialize the UI appearance
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# Folder to save screenshots
SCREENSHOT_DIR = "detections"
os.makedirs(SCREENSHOT_DIR, exist_ok=True)


class ComputerVisionApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Live Object Detection AI")
        self.geometry("1200x750")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Load model
        self.model = YOLO(r'C:\Users\Ivan\runs\detect\train3\weights\best.pt')

        # Screenshot state
        self.screenshots = []           # list of (PIL.Image, label_str, timestamp_str)
        self.last_screenshot_time = 0
        self.screenshot_cooldown = 2.0  # seconds between auto-screenshots
        self.gallery_thumb_size = (160, 110)
        self.selected_screenshot = None

        self.setup_ui()

        self.cap = cv2.VideoCapture(0)
        self.is_running = False

    # ------------------------------------------------------------------ UI --
    def setup_ui(self):
        # ── Left control panel ──────────────────────────────────────────────
        self.control_frame = ctk.CTkFrame(self, width=200, corner_radius=10)
        self.control_frame.pack(side="left", fill="y", padx=10, pady=10)
        self.control_frame.pack_propagate(False)

        ctk.CTkLabel(self.control_frame, text="AI Vision",
                     font=("Arial", 24, "bold")).pack(pady=(20, 30))

        self.start_btn = ctk.CTkButton(self.control_frame, text="▶  Start Camera",
                                       command=self.start_video)
        self.start_btn.pack(pady=8, padx=20)

        self.stop_btn = ctk.CTkButton(self.control_frame, text="⏹  Stop Camera",
                                      command=self.stop_video,
                                      fg_color="red", hover_color="darkred")
        self.stop_btn.pack(pady=8, padx=20)

        # Screenshot controls
        ctk.CTkLabel(self.control_frame, text="Screenshot Settings",
                     font=("Arial", 13, "bold")).pack(pady=(40, 6))

        ctk.CTkLabel(self.control_frame, text="Cooldown (seconds):",
                     font=("Arial", 11)).pack()
        self.cooldown_slider = ctk.CTkSlider(self.control_frame, from_=0.5, to=10,
                                             number_of_steps=19,
                                             command=self._on_cooldown_change)
        self.cooldown_slider.set(self.screenshot_cooldown)
        self.cooldown_slider.pack(padx=20, pady=4)
        self.cooldown_label = ctk.CTkLabel(self.control_frame,
                                           text=f"{self.screenshot_cooldown:.1f}s",
                                           font=("Arial", 11))
        self.cooldown_label.pack()

        self.manual_btn = ctk.CTkButton(self.control_frame, text="📸  Manual Screenshot",
                                         command=self.manual_screenshot,
                                         fg_color="#1f6aa5")
        self.manual_btn.pack(pady=12, padx=20)

        self.clear_btn = ctk.CTkButton(self.control_frame, text="🗑  Clear Gallery",
                                       command=self.clear_gallery,
                                       fg_color="#555", hover_color="#333")
        self.clear_btn.pack(pady=4, padx=20)

        # Stats
        self.stats_label = ctk.CTkLabel(self.control_frame, text="Screenshots: 0",
                                         font=("Arial", 11), text_color="gray")
        self.stats_label.pack(pady=(20, 4))

        self.last_label = ctk.CTkLabel(self.control_frame, text="Last: —",
                                        font=("Arial", 10), text_color="gray",
                                        wraplength=170)
        self.last_label.pack(padx=8)

        # Future
        ctk.CTkLabel(self.control_frame, text="Future Optimizations:",
                     font=("Arial", 13, "bold")).pack(pady=(30, 6))
        ctk.CTkButton(self.control_frame, text="Train New Data\n(Coming Soon)",
                      state="disabled", fg_color="gray").pack(pady=6, padx=20)

        # ── Right: tabview ──────────────────────────────────────────────────
        self.tabview = ctk.CTkTabview(self, corner_radius=10)
        self.tabview.pack(side="right", fill="both", expand=True,
                          padx=(0, 10), pady=10)

        self.tab_camera = self.tabview.add("📷  Live Camera")
        self.tab_gallery = self.tabview.add("🖼  Gallery")

        # Camera tab
        self.video_label = ctk.CTkLabel(self.tab_camera,
                                         text="Click 'Start Camera' to begin",
                                         font=("Arial", 18))
        self.video_label.pack(fill="both", expand=True, padx=10, pady=10)

        # Gallery tab
        self._build_gallery_tab()

    def _build_gallery_tab(self):
        """Create the scrollable gallery + detail panel."""
        self.gallery_frame = ctk.CTkFrame(self.tab_gallery, corner_radius=8)
        self.gallery_frame.pack(fill="both", expand=True, padx=6, pady=6)

        # Top: scrollable thumbnail grid
        self.scroll_frame = ctk.CTkScrollableFrame(self.gallery_frame,
                                                    label_text="Detected Objects",
                                                    corner_radius=6)
        self.scroll_frame.pack(fill="both", expand=True, padx=4, pady=(4, 0))

        # Bottom: detail view
        self.detail_frame = ctk.CTkFrame(self.gallery_frame, height=180,
                                          corner_radius=6)
        self.detail_frame.pack(fill="x", padx=4, pady=4)
        self.detail_frame.pack_propagate(False)

        self.detail_image_label = ctk.CTkLabel(self.detail_frame, text="",
                                                width=240)
        self.detail_image_label.pack(side="left", padx=10, pady=8)

        detail_text_frame = ctk.CTkFrame(self.detail_frame, fg_color="transparent")
        detail_text_frame.pack(side="left", fill="both", expand=True, pady=8)

        self.detail_title = ctk.CTkLabel(detail_text_frame, text="Select a screenshot",
                                          font=("Arial", 15, "bold"), anchor="w")
        self.detail_title.pack(anchor="w", padx=6)

        self.detail_time = ctk.CTkLabel(detail_text_frame, text="",
                                         font=("Arial", 11), text_color="gray",
                                         anchor="w")
        self.detail_time.pack(anchor="w", padx=6, pady=2)

        self.detail_objects = ctk.CTkLabel(detail_text_frame, text="",
                                            font=("Arial", 11), anchor="w",
                                            wraplength=400, justify="left")
        self.detail_objects.pack(anchor="w", padx=6)

        self.save_btn = ctk.CTkButton(detail_text_frame, text="💾  Save to Disk",
                                       command=self.save_selected, width=140)
        self.save_btn.pack(anchor="w", padx=6, pady=8)

        self.thumb_widgets = []  # keep references to avoid GC

    # --------------------------------------------------------- video loop ---
    def start_video(self):
        if not self.is_running:
            # Re-open the camera if it was released on stop
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(0)
            self.is_running = True
            self.update_frame()

    def stop_video(self):
        self.is_running = False
        self.cap.release()  # Release so it can be cleanly reopened later
        self.video_label.configure(image=None, text="Click 'Start Camera' to begin")

    def update_frame(self):
        if not self.is_running:
            return

        ret, frame = self.cap.read()
        if ret:
            results = self.model(frame, stream=True, verbose=False)
            annotated = frame.copy()
            detected_labels = []

            for r in results:
                annotated = r.plot()
                if r.boxes is not None and len(r.boxes):
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        label = self.model.names.get(cls_id, str(cls_id))
                        conf = float(box.conf[0])
                        detected_labels.append(f"{label} ({conf:.0%})")

            # Auto-screenshot when objects are detected
            now = time.time()
            if detected_labels and (now - self.last_screenshot_time) >= self.screenshot_cooldown:
                self.last_screenshot_time = now
                self._take_screenshot(annotated, detected_labels)

            # Display
            frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            ctk_image = ctk.CTkImage(light_image=pil_image, dark_image=pil_image,
                                      size=(pil_image.width, pil_image.height))
            self.video_label.configure(image=ctk_image, text="")

        self.after(10, self.update_frame)

    # --------------------------------------------------- screenshot logic ---
    def _take_screenshot(self, bgr_frame, labels: list):
        """Store screenshot and refresh gallery."""
        pil_img = Image.fromarray(cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB))
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        label_str = ", ".join(labels)
        self.screenshots.append((pil_img, label_str, ts))
        self._refresh_gallery()
        self._update_stats(label_str)

    def manual_screenshot(self):
        """Manually capture current frame regardless of detections."""
        ret, frame = self.cap.read()
        if ret:
            results = self.model(frame, stream=True, verbose=False)
            annotated = frame.copy()
            labels = []
            for r in results:
                annotated = r.plot()
                if r.boxes is not None and len(r.boxes):
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        label = self.model.names.get(cls_id, str(cls_id))
                        conf = float(box.conf[0])
                        labels.append(f"{label} ({conf:.0%})")
            if not labels:
                labels = ["Manual capture"]
            self._take_screenshot(annotated, labels)

    def clear_gallery(self):
        self.screenshots.clear()
        self.selected_screenshot = None
        self._refresh_gallery()
        self.detail_title.configure(text="Select a screenshot")
        self.detail_time.configure(text="")
        self.detail_objects.configure(text="")
        self.detail_image_label.configure(image=None, text="")
        self.stats_label.configure(text="Screenshots: 0")
        self.last_label.configure(text="Last: —")

    def save_selected(self):
        """Save the currently selected screenshot to disk."""
        if self.selected_screenshot is None:
            return
        pil_img, label_str, ts = self.selected_screenshot
        safe_ts = ts.replace(":", "-").replace(" ", "_")
        filename = os.path.join(SCREENSHOT_DIR, f"detection_{safe_ts}.jpg")
        pil_img.save(filename, "JPEG", quality=92)
        self.detail_title.configure(text=f"✅ Saved: {os.path.basename(filename)}")

    # --------------------------------------------------- gallery refresh ---
    def _refresh_gallery(self):
        """Rebuild thumbnail grid."""
        # Clear old widgets
        for w in self.thumb_widgets:
            w.destroy()
        self.thumb_widgets.clear()

        cols = 4
        for idx, (pil_img, label_str, ts) in enumerate(reversed(self.screenshots)):
            real_idx = len(self.screenshots) - 1 - idx  # newest first
            thumb = pil_img.copy()
            thumb.thumbnail(self.gallery_thumb_size, Image.LANCZOS)
            ctk_thumb = ctk.CTkImage(light_image=thumb, dark_image=thumb,
                                      size=(thumb.width, thumb.height))

            cell = ctk.CTkFrame(self.scroll_frame, corner_radius=6,
                                 border_width=1, border_color="#333")
            cell.grid(row=idx // cols, column=idx % cols, padx=6, pady=6, sticky="nsew")

            img_btn = ctk.CTkButton(cell, image=ctk_thumb, text="",
                                     width=thumb.width, height=thumb.height,
                                     fg_color="transparent", hover_color="#2a2a2a",
                                     command=lambda i=real_idx: self._select_screenshot(i))
            img_btn.pack(padx=4, pady=(4, 0))

            # Truncate label for thumb
            short_label = label_str if len(label_str) <= 22 else label_str[:20] + "…"
            ctk.CTkLabel(cell, text=short_label, font=("Arial", 9),
                          text_color="#aaa").pack(pady=(2, 4))

            self.thumb_widgets.extend([cell])

        # configure column weights
        for c in range(cols):
            self.scroll_frame.grid_columnconfigure(c, weight=1)

    def _select_screenshot(self, idx: int):
        self.selected_screenshot = self.screenshots[idx]
        pil_img, label_str, ts = self.selected_screenshot

        # Large preview in detail panel
        preview = pil_img.copy()
        preview.thumbnail((240, 160), Image.LANCZOS)
        ctk_preview = ctk.CTkImage(light_image=preview, dark_image=preview,
                                    size=(preview.width, preview.height))
        self.detail_image_label.configure(image=ctk_preview, text="")

        self.detail_title.configure(text=f"Screenshot #{idx + 1}")
        self.detail_time.configure(text=f"🕐 {ts}")
        self.detail_objects.configure(text=f"🔍 Detected: {label_str}")

    def _update_stats(self, last_label: str):
        self.stats_label.configure(text=f"Screenshots: {len(self.screenshots)}")
        short = last_label if len(last_label) <= 30 else last_label[:28] + "…"
        self.last_label.configure(text=f"Last: {short}")

    def _on_cooldown_change(self, value):
        self.screenshot_cooldown = round(float(value), 1)
        self.cooldown_label.configure(text=f"{self.screenshot_cooldown:.1f}s")

    # -------------------------------------------------------------- close ---
    def on_closing(self):
        self.is_running = False
        if self.cap.isOpened():
            self.cap.release()
        self.destroy()


if __name__ == "__main__":
    app = ComputerVisionApp()
    app.mainloop()