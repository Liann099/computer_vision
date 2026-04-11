import cv2
import customtkinter as ctk
from PIL import Image
from ultralytics import YOLO

# Initialize the UI appearance
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class ComputerVisionApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Window setup
        self.title("Live Object Detection AI")
        self.geometry("1000x700")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Load the pre-trained YOLOv8 Nano model (downloads automatically the first time)
        # To use your own trained model in the future, change this to: YOLO('runs/detect/train/weights/best.pt')
        self.model = YOLO(r'C:\Users\Ivan\runs\detect\train3\weights\best.pt') 

        # UI Layout
        self.setup_ui()

        # Camera setup (0 is usually the default laptop webcam)
        self.cap = cv2.VideoCapture(0)
        self.is_running = False

    def setup_ui(self):
        # Left Panel: Controls
        self.control_frame = ctk.CTkFrame(self, width=200, corner_radius=10)
        self.control_frame.pack(side="left", fill="y", padx=10, pady=10)

        self.title_label = ctk.CTkLabel(self.control_frame, text="AI Vision", font=("Arial", 24, "bold"))
        self.title_label.pack(pady=(20, 30))

        self.start_btn = ctk.CTkButton(self.control_frame, text="Start Camera", command=self.start_video)
        self.start_btn.pack(pady=10, padx=20)

        self.stop_btn = ctk.CTkButton(self.control_frame, text="Stop Camera", command=self.stop_video, fg_color="red", hover_color="darkred")
        self.stop_btn.pack(pady=10, padx=20)

        # Future-proofing: Placeholder for training UI
        self.train_label = ctk.CTkLabel(self.control_frame, text="Future Optimizations:", font=("Arial", 14, "bold"))
        self.train_label.pack(pady=(50, 10))
        
        self.train_btn = ctk.CTkButton(self.control_frame, text="Train New Data\n(Coming Soon)", state="disabled", fg_color="gray")
        self.train_btn.pack(pady=10, padx=20)

        # Right Panel: Video Feed
        self.video_frame = ctk.CTkFrame(self, corner_radius=10)
        self.video_frame.pack(side="right", fill="both", expand=True, padx=(0, 10), pady=10)

        self.video_label = ctk.CTkLabel(self.video_frame, text="Click 'Start Camera' to begin", font=("Arial", 18))
        self.video_label.pack(fill="both", expand=True, padx=10, pady=10)

    def start_video(self):
        if not self.is_running:
            self.is_running = True
            self.update_frame()

    def stop_video(self):
        self.is_running = False
        self.video_label.configure(image=None, text="Camera Stopped")

    def update_frame(self):
        if self.is_running:
            ret, frame = self.cap.read()
            if ret:
                # 1. Run YOLOv8 inference on the frame
                results = self.model(frame, stream=True, verbose=False)
                
                # 2. Draw bounding boxes on the frame
                for r in results:
                    frame = r.plot() # Ultralytics handles the drawing of boxes and labels
                
                # 3. Convert OpenCV image (BGR) to CustomTkinter image (RGB)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                
                # Resize image to fit the UI dynamically (optional, keeping it simple here)
                ctk_image = ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=(pil_image.width, pil_image.height))
                
                # 4. Update the UI
                self.video_label.configure(image=ctk_image, text="")
            
            # Schedule the next frame update in 10 milliseconds
            self.after(10, self.update_frame)

    def on_closing(self):
        self.is_running = False
        if self.cap.isOpened():
            self.cap.release()
        self.destroy()

if __name__ == "__main__":
    app = ComputerVisionApp()
    app.mainloop()