from ultralytics import YOLO

# Load the base model
model = YOLO('yolov8n.pt')

# Train it on your custom dataset
model.train(data='path/to/your/dataset.yaml', epochs=50, imgsz=640)