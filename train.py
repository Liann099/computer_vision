from ultralytics import YOLO

# load model awal
model = YOLO('yolov8n.pt')

# training
model.train(
    data='dataset/data.yaml',
    epochs=50,
    imgsz=640
)