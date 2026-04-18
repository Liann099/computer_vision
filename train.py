from ultralytics import YOLO

# load model awal
model = YOLO('yolov8n.pt')

# training
model.train(
    data='C:/Users/Ivan/Prasmul/Big Data for social media/computer_vision/dataset/data.yaml',
    epochs=50,
    imgsz=640
)