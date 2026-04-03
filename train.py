from ultralytics import YOLO

model = YOLO("yolov8n.pt") 
model.train(
    # Use the full name of the folder you extracted
    data="My First Project.v1-dataset.yolov8/data.yaml", 
    epochs=50,
    imgsz=640,
    batch=16
)