from ultralytics import YOLO

# Load the YOLOv8 model (use yolov8n.pt for a small model or yolov8x.pt for high accuracy)
model = YOLO("yolo11x.pt")  # You can replace with 'yolov8s.pt', 'yolov8m.pt', etc.

# Perform inference on a video
results = model.predict(
    source="/home/javier/pytorch/RoadVision3D/roadvision3d/data/delicias1_x1_30fps.mp4",
    conf=0.5,
    iou=0.4,
    device="1",  # Use GPU
    save=True
)
# The output video with detections will be saved in the "runs/detect" directory.
print("Video inference completed! Check the 'runs/detect' folder.")