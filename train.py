from ultralytics import YOLO
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# Function to train YOLOv5 with minimal time
def train_yolo():
    model = YOLO(
        "yolov5s.pt"
    )  # Load the YOLOv5 small model (use "yolov5s" for faster training)
    model.train(
        data=r"data\dataset.yaml", 
        epochs=25, 
        imgsz=416,  
        batch=16,  # Increased batch size for faster training
        cache=False,  # Avoided caching to save time
        device="cuda",  # Using GPU
        half=True, 
        workers=4,  
    )


if __name__ == "__main__":
    train_yolo()
