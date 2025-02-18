from ultralytics import YOLO
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# Function to train YOLOv5 with minimal time
def train_yolo():
    model = YOLO(
        "yolov5s.pt"
    )  # Load the YOLOv5 small model (use "yolov5s" for faster training)
    model.train(
        data=r"data\dataset.yaml",  # Correct path to dataset.yaml using raw string
        epochs=25,  # Fewer epochs for faster training
        imgsz=416,  # Reduced image size for faster training (can try 416 for balance)
        batch=16,  # Increased batch size for faster training
        cache=False,  # Avoid caching to save time
        device="cuda",  # Using GPU
        half=True,  # Mixed precision to speed up training and reduce memory usage
        workers=4,  # Increased number of workers to load data faster
    )


if __name__ == "__main__":
    train_yolo()
