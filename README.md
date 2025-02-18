Weapon Detection System using YOLOv5

This project implements a real-time weapon detection system using YOLOv5, designed to identify objects such as knives and handguns in images. The goal is to enhance surveillance and security by detecting potential threats in real-time.

Despite initial challenges with confidence scores (around 0.6) and misclassifications, this project serves as a foundation for future improvements in model accuracy, dataset quality, and real-time deployment.

Technologies Used

YOLOv5 – Object detection model for real-time weapon recognition

Python 3.11 – The primary programming language

OpenCV – For image processing and visualization

PyTorch – For training and inference

CUDA (optional) – For GPU acceleration


Project Features

Weapon Detection – Identifies knives and handguns in images

Bounding Box Visualization – Draws detection boxes on images

Alert System – Triggers a console alert if a weapon is detected

Custom Dataset Training – Uses a curated dataset for improved accuracy


Installation & Usage

1. Clone the repository


2. Install dependencies from the requirements file


3. Download the dataset and place it in the data/ directory


4. Train the model using the training script


5. Run detection on images



Challenges & Issues

Low Confidence Scores (around 0.6) due to a small dataset, insufficient training epochs, or potential class labeling inconsistencies

False Positives and False Negatives, leading to misclassifications or missed detections

Running on CPU results in slow inference, requiring GPU support for real-time processing


Future Improvements

Increase dataset size to improve model generalization

Fine-tune YOLOv5 hyperparameters such as learning rate, batch size, and epochs

Add video stream support for real-time weapon detection in CCTV footage

Deploy as a web application using Flask or FastAPI


#**Conclusion**

This project is an ongoing effort to develop an effective weapon detection system using YOLOv5. While the current model struggles with accuracy, improvements in data, training, and deployment can make it more reliable for real-world use.