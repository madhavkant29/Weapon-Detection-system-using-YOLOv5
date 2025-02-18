***Weapon Detection System using YOLOv5***

This project implements a real-time weapon detection system using YOLOv5, designed to identify objects such as knives and handguns in images. The goal is to enhance surveillance and security by detecting potential threats in real-time.

Despite initial challenges with confidence scores (around 0.6) and misclassifications, this project serves as a foundation for future improvements in model accuracy, dataset quality, and real-time deployment.

**Technologies Used**

YOLOv5 – Object detection model for real-time weapon recognition

Python 3.11 – The primary programming language

OpenCV – For image processing and visualization

PyTorch – For training and inference

CUDA (optional) – For GPU acceleration


**Project Features**

Weapon Detection – Identifies knives and handguns in images

Bounding Box Visualization – Draws detection boxes on images

Alert System – Triggers a console alert if a weapon is detected

Custom Dataset Training – Uses a curated dataset for improved accuracy


**Installation & Usage**

1. Clone the repository


2. Install dependencies from the requirements file


3. Download this 
https://www.kaggle.com/datasets/raghavnanjappan/weapon-dataset-for-yolov5
dataset and place it in the data/ directory. (Has 4000 images)


4. Train the model using the training script


5. Run detection on images



**Challenges & Issues**

Low Confidence Scores (around 0.6) due to a limited training mainly insufficient training epochs(only 25)

False Positives and False Negatives, leading to misclassifications or missed detections

Running on CPU results in slow inference, requiring GPU support for real-time processing


**Future Improvements**

Extend training duration and experiment with larger batch sizes for better model convergence.

Fine-tune YOLOv5 hyperparameters, including learning rate and anchor sizes, to optimize detection accuracy.

Incorporate real-time video stream processing for CCTV surveillance applications.

Deploy as a web application using Flask or FastAPI for easy integration with security systems.


**Conclusion**

This project is an ongoing effort to build an effective weapon detection system using YOLOv5. While the current model faces accuracy challenges, improvements in dataset size, training strategies, and deployment can enhance its reliability for real-world security applications.
