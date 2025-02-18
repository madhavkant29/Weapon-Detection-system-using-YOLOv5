# Weapon-Detection-system-using-YOLOv5
Failed

## **Project Overview**

This project aims to implement a weapon detection system using YOLOv5 to identify objects such as knives and handguns in images. The goal was to leverage deep learning and computer vision to detect weapons in surveillance or security camera footage.

Despite attempting to train and deploy the YOLOv5 model, the project faced challenges, such as consistently low confidence levels (~0.6) and incorrect classification (e.g., failing to detect weapons or misclassifying them).

## **Technologies Used**

- **YOLOv5**: A state-of-the-art object detection model built with PyTorch.
- **Python 3.11**: The primary programming language used for the project.
- **OpenCV**: For image processing and displaying results.
- **PyTorch**: For training and running the YOLOv5 model.
- **CUDA (optional)**: For GPU acceleration (though the current implementation uses CPU).

## **Project Features**

- **Weapon Detection**: The system attempts to detect knives and handguns in images.
- **Alert System**: Triggers an alert when a weapon is detected, printing "Weapon detected!" to the console.
- **Customizable Detection Classes**: Can be modified to detect other types of objects beyond weapons.

## **Known Issues**

- **Low Confidence**: The model often produces a consistent confidence of around 0.6, even when it successfully detects weapons. This issue could be due to several factors:
  - Incorrect or insufficient training data.
  - Model underfitting or poor generalization.
  - Incorrect class labeling or dataset mismatch.
- **False Negatives/False Positives**: Sometimes the model fails to detect weapons or misclassifies other objects as weapons.
- **No GPU Acceleration**: The project currently runs on CPU, which significantly slows down inference.

## **Data**
dataset should be in data folder that you have to create on your own. The link for the kaggle dataset: https://www.kaggle.com/datasets/raghavnanjappan/weapon-dataset-for-yolov5
## **Future Improvements**

- **Better Dataset**: Increase the quality and diversity of the training dataset, adding more variations of knives, handguns, and non-weapon objects.
- **Hyperparameter Tuning**: Experiment with different model architectures, learning rates, and batch sizes to improve detection performance.
- **GPU Usage**: Ensure that the model uses GPU acceleration to speed up training and inference.
- **Improved Postprocessing**: Investigate postprocessing techniques like non-maximum suppression (NMS) for better object localization.

## **Conclusion**

While this project aimed to develop a weapon detection system using YOLOv5, it did not achieve the desired performance. The confidence values were often stuck at 0.6, resulting in inaccurate detection results. Further work is needed in refining the dataset, improving training parameters, and ensuring proper model deployment for real-world use.

Might pick this up later someday
