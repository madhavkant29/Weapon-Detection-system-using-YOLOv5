import cv2
import torch
from yolov5 import YOLOv5


WEAPON_CLASS_IDS = [0, 1]  # 0: Knife, 1: Handgun
NO_WEAPON_CLASS_ID = -1  # For "No Weapon Detected"


# Function to trigger an alert (e.g., send an email)
def alert():
    print("Weapon detected!")  



def process_image(image_path, model):
    image = cv2.imread(image_path) 
    results = model.predict(image)  
    results.render()  # Draw bounding boxes on the image
    cv2.imshow("Weapon Detection", image)  # Display the image with bounding boxes

    # Extract the detections (class ids and their corresponding confidences)
    detections = results.pandas().xywh[0]

    weapon_detected = False  # Flag to track if a weapon is detected
    no_weapon_detected = True  # Assume no weapon detected initially

    # Loop through each detection
    for index, row in detections.iterrows():
        class_id = row["class"]  # Class ID of the detected object
        confidence = row["confidence"]  # Confidence of the detection

        # Debugging: print the detected class and confidence for all objects
        print(f"Detected Class: {class_id}, Confidence: {confidence}")

        if class_id in WEAPON_CLASS_IDS and confidence > 0.7:  # Adjust threshold to 0.7
            alert()
            weapon_detected = True  # Set the flag to True if a weapon is detected
            no_weapon_detected = False  # Mark that weapon has been detected

    if weapon_detected:
        print("Weapon detected!")  # Alert message
    elif no_weapon_detected:
        print("No weapon detected.")  # Message when no weapon is found
    else:
        print("No weapon detected.")  # Default message for no weapon detection

    cv2.waitKey(0)  # Wait until a key is pressed to close the image


def detect_from_path(input_type, input_path):
    device = "cpu"  # Force using CPU, regardless of CUDA availability
    model = YOLOv5(
        "yolov5s.pt", device=device
    )  

    if input_type == "image":
        process_image(input_path, model)
    else:
        print("Invalid input type. Please use 'image'.")

    cv2.destroyAllWindows()


detect_from_path(
    "image",
    r"C:\Users\madha\.vscode\python\surveillance anamoly gun and knife\data\dataset\images\train\4060.jpg",
) 
