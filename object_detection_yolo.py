from yolo import detect_objects_in_images
import os
import cv2
import numpy as np
from utils import load_image, save_image

def load_images_and_process(folder_path):
    labelsPath = "yolo-coco/coco.names"
    LABELS = open(labelsPath).read().strip().split("\n")
    
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
		dtype="uint8")

    weightsPath = "yolo-coco/yolov3.weights"
    configPath = "yolo-coco/yolov3.cfg"
    
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    # Get a list of all image files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]

    for image_file in image_files:
        # Build full path to the image
        image_path = os.path.join(folder_path, image_file)
        
        # Load the image using OpenCV
        image = load_image(image_path)
        orig = image.copy()
        
        if image is None:
            print(f"Failed to load image: {image_path}")
            continue
        
        print(f"Processing image: {image_file}")
        
        # Call your detect objects method to process the image
        idxs, boxes, confidences, classIDs = detect_objects_in_images(net, image)

        if len(idxs) > 0:
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                	0.5, color, 2)
	
	
        folder_name = 'results/' + image_file.split(".")[0]
        save_image(folder_name, '1_original_image.jpg', orig)
        save_image(folder_name, '1_objects_detected_image.jpg', image)

# Folder containing the images
image_folder = 'images'

# Load images and process them one by one
load_images_and_process(image_folder)