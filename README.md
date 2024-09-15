# Object Detection Using YOLO
A Python-based image detection project that processes images to detect objects using the YOLO (You Only Look Once) model. This script leverages OpenCV and the YOLOv3 model for detecting objects in images and saves the results in a specified directory.

## Features
- Detects objects in all images from a folder.
- Utilizes YOLOv3 model for object detection.
- Saves processed results with bounding boxes and labels.
- Leverages OpenCV and Darknet model for image processing.

## Set Up
### Clone the Repository

```bash
git clone https://github.com/your-username/YOLO-Object-Detection.git
cd YOLO-Object-Detection
```

### Set Up a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### Install Dependencies
Make sure you have the latest versions of the required libraries. Install them using the requirements.txt file:

```bash
pip install -r requirements.txt
```

## Usage
### Prepare Your Folders

Place all images you want to process in a folder named images.
Ensure there is a folder named results where the processed images will be saved.
Download the yolov3.weights from https://sourceforge.net/projects/yolov3.mirror/

### Run the Script

Execute the script to process images:

```bash
python image_detection_yolo.py
```

The script will:
- Load the YOLO model with configuration and weights.
- Read all images from the images folder.
- Process each image to detect objects using YOLOv3.
- Save the processed images with detected objects and bounding boxes to the results folder with the same name.

Ensure that you have images in the images folder, and you will find the results in the results folder.
