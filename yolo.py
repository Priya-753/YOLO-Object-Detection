import numpy as np
import time
import cv2

def detect_objects_in_images(net, image, confidence = 0.5, threshold = 0.3):	
	(H, W) = image.shape[:2]
	
	ln = net.getLayerNames()
	ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
	
	blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	layerOutputs = net.forward(ln)
	end = time.time()
	
	print("[INFO] YOLO took {:.6f} seconds".format(end - start))

	boxes = []
	confidences = []
	classIDs = []

	for output in layerOutputs:
		for detection in output:
			scores = detection[5:]
			classID = np.argmax(scores)
			current_confidence = scores[classID]
			if current_confidence > confidence:

				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")
				
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))
				
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(current_confidence))
				classIDs.append(classID)


	idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence,
		threshold)

	return idxs, boxes, confidences, classIDs
