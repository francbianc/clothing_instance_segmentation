import numpy as np 
import cv2 
import matplotlib.pyplot as plt
import mtcnn
from mtcnn.mtcnn import MTCNN
import os 
import tqdm

path = '...' #@@@ OVERRIDE: Path of a folder containing images 

# Initialize Multi-task Cascaded Convolutional Neural Network
detector = MTCNN()
all_ids = sorted(os.listdir(path))

# Create an empty list to store the ids of images without a face
no_face_ids = []
for id in tqdm.tqdm(all_ids):
    img = cv2.imread(os.path.join(path, id))
    faces = detector.detect_faces(img)
    if not faces: 
        no_face_ids.append(id)

# Save in a txt file the ids of images with no face
with open(os.path.join(path, 'ids_no_face.txt'), 'w') as f:
    f.write('\n'.join(no_face_ids))

# Visuaize face bbox 
def visualize_face_detector(path, id):
    detector = MTCNN()
    img = cv2.imread(os.path.join(path, id))
    faces = detector.detect_faces(img)
    for result in faces:
        x, y, w, h = result['box']
        x1, y1 = x + w, y + h
        print(x, y, x1, y1)
        cv2.rectangle(img, (x, y), (x1, y1), (255, 0, 0), 2)
        plt.imshow(img);