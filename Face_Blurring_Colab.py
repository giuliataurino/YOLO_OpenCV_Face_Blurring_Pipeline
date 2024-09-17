#install yoloface and import face_analysis from package
#import necessary packages
!pip3 install yoloface
from yoloface import face_analysis
face=face_analysis()
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

#from google.colab (or elsewhere) import sample images
image_location = 'M214_1015935974_0001_0002.jpg' #Archive_image_test.jpg
image = Image.open(image_location)
plt.imshow(image)
plt.show()

#convert to greyscale
import cv2
image = cv2.imread("/content/sample_data/M214_1015935974_0048_0007.jpg")
grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite("M214_1015935974_0048_0007_grayscale.jpg", grayscale_image)
plt.imshow(image)
plt.show()

#create a function yolo_face_detection to count detected faces and return coordinates
def yolo_face_detection(image_location):
    img,box,conf=face.face_detection(image_location, model='tiny')
    print(str(len(box)) + " total faces detected.")
    x_list, y_list, h_list, w_list = [], [], [], []
    numb_faces_detected = len(box)
    for i in range(len(box)):
        x, y, h, w = box[i]
        x_list.append(x), y_list.append(y), h_list.append(h), w_list.append(w)
        print(f"Face detected in the box (x,y,x+w,y+h) {x} {y} {x+w} {y+h}")
    return x_list, y_list, h_list, w_list,numb_faces_detected

#List numb faces detected
x_list, y_list, h_list, w_list, numb_faces_detected = yolo_face_detection(image_location)

#Blur faces
image_np = np.array(image)
for i in range(numb_faces_detected):
    x, y, h, w = x_list[i], y_list[i], h_list[i], w_list[i]
    f = image_np[y:y + h, x:x + w]
    blurred_face = cv2.GaussianBlur (f, (99, 99), 15)  # You can adjust blur parameters
    image_np[y:y + h, x:x + w] = blurred_face

# Convert the numpy array back to an image
image = Image.fromarray(image_np)
plt.imshow(image)
plt.show()

