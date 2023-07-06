import cv2
from deepface import DeepFace
img=cv2.imread("kohli.jpg")
results=DeepFace.analyze(img,actions={"race"})
print(results)