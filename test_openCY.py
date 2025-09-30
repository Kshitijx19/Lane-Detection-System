import cv2 
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread("road.jpg")
gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.subplot(1,2,2)
plt.title("Grayscale")
plt.imshow(gray, cmap="gray")
plt.show()