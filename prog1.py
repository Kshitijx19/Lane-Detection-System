import cv2 as cv 
import numpy as np

'''Load your road image

Convert to grayscale

Apply Gaussian blur

Detect edges using Canny

Apply region of interest (mask the road area)

Use Hough Line Transform to detect lane lines

Draw the detected lines back on the original image

Display the results with cv2.imshow()

Tip: Start with one image — its easier to debug.'''

img=cv.imread("road1.jpg")
#cv.imshow("Road",img)

# Create a copy for drawing lines
line_image = np.copy(img)

#gray image
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)

#gaussian blur
blur=cv.GaussianBlur(gray,(5,5),cv.BORDER_DEFAULT)

#canny
canny=cv.Canny(blur,50,150)

height,width=img.shape[:2]

#defining polygone for ROI
mask=np.zeros_like(canny)
polygon = np.array([[
    (width * 0.1, height),          # Bottom-left: 10% from left, 100% from top
    (width * 0.9, height),          # Bottom-right: 90% from left, 100% from top
    (width * 0.6, height * 0.6),    # Top-right: 60% from left, 60% from top
    (width * 0.4, height * 0.6)     # Top-left: 40% from left, 60% from top
]], np.int32)

#fill polygon with white on mask
cv.fillPoly(mask,[polygon],255)

# Apply mask
masked_image = cv.bitwise_and(canny, mask)
# Use Hough Line Transform to detect lane lines
lines = cv.HoughLinesP(masked_image, 2, np.pi/180, threshold=40, 
                       minLineLength=40, maxLineGap=150)

# Draw the detected lines
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 3)


# Combine original image with line image
result = cv.addWeighted(img, 0.8, line_image, 1, 0)

# Display results
#cv.imshow("Canny", canny)
#cv.imshow("Masked Road", masked_image)
cv.imshow("Lane Detection", result)
cv.waitKey(0)
cv.destroyAllWindows()