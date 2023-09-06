import cv2
import imutils
import numpy as np
import RPi.GPIO as GPIO
import time
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'


# get the image and save to variable capturedImage
capturedImage = cv2.imread(
    '/home/pi/repos/captured_image.jpg', cv2.IMREAD_COLOR)

# resize the image to width 600 and height 400
capturedImage = cv2.resize(capturedImage, (600, 400))

# convert the image to gray colour
greyedImage = cv2.cvtColor(capturedImage, cv2.COLOR_BGR2GRAY)

# remove noise and enhance image quality while keeping the edges sharp
greyedImage = cv2.bilateralFilter(greyedImage, 13, 15, 15)

# detect the edges in an image using canny edge detection algorithm while specifying the gradient lower and upper threshold limit
detectedEdges = cv2.Canny(greyedImage, 30, 200)

# find the lines connecting each point of the same colour in the border of the detected image edges
detectedContours = cv2.findContours(
    detectedEdges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# extract the output of the detected contours
detectedContours = imutils.grab_contours(detectedContours)

# sort the detected Contours in descending order and select top 10
detectedContours = sorted(
    detectedContours, key=cv2.contourArea, reverse=True)[:10]

storeContour = None

# draw the countour of the entire image with a red colour and border thickness 3

for cnt in detectedContours:

    contourPerimeter = cv2.arcLength(cnt, True)
    contourApproximation = cv2.approxPolyDP(
        cnt, 0.018 * contourPerimeter, True)

    if len(contourApproximation) == 4:
        storeContour = contourApproximation
        break

if storeContour is None:
    detected = 0
    print("Could not dectect contour")
else:
    detected = 1

if detected == 1:
    cv2.drawContours(capturedImage, [storeContour], -1, (0, 0, 255), 3)

# create a new array of the greyedImage shape with datatype uint8
maskedImage = np.zeros(greyedImage.shape, np.uint8)

# draw the contour of the maskedImage usig the contour approximation value of the original image,  a white colour with a filled border
newImage = cv2.drawContours(maskedImage, [storeContour], 0, 255, -1,)

# performs a bitwise AND operation between the capturedImage and maskedImage arrays,
newImage = cv2.bitwise_and(
    capturedImage, capturedImage, mask=maskedImage)

# checks where the maskedImage has colour white and create an array indices to variable x and y
(x, y) = np.where(maskedImage == 255)

# find the minimum value in the variable x and y
(topx, topy) = (np.min(x), np.min(y))

# find the maximum value in the variable x and y
(bottomx, bottomy) = (np.max(x), np.max(y))

# specifies the vertical and horizontal range of rows to crop from the greyedimage.
croppedPlateNumber = greyedImage[topx:bottomx+1, topy:bottomy+1]

text = pytesseract.image_to_string(
    croppedPlateNumber, lang='eng', config='--psm 11')
print("Detected license plate Number: ", text)

# resize original image
capturedImage = cv2.resize(capturedImage, (500, 300))

# resize extracted plate number
croppedPlateNumber = cv2.resize(croppedPlateNumber, (400, 200))

# show original and extracted number plate
cv2.imshow('capturedimage', capturedImage)
cv2.imshow('platenumber', croppedPlateNumber)

# turn on LED light
GPIO.setmode(GPIO.BCM)
GPIO.setup(18, GPIO.OUT)
GPIO.output(18, GPIO.HIGH)

# pause for a keyboard event
cv2.waitKey(0)

# reset and released the GPIO (General Purpose Input/Output)
GPIO.cleanup()

# close all windows created in the program execution
cv2.destroyAllWindows()
