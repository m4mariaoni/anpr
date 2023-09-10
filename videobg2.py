import cv2
import pytesseract
import os

# Set the path to the Tesseract executable (change as needed)
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

# Load the video and create the background subtractor
cap = cv2.VideoCapture('istock1.mp4')
fgbg = cv2.createBackgroundSubtractorMOG2()

# Parameters for region filtering
min_area = 1000  # Minimum area to filter out small regions
min_width = 100   # Minimum width of detected region
min_height = 100  # Minimum height of detected region

output_folder = 'repos'
os.makedirs(output_folder, exist_ok=True)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Apply background subtraction
    fgmask = fgbg.apply(frame)

    # Threshold the mask to obtain a binary image
    _, thresh = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate through the contours
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(contour)
            if w > min_width and h > min_height:
                number_plate = frame[y:y+h, x:x+w]

                # Save the number plate as an image
                image_filename = os.path.join(
                    output_folder, f'plate_{len(os.listdir(output_folder))}.jpg')
                cv2.imwrite(image_filename, number_plate)

                # Perform OCR on the number plate image
                extracted_text = pytesseract.image_to_string(number_plate)
                if extracted_text.strip():  # Check if OCR extracted any text
                    print(f'Extracted Plate Number: {extracted_text.strip()}')

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
