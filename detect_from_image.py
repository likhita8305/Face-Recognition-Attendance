import cv2
import os

# Load Haar Cascade
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
if not os.path.exists(cascade_path):
    cascade_path = 'haarcascade_frontalface_default.xml'

face_cascade = cv2.CascadeClassifier(cascade_path)

# Input image path
image_path = "sample_images/test_face.jpg"  # Replace with your test image
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image not found at {image_path}")

# Read and process image
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Show output
cv2.imshow("Detected Faces", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

