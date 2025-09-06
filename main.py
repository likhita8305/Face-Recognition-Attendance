import cv2
import os
import csv
from datetime import datetime

# Haar Cascade file for face detection
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

face_cascade = cv2.CascadeClassifier(cascade_path)
if face_cascade.empty():
    raise RuntimeError("Failed to load Haar Cascade file!")

# Create an 'attendance.csv' file if it doesn't exist
if not os.path.exists("attendance.csv"):
    with open("attendance.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Name", "Date", "Time"])

def mark_attendance(name):
    """Add a person's attendance with date and time."""
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    with open("attendance.csv", mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([name, date, time])
    print(f"[INFO] Attendance marked for {name} at {time}")

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame from webcam")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Here you can replace 'Unknown' with actual face recognition logic
        name = "Unknown"
        mark_attendance(name)

    # Display the frame
    cv2.imshow("Face Recognition Attendance", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
