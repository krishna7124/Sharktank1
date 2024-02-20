# Modification TestCase1.py:  Detects only when eyes is closed
# Modification Testcase2.py: Detects small and faces at far distance also can detect multiple faces
# Modification Testcase3.py: Improved Eye detection algo. Added Perfect time frame alert after 5s
# Modification Testcase4.py: Improved Detection algo while face is moving
# Modification Testcase5.py: Added Alarm Buzz system to alert driver if they close their eyes.
# Modification Testcase6.py: Added Alarm to ask the driver to take rest aftre particular kilometers or hours

import numpy as np
import cv2
import time
import pygame

# Initialize pygame mixer
pygame.mixer.init()

# Load the sound files
alert_sound = pygame.mixer.Sound("alert.wav")  # Alert sound for eyes closed
# Sound to remind driver to take a rest
rest_sound = pygame.mixer.Sound("rest.wav")

# Rest reminder interval in hours
rest_interval_hours = 0.01   # Reminder to take rest every 4 hours

# Total journey distance in kilometers
total_journey_distance_km = 1000

# Average speed assumption in km/h
average_speed_km_per_hour = 70

# Calculate total journey time in hours
total_journey_time_hours = total_journey_distance_km / average_speed_km_per_hour

# Variables to store execution state
eyes_open_timer = 0
alert_triggered = False
start_time = 0
start_detection = False

# Starting the video capture
cap = cv2.VideoCapture(0)
# Set camera resolution to 640x480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Initialize face and eye cascade classifiers
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml')

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    # Convert the recorded image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detecting the face for region of image to be fed to eye classifier
    if start_detection:
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                # Draw rectangle around face
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Eye detection within the region of interest (face)
                roi_gray = gray[y:y + h, x:x + w]
                eyes = eye_cascade.detectMultiScale(
                    roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))

                # Check if detected objects are eyes
                for (ex, ey, ew, eh) in eyes:
                    eye_center = (x + ex + ew // 2, y + ey + eh // 2)
                    cv2.circle(img, eye_center, min(
                        ew, eh) // 2, (255, 255, 255), 2)

                # Check if both eyes are open
                if len(eyes) >= 2:
                    # Reset timer if both eyes are open
                    if alert_triggered:
                        alert_triggered = False
                        # Stop the alert sound if it's playing
                        pygame.mixer.stop()
                    eyes_open_timer = 0
                    cv2.putText(img, "Eyes open!", (x, y - 10),
                                cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
                else:
                    # Increment timer if eyes are closed
                    eyes_open_timer += 1
                    # Check if 5 seconds have elapsed
                    if eyes_open_timer > 5 and not alert_triggered:
                        print("Alert: Eyes closed for too long")
                        alert_triggered = True
                        # Play the alert sound
                        alert_sound.play()
                    cv2.putText(img, "Eyes closed!", (x, y - 10),
                                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
        else:
            cv2.putText(img, "No face detected", (100, 100),
                        cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
    else:
        cv2.putText(img, "Press 's' to start detection", (100, 100),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    # Calculate time elapsed
    if start_detection:
        elapsed_time = time.time() - start_time
        # Check if rest reminder needs to be triggered
        if elapsed_time >= rest_interval_hours * 3600:
            print("Alert: Please take a rest")
            rest_sound.play()
            # Reset start time for rest interval
            start_time = time.time()

    # Display the captured frame
    cv2.imshow('img', img)
    a = cv2.waitKey(1)
    if a == ord('q'):
        break
    elif a == ord('s'):
        start_detection = True
        start_time = time.time()  # Start time for time elapsed calculation

cap.release()
cv2.destroyAllWindows()
