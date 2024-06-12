import cv2
import time
import math as m
import mediapipe as mp
import math
import keyboard
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

script_directory = os.path.dirname(os.path.abspath(__file__))
hands_module_path = os.path.join(script_directory, 'mediapipe', 'python', 'solutions', 'hands.py')
os.chdir('D:/AIML/dataset/Toyata_PAP')
import pymongo

import matplotlib.pyplot as plt

# Initialize MediaPipe hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize MediaPipe drawing module
mp_drawing = mp.solutions.drawing_utils



# Initialize variables for line and count
line_x = 50
count = 0
line_touched = False




# Connect to the MongoDB server
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["Toyota_PAP"]
collection = db["Pick_and_Place"]



trainer_time = 52

try:
    stored_data = pd.read_excel('graph.xlsx')
    data = stored_data.values.tolist()
except FileNotFoundError:
    data = []

# Start the serial number from the next available number
if data:
    start_serial = data[-1][0] + 1
else:
    start_serial = 1

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()


# Calculate distance
def findDistance(x1, y1, x2, y2):
    dist = m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist


# Initialize variables


# Calculate angle.
def findAngle(x1, y1, x2, y2):
    theta = m.acos((y2 - y1) * (-y1) / (m.sqrt(
        (x2 - x1) ** 2 + (y2 - y1) ** 2) * y1))
    degree = int(180 / m.pi) * theta
    return degree

def calculate_angle(a, b, c):
    """Calculate the angle between three points."""
    ang_rad = math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    return math.degrees(ang_rad)


"""
Function to send alert. Use this function to send alert when bad posture detected.
Feel free to get creative and customize as per your convenience.
"""


def sendWarning(x):
    pass


# =============================CONSTANTS and INITIALIZATIONS=====================================#
# Initilize frame counters.
start_time = None
elapsed_time = 0
good_frames = 0

bad_frames = 0
badT=0
badN=0
badD=0

is_counting = False

# Font type.
font = cv2.FONT_HERSHEY_SIMPLEX

# Colors.
blue = (255, 0, 0)
red = (0, 0, 255)
green = (0, 255, 0)
dark_blue = (127, 20, 0)
light_green = (127, 233, 100)
yellow = (0, 255, 255)
pink = (255, 0, 255)
orange = (0, 165, 255)
# Initialize mediapipe pose class.

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
# ===============================================================================================#


if __name__ == "__main__":
    start_time = None
    elapsed_time = 0
    good_frames = 0
    bad_frames = 0
    badT = 0
    badN = 0
    badD = 0
    is_counting = False
    # For webcam input replace file name with 0.
    file_name = 'input.mp4'
    cap = cv2.VideoCapture(0)

    # Meta.
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (width, height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Video writer.
    video_output = cv2.VideoWriter('output.mp4', fourcc, fps, frame_size)

    while cap.isOpened():

        # Capture frames.
        success, image = cap.read()
        # Check for key press
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define the lower and upper bounds for the blue color
        lower_blue = np.array([80, 150, 100])
        upper_blue = np.array([150, 255, 255])

        # Create a mask that filters only blue color
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Initialize the center of the blue card
        center = None



        if contours:
            # Find the largest contour (assuming the blue card is the largest)
            largest_contour = max(contours, key=cv2.contourArea)

            # Get the bounding box of the blue card
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Calculate the center of the blue card
            cx = x + w // 2
            cy = y + h // 2
            center = (cx, cy)
            table_position = (cx, cy)
            # Draw a bounding box around the blue card
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), 1)

            # Draw a red circle at the center
            cv2.circle(image, center, 5, (0, 0, 255), -1)



        # Check if the 'A' key is pressed to start the timer
        if keyboard.is_pressed('a') and start_time is None:
            start_time = time.time()  # Start the timer
            is_counting = True

            # Check if the 'S' key is pressed to stop the timer
        if keyboard.is_pressed('s') and start_time is not None:
            elapsed_time += time.time() - start_time  # Add the elapsed time to the total
            start_time = None  # Stop the timer
            is_counting = False
            # Check if the 'R' key is pressed to reset the timer
        if keyboard.is_pressed('r'):
            start_time = None
            elapsed_time = 0
            good_frames = 0
            bad_frames = 0
            badT = 0
            badN = 0
            badD = 0
            is_counting = False

        if start_time is not None:
            current_time = time.time()
            elapsed_time += current_time - start_time
            start_time = current_time

        # Convert the total elapsed time to hours, minutes, and seconds


        if not success:
            print("Null.Frames")
            break
        # Get fps.
        fps = cap.get(cv2.CAP_PROP_FPS)
        # Get height and width.
        h, w = image.shape[:2]

        # Convert the BGR image to RGB.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image.
        keypoints = pose.process(image)
        results = hands.process(image)
        # Convert the image back to BGR.
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Use lm and lmPose as representative of the following methods.
        lm = keypoints.pose_landmarks
        lmPose = mp_pose.PoseLandmark

        # Acquire the landmark coordinates.
        # Once aligned properly, left or right should not be a concern.

        # Left wrist.
        l_wrist_x = int(lm.landmark[lmPose.LEFT_WRIST].x * w)
        l_wrist_y = int(lm.landmark[lmPose.LEFT_WRIST].y * h)
        # Right wrist
        r_wrist_x = int(lm.landmark[lmPose.RIGHT_WRIST].x * w)
        r_wrist_y = int(lm.landmark[lmPose.RIGHT_WRIST].y * h)

        # Left elbow.
        l_elbow_x = int(lm.landmark[lmPose.LEFT_ELBOW].x * w)
        l_elbow_y = int(lm.landmark[lmPose.LEFT_ELBOW].y * h)
        # Right elbow
        r_elbow_x = int(lm.landmark[lmPose.RIGHT_ELBOW].x * w)
        r_elbow_y = int(lm.landmark[lmPose.RIGHT_ELBOW].y * h)

        # Left shoulder.
        l_shldr_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * w)
        l_shldr_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)
        # Right shoulder
        r_shldr_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w)
        r_shldr_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h)
        # Left ear.
        l_ear_x = int(lm.landmark[lmPose.LEFT_EAR].x * w)
        l_ear_y = int(lm.landmark[lmPose.LEFT_EAR].y * h)
        # Left hip.
        l_hip_x = int(lm.landmark[lmPose.LEFT_HIP].x * w)
        l_hip_y = int(lm.landmark[lmPose.LEFT_HIP].y * h)

        # right_shoulder = (int(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].x * image.shape[1]),
        #                   int(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].y * image.shape[0]))
        # right_elbow = (int(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ELBOW].x * image.shape[1]),
        #                int(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ELBOW].y * image.shape[0]))
        # right_wrist = (int(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST].x * image.shape[1]),
        #                int(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST].y * image.shape[0]))

        # Extract keypoints for left side
        # left_shoulder = (
        # int(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].x * image.shape[1]),
        # int(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].y * image.shape[0]))
        # left_elbow = (int(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_ELBOW].x * image.shape[1]),
        #               int(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_ELBOW].y * image.shape[0]))
        # left_wrist = (int(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST].x * image.shape[1]),
        #               int(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST].y * image.shape[0]))
        # left_heap = (int(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_HIP].x * image.shape[1]),
        #               int(results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_HIP].y * image.shape[0]))

        # Calculate distance between left shoulder and right shoulder points.
        offset = findDistance(l_shldr_x, l_shldr_y, r_shldr_x, r_shldr_y)

        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                # Assuming that the left hand is the first detected hand
                left_fingertip_landmark = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                # Convert landmark coordinates to pixel values
                image_h, image_w, _ = image.shape
                left_fingertip_x = int(left_fingertip_landmark.x * image_w)
                left_fingertip_y = int(left_fingertip_landmark.y * image_h)

                # Draw a point on the left fingertip
                cv2.circle(image, (left_fingertip_x, left_fingertip_y), 5, (0, 0, 255), -1)

                # Check if the fingertip touches the line (within a range)
                if abs(left_fingertip_x - line_x) <= 10:
                    if not line_touched:
                        count += 1
                        line_touched = True
                else:
                    line_touched = False

        cv2.line(image, (line_x, 0), (line_x, 600), (255, 255, 255), 1)

        # Display the count
        cv2.putText(image, f'Count: {int(count / 2)}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 205), 2)





        if keypoints.pose_landmarks:
            # Extract hip position
            hip = keypoints.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]

            hip_x = int(hip.x * image.shape[1])
            hip_y = int(hip.y * image.shape[0])

            # Calculate Euclidean distance between person's hip and table
            distance = math.sqrt((hip_x - table_position[0]) ** 2 + (hip_y - table_position[1]) ** 2)
            distance = (distance / 10) + 30
            # Determine color based on distance range
            total_elapsed_seconds = int(elapsed_time)
            elapsed_hours = total_elapsed_seconds // 3600
            elapsed_minutes = (total_elapsed_seconds % 3600) // 60
            elapsed_seconds = total_elapsed_seconds % 60

            # Format the elapsed time as a string
            elapsed_time_string = f"Elapsed Time:{elapsed_minutes:02d}:{elapsed_seconds:02d}"

            # Display the elapsed time on the video frame

            cv2.putText(image, elapsed_time_string, (w - 280, h - 50), font, 0.9, (255, 255, 255), 2)
            if distance < 50:
                badT += 0
                badN += 0
                badD += 0
                color = (0, 255, 0)  # Green
                cv2.putText(image, f"Distance: {distance:.2f} cm", (w - 200, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            color, 2)

            elif 50.1 <= distance <= 60.0:
                badT += 0
                badN += 0
                badD += 1
                color = (255, 0, 0)  # Blue
                cv2.putText(image, f"Distance: {distance:.2f} cm", (w - 200, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            color, 2)
            elif 60.1 <= distance <= 70.0:
                badT += 0
                badN += 0
                badD += 2
                color = (0, 255, 255)  # Yellow
                cv2.putText(image, f"Distance: {distance:.2f} cm", (w - 200, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            color, 2)
            elif 70.1 <= distance <= 80.0:
                badT += 0
                badN += 0
                badD += 3
                color = (0, 165, 255)  # Orange
                cv2.putText(image, f"Distance: {distance:.2f} cm", (w - 200, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            color, 2)
            elif distance > 80.1:
                badT += 0
                badN += 0
                badD += 4
                color = (0, 0, 255)  # Red
                cv2.putText(image, f"Distance: {distance:.2f} cm", (w - 200, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            color, 2)

            # Draw circle at person's hip and line to table with the determined color
            cv2.circle(image, (hip_x, hip_y), 5, color, -1)
            cv2.line(image, (hip_x, hip_y), table_position, color, 2)

            # Convert elapsed time to hours, minutes, and seconds

            # Format the time as a string

        # Display the elapsed time on the video frame

        # Assist to align the camera to point at the side view of the person.
        # Offset threshold 30 is based on results obtained from analysis over 100 samples.
        if offset < 95:
            cv2.putText(image, str(int(offset)) + ' Aligned', (w - 235, 30), font, 0.9, green, 2)
        else:
            cv2.putText(image, str(int(offset)) + ' Not Aligned', (w - 235, 30), font, 0.9, red, 2)

        # Calculate angles.
        neck_inclination = findAngle(l_shldr_x, l_shldr_y, l_ear_x, l_ear_y)
        torso_inclination = findAngle(l_hip_x, l_hip_y, l_shldr_x, l_shldr_y)
        elbow_inclination = findAngle(l_elbow_x, l_elbow_y, l_wrist_x, l_wrist_y)
        sholder_inclination = findAngle(l_shldr_x, l_shldr_y, l_elbow_x, l_elbow_y)
        # Draw landmarks.
        cv2.circle(image, (l_shldr_x, l_shldr_y), 7, yellow, -1)
        cv2.circle(image, (l_ear_x, l_ear_y), 7, yellow, -1)
        cv2.circle(image, (l_wrist_x, l_wrist_y), 7, yellow, -1)
        cv2.circle(image, (l_elbow_x, l_elbow_y), 7, yellow, -1)

        # Let's take y - coordinate of P3 100px above x1,  for display elegance.
        # Although we are taking y = 0 while calculating angle between P1,P2,P3.
        cv2.circle(image, (l_shldr_x, l_shldr_y - 100), 7, yellow, -1)
        cv2.circle(image, (r_shldr_x, r_shldr_y), 7, pink, -1)

        #cv2.circle(image, (r_wrist_x, r_wrist_y), 7, pink, -1)
        #cv2.circle(image, (r_elbow_x, r_elbow_y), 7, pink, -1)

        cv2.circle(image, (l_hip_x, l_hip_y), 7, yellow, -1)

        # Similarly, here we are taking y - coordinate 100px above x1. Note that
        # you can take any value for y, not necessarily 100 or 200 pixels.
        cv2.circle(image, (l_hip_x, l_hip_y - 100), 7, yellow, -1)

        # Put text, Posture and angle inclination.
        # Text string for display.
        angle_text_string = 'Neck : ' + str(int(neck_inclination)) + '  Torso : ' + str(int(torso_inclination))

        #right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
        # left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        # left_shoulder_angle = calculate_angle(left_heap,left_shoulder, left_elbow,)

        # Determine whether good posture or bad posture.
        # The threshold angles have been set based on intuition.
        # if (left_elbow_angle < 90):
        #     cv2.putText(image, f"{left_elbow_angle:.2f} ", (left_elbow[0], left_elbow[1] - 20),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # else:cv2.putText(image, f"{left_elbow_angle:.2f} ", (left_elbow[0], left_elbow[1] - 20),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        #
        #
        # if (left_shoulder_angle < 90):
        #     cv2.putText(image, f"{left_shoulder_angle:.2f} ", (left_shoulder[0], left_shoulder[1] - 20),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # else:cv2.putText(image, f"{left_shoulder_angle:.2f} ", (left_shoulder[0], left_shoulder[1] - 20),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        if(elbow_inclination>90):
            cv2.putText(image, str(int(elbow_inclination)), (l_elbow_x + 10, l_elbow_y), font, 0.9, (0, 255, 0),
                        2)
        else:cv2.putText(image, str(int(elbow_inclination)), (l_elbow_x + 10, l_elbow_y), font, 0.9, (0, 0, 255),
                        2)

        if (sholder_inclination > 120):
            cv2.putText(image, str(int(sholder_inclination)), (l_shldr_x + 30, l_shldr_y+20), font, 0.9, (0, 255, 0),
                        2)
        else:
            cv2.putText(image, str(int(elbow_inclination)), (l_shldr_x + 10, l_shldr_y+20), font, 0.9, (0, 0, 255),
                        2)


        if is_counting:

            if ((torso_inclination < 10)):
                badT += 0
                badN += 0
                badD += 0
                if ((neck_inclination < 30)):
                    bad_frames += 0
                    good_frames += 1
                    cv2.putText(image, angle_text_string, (10, 30), font, 0.9, (0, 255, 0), 2)
                    cv2.putText(image, str(int(neck_inclination)), (l_shldr_x + 10, l_shldr_y), font, 0.9, (0, 255, 0),
                                2)

                    # Join landmarks.
                    cv2.line(image, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), (0, 255, 0), 4)
                    cv2.line(image, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 100), (0, 255, 0), 4)


                elif ((31 <= neck_inclination <= 40)):

                    bad_frames += 0
                    badT += 0
                    badN += 1
                    badD += 0
                    good_frames += 1

                    cv2.putText(image, angle_text_string, (10, 30), font, 0.9, (255, 0, 0), 2)

                    cv2.putText(image, str(int(neck_inclination)), (l_shldr_x + 10, l_shldr_y), font, 0.9, (255, 0, 0),
                                2)

                    # Join landmarks.

                    cv2.line(image, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), (255, 0, 0), 4)

                    cv2.line(image, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 100), (255, 0, 0), 4)


                elif ((41 <= neck_inclination <= 50)):

                    bad_frames += 0
                    badT += 0
                    badN += 2
                    badD += 0
                    good_frames += 0

                    cv2.putText(image, angle_text_string, (10, 30), font, 0.9, (0, 255, 255), 2)

                    cv2.putText(image, str(int(neck_inclination)), (l_shldr_x + 10, l_shldr_y), font, 0.9,
                                (0, 255, 255), 2)

                    # Join landmarks.

                    cv2.line(image, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), (0, 255, 255), 4)

                    cv2.line(image, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 100), (0, 255, 255), 4)

                elif ((51 <= neck_inclination <= 80)):

                    bad_frames += 1
                    badT += 0
                    badN += 3
                    badD += 0
                    good_frames += 0

                    cv2.putText(image, angle_text_string, (10, 30), font, 0.9, (0, 165, 255), 2)

                    cv2.putText(image, str(int(neck_inclination)), (l_shldr_x + 10, l_shldr_y), font, 0.9,
                                (0, 165, 255), 2)

                    # Join landmarks.

                    cv2.line(image, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), (0, 165, 255), 4)

                    cv2.line(image, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 100), (0, 165, 255), 4)



                elif ((neck_inclination >= 71)):

                    good_frames += 0

                    bad_frames += 1
                    badT += 0
                    badN += 4
                    badD += 0

                    cv2.putText(image, angle_text_string, (10, 30), font, 0.9, (0, 0, 255), 2)

                    cv2.putText(image, str(int(neck_inclination)), (l_shldr_x + 10, l_shldr_y), font, 0.9, (0, 0, 255),
                                2)

                    # Join landmarks.

                    cv2.line(image, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), (0, 0, 255), 4)

                    cv2.line(image, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 100), (0, 0, 255), 4)

                cv2.putText(image, angle_text_string, (10, 30), font, 0.9, (0, 255, 0), 2)

                cv2.putText(image, str(int(torso_inclination)), (l_hip_x + 10, l_hip_y), font, 0.9, (0, 255, 0), 2)

                # Join landmarks.
                cv2.line(image, (l_hip_x, l_hip_y), (l_shldr_x, l_shldr_y), (0, 255, 0), 4)
                cv2.line(image, (l_hip_x, l_hip_y), (l_hip_x, l_hip_y - 100), (0, 255, 0), 4)

            elif ((11 <= torso_inclination <= 30)):
                badT += 1
                badN += 0
                badD += 0
                if ((neck_inclination < 30)):
                    bad_frames += 0
                    good_frames += 1
                    badT += 0
                    badN += 0
                    badD += 0

                    cv2.putText(image, angle_text_string, (10, 30), font, 0.9, (0, 255, 0), 2)
                    cv2.putText(image, str(int(neck_inclination)), (l_shldr_x + 10, l_shldr_y), font, 0.9, (0, 255, 0),
                                2)

                    # Join landmarks.

                    cv2.line(image, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), (0, 255, 0), 4)

                    cv2.line(image, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 100), (0, 255, 0), 4)

                elif ((31 <= neck_inclination <= 40)):
                    badT += 0
                    badN += 1
                    badD += 0
                    bad_frames += 0

                    good_frames += 1

                    cv2.putText(image, angle_text_string, (10, 30), font, 0.9, (255, 0, 0), 2)

                    cv2.putText(image, str(int(neck_inclination)), (l_shldr_x + 10, l_shldr_y), font, 0.9, (255, 0, 0),
                                2)

                    # Join landmarks.

                    cv2.line(image, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), (255, 0, 0), 4)

                    cv2.line(image, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 100), (255, 0, 0), 4)


                elif ((41 <= neck_inclination <= 50)):

                    bad_frames += 0
                    badT += 0
                    badN += 2
                    badD += 0
                    good_frames += 0

                    cv2.putText(image, angle_text_string, (10, 30), font, 0.9, (0, 255, 255), 2)

                    cv2.putText(image, str(int(neck_inclination)), (l_shldr_x + 10, l_shldr_y), font, 0.9,
                                (0, 255, 255), 2)

                    # Join landmarks.

                    cv2.line(image, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), (0, 255, 255), 4)

                    cv2.line(image, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 100), (0, 255, 255), 4)

                elif ((51 <= neck_inclination <= 80)):

                    bad_frames += 1
                    badT += 0
                    badN += 3
                    badD += 0
                    good_frames += 0

                    cv2.putText(image, angle_text_string, (10, 30), font, 0.9, (0, 165, 255), 2)

                    cv2.putText(image, str(int(neck_inclination)), (l_shldr_x + 10, l_shldr_y), font, 0.9,
                                (0, 165, 255), 2)

                    # Join landmarks.

                    cv2.line(image, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), (0, 165, 255), 4)

                    cv2.line(image, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 100), (0, 165, 255), 4)


                elif ((neck_inclination >= 71)):

                    good_frames += 0
                    badT += 0
                    badN += 4
                    badD += 0
                    bad_frames += 1

                    cv2.putText(image, angle_text_string, (10, 30), font, 0.9, (0, 0, 255), 2)

                    cv2.putText(image, str(int(neck_inclination)), (l_shldr_x + 10, l_shldr_y), font, 0.9, (0, 0, 255),
                                2)

                    # Join landmarks.

                    cv2.line(image, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), (0, 0, 255), 4)

                    cv2.line(image, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 100), (0, 0, 255), 4)

                cv2.putText(image, angle_text_string, (10, 30), font, 0.9, (255, 0, 0), 2)

                cv2.putText(image, str(int(torso_inclination)), (l_hip_x + 10, l_hip_y), font, 0.9, (255, 0, 0), 2)

                # Join landmarks.
                cv2.line(image, (l_hip_x, l_hip_y), (l_shldr_x, l_shldr_y), (255, 0, 0), 4)
                cv2.line(image, (l_hip_x, l_hip_y), (l_hip_x, l_hip_y - 100), (255, 0, 0), 4)

            elif ((31 <= torso_inclination <= 45)):
                bad_frames += 0
                good_frames += 0
                badT += 2
                badN += 0
                badD += 0
                cv2.putText(image, angle_text_string, (10, 30), font, 0.9, (0, 255, 255), 2)

                cv2.putText(image, str(int(torso_inclination)), (l_hip_x + 10, l_hip_y), font, 0.9, (0, 255, 255), 2)

                # Join landmarks.
                cv2.line(image, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), (0, 255, 255), 4)
                cv2.line(image, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 100), (0, 255, 255), 4)
                cv2.line(image, (l_hip_x, l_hip_y), (l_shldr_x, l_shldr_y), (0, 255, 255), 4)
                cv2.line(image, (l_hip_x, l_hip_y), (l_hip_x, l_hip_y - 100), (0, 255, 255), 4)

            elif ((46 <= torso_inclination <= 75)):
                badT += 3
                badN += 0
                badD += 0
                if ((neck_inclination < 30)):
                    bad_frames += 0
                    good_frames += 1
                    badT += 0
                    badN += 0
                    badD += 0
                    cv2.putText(image, angle_text_string, (10, 30), font, 0.9, (0, 255, 0), 2)
                    cv2.putText(image, str(int(neck_inclination)), (l_shldr_x + 10, l_shldr_y), font, 0.9, (0, 255, 0),
                                2)

                    # Join landmarks.
                    cv2.line(image, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), (0, 255, 0), 4)
                    cv2.line(image, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 100), (0, 255, 0), 4)

                elif ((31 <= neck_inclination <= 40)):

                    bad_frames += 0
                    badT += 0
                    badN += 1
                    badD += 0
                    good_frames += 1

                    cv2.putText(image, angle_text_string, (10, 30), font, 0.9, (255, 0, 0), 2)

                    cv2.putText(image, str(int(neck_inclination)), (l_shldr_x + 10, l_shldr_y), font, 0.9, (255, 0, 0),
                                2)

                    # Join landmarks.

                    cv2.line(image, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), (255, 0, 0), 4)

                    cv2.line(image, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 100), (255, 0, 0), 4)


                elif ((41 <= neck_inclination <= 50)):

                    bad_frames += 0
                    badT += 0
                    badN += 2
                    badD += 0
                    good_frames += 0

                    cv2.putText(image, angle_text_string, (10, 30), font, 0.9, (0, 255, 255), 2)

                    cv2.putText(image, str(int(neck_inclination)), (l_shldr_x + 10, l_shldr_y), font, 0.9,
                                (0, 255, 255), 2)

                    # Join landmarks.

                    cv2.line(image, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), (0, 255, 255), 4)

                    cv2.line(image, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 100), (0, 255, 255), 4)
                elif ((51 <= neck_inclination <= 80)):

                    bad_frames += 1
                    badT += 0
                    badN += 3
                    badD += 0
                    good_frames += 0

                    cv2.putText(image, angle_text_string, (10, 30), font, 0.9, (0, 165, 255), 2)

                    cv2.putText(image, str(int(neck_inclination)), (l_shldr_x + 10, l_shldr_y), font, 0.9,
                                (0, 165, 255), 2)

                    # Join landmarks.

                    cv2.line(image, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), (0, 165, 255), 4)

                    cv2.line(image, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 100), (0, 165, 255), 4)



                elif ((neck_inclination >= 71)):

                    good_frames += 0
                    badT += 0
                    badN += 4
                    badD += 0
                    bad_frames += 1

                    cv2.putText(image, angle_text_string, (10, 30), font, 0.9, (0, 0, 255), 2)

                    cv2.putText(image, str(int(neck_inclination)), (l_shldr_x + 10, l_shldr_y), font, 0.9, (0, 0, 255),
                                2)

                    # Join landmarks.

                    cv2.line(image, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), (0, 0, 255), 4)

                    cv2.line(image, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 100), (0, 0, 255), 4)

                cv2.putText(image, angle_text_string, (10, 30), font, 0.9, (0, 165, 255), 2)

                cv2.putText(image, str(int(torso_inclination)), (l_hip_x + 10, l_hip_y), font, 0.9, (0, 165, 255), 2)

                # Join landmarks.
                cv2.line(image, (l_hip_x, l_hip_y), (l_shldr_x, l_shldr_y), (0, 165, 255), 4)
                cv2.line(image, (l_hip_x, l_hip_y), (l_hip_x, l_hip_y - 100), (0, 165, 255), 4)


            elif ((torso_inclination >= 76)):
                badT += 4
                badN += 0
                badD += 0
                if ((neck_inclination < 30)):
                    bad_frames += 0
                    good_frames += 1
                    badT += 0
                    badN += 0
                    badD += 0
                    cv2.putText(image, angle_text_string, (10, 30), font, 0.9, (0, 255, 0), 2)
                    cv2.putText(image, str(int(neck_inclination)), (l_shldr_x + 10, l_shldr_y), font, 0.9, (0, 255, 0),
                                2)

                    # Join landmarks.
                    cv2.line(image, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), (0, 255, 0), 4)
                    cv2.line(image, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 100), (0, 255, 0), 4)


                elif ((31 <= neck_inclination <= 40)):

                    bad_frames += 0
                    badT += 0
                    badN += 1
                    badD += 0
                    good_frames += 1

                    cv2.putText(image, angle_text_string, (10, 30), font, 0.9, (255, 0, 0), 2)

                    cv2.putText(image, str(int(neck_inclination)), (l_shldr_x + 10, l_shldr_y), font, 0.9, (255, 0, 0),
                                2)

                    # Join landmarks.

                    cv2.line(image, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), (255, 0, 0), 4)

                    cv2.line(image, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 100), (255, 0, 0), 4)

                elif ((41 <= neck_inclination <= 50)):

                    bad_frames += 0
                    badT += 0
                    badN += 2
                    badD += 0
                    good_frames += 0

                    cv2.putText(image, angle_text_string, (10, 30), font, 0.9, (0, 255, 255), 2)

                    cv2.putText(image, str(int(neck_inclination)), (l_shldr_x + 10, l_shldr_y), font, 0.9,
                                (0, 255, 255), 2)

                    # Join landmarks.

                    cv2.line(image, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), (0, 255, 255), 4)

                    cv2.line(image, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 100), (0, 255, 255), 4)

                elif ((51 <= neck_inclination <= 80)):

                    bad_frames += 1
                    badT += 0
                    badN += 3
                    badD += 0
                    good_frames += 0

                    cv2.putText(image, angle_text_string, (10, 30), font, 0.9, (0, 165, 255), 2)

                    cv2.putText(image, str(int(neck_inclination)), (l_shldr_x + 10, l_shldr_y), font, 0.9,
                                (0, 165, 255), 2)

                    # Join landmarks.

                    cv2.line(image, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), (0, 165, 255), 4)

                    cv2.line(image, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 100), (0, 165, 255), 4)



                elif ((neck_inclination >= 71)):

                    good_frames += 0
                    badT += 0
                    badN += 4
                    badD += 0
                    bad_frames += 1

                    cv2.putText(image, angle_text_string, (10, 30), font, 0.9, (0, 0, 255), 2)

                    cv2.putText(image, str(int(neck_inclination)), (l_shldr_x + 10, l_shldr_y), font, 0.9, (0, 0, 255),
                                2)

                    # Join landmarks.

                    cv2.line(image, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), (0, 0, 255), 4)

                    cv2.line(image, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 100), (0, 0, 255), 4)

                cv2.putText(image, angle_text_string, (10, 30), font, 0.9, (0, 0, 255), 2)

                cv2.putText(image, str(int(torso_inclination)), (l_hip_x + 10, l_hip_y), font, 0.9, (0, 0, 255), 2)

                # Join landmarks.
                cv2.line(image, (l_hip_x, l_hip_y), (l_shldr_x, l_shldr_y), (0, 0, 255), 4)
                cv2.line(image, (l_hip_x, l_hip_y), (l_hip_x, l_hip_y - 100), (0, 0, 255), 4)

            # Calculate the time of remaining in a particular posture.


        good_time = (1 / fps) * good_frames
        bad_time = (1 / fps) * bad_frames
        badTe = (1 / fps) * badT
        badNe = (1 / fps) * badN
        badDe = (1 / fps) * badD

        # Pose time.

        time_string_good = 'Good Posture Time : ' + str(round(good_time, 1)) + 's'
        cv2.putText(image, time_string_good, (10, h - 20), font, 0.9, green, 2)

        time_string_bad = 'Bad Posture Time : ' + str(round(bad_time, 1)) + 's'
        cv2.putText(image, time_string_bad, (10, h - 50), font, 0.9, red, 2)

        # If you stay in bad posture for more than 3 minutes (180s) send an alert.
        if bad_time > 180:
            sendWarning()
        # Write frames.

        video_output.write(image)


        # Display.
        cv2.imshow('PIC and PLACE', image)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
       # After calculating good_time and bad_time

        # Append the data to the data list
data.append([start_serial, elapsed_time, good_time, bad_time, badTe/3,badNe/3,badDe/3,])

        # Increment the serial number for the next entry
start_serial += 1

        # Create a new DataFrame with the updated data
new_data = pd.DataFrame(data, columns=["Serial", "Elapsed Time", "Good Time", "Bad Time","TS","BS","DS"])

        # Write the new data to the Excel file
new_data.to_excel('data2.xlsx', index=False)

cap.release()
cv2.destroyAllWindows()


# Load data from data1.xlsx
try:
    stored_data = pd.read_excel('data1.xlsx')
    data = stored_data.values.tolist()
except FileNotFoundError:
    data = []

# Load elapsed time from data2.xlsx
try:
    stored_elapsed_data = pd.read_excel('data2.xlsx')
    elapsed_data = stored_elapsed_data.values.tolist()
except FileNotFoundError:
    elapsed_data = []

# Get the last elapsed time
if elapsed_data:
    last_elapsed_time = elapsed_data[-1][1]
else:
    last_elapsed_time = 0


# Calculate trainee time and other metrics
trainee_time = round(float(last_elapsed_time))
trainer_time = 52  # Fixed trainer time value
time_difference = last_elapsed_time - trainer_time
efficiency = (1 - (time_difference / trainer_time)) * 100
efficiency2 = ((1 - (time_difference / trainer_time)) * 100)-((count/2)*0.5)-((bad_time)*0.5)
data_document = {
        "Serial Number": start_serial,
        "Trainer Time": trainer_time,
        "Trainee Time": trainee_time,
        "Difference": time_difference,
        "Efficiency": efficiency2,

    }
collection.insert_one(data_document)

# Append the new entry to data
if data:
    start_serial = data[-1][0] + 1
else:
    start_serial = 1

data.append((start_serial, trainer_time, trainee_time, time_difference, efficiency2, bad_time, good_time, count/2,badTe/3,badNe/3,badDe/3 ))

# Store the data in an Excel file
df = pd.DataFrame(data, columns=["Serial Number", "Trainer Time", "Trainee Time", "Difference", "Efficiency", "Good_time", "Bad_Time", "Count","Tarso Score","Neck Score","Distance Score"])
df.to_excel('data1.xlsx', index=False)

# Print results
print("\nResults:")
for entry in data:
    serial, trainer, trainee, difference, eff , GT, BT, CT,TS,NS,DS= entry
    print(f"Entry {serial} - Trainer: {trainer} seconds, Trainee: {trainee} seconds, Difference: {difference:.2f} seconds, Efficiency: {eff:.2f}% Good_time:{GT:.2f}s, Bad_time:{BT:.2f}s, Count:{CT:.2f} ,TS:{TS:.2f} ,NS:{NS:.2f} ,DS:{DS:.2f} ")

# Plot graph for Trainer Time and Trainee Time
trainee_times = [entry[2] for entry in data]
trainer_times = [entry[1] for entry in data]

plt.figure(figsize=(10, 6))
plt.plot(trainee_times, label='Trainee Time')
plt.plot(trainer_times, label='Trainer Time')
plt.scatter(range(len(trainee_times)), trainee_times, color='blue', label='Trainee Time Points', marker='o')
plt.scatter(range(len(trainer_times)), trainer_times, color='red', label='Trainer Time Points', marker='s')
plt.xlabel('Number of cycles')
plt.ylabel('Time(seconds)')
plt.title('Trainer Time and Trainee Time over Serial Number')
plt.legend()
plt.show()

# Plot graph for Efficiency
efficiencies = [entry[4] for entry in data]

plt.figure(figsize=(10, 6))
plt.plot(efficiencies, label='Efficiency')
plt.scatter(range(len(efficiencies)), efficiencies, color='green', label='Efficiency Points', marker='^')
plt.xlabel('Number of Cycles')
plt.ylabel('Efficiency(%)')
plt.title('Efficiency over Cycles')
plt.legend()
plt.show()
retrieved_data = collection.find()

# Plot graph for Efficiency
Counts = [entry[7] for entry in data]
TSs = [entry[8] for entry in data]
NSs= [entry[9] for entry in data]
DSs = [entry[10] for entry in data]



plt.figure(figsize=(10, 6))
plt.plot(Counts, label='Count')
plt.plot(NSs, label='Neck Score')
plt.plot(TSs, label='Tarso Score')
plt.plot(DSs, label='Distance Score')
plt.scatter(range(len(Counts)), Counts, color='green', label='Count Points', marker='o')
plt.scatter(range(len(NSs)), NSs, color='yellow', label='Neck Points', marker='s')
plt.scatter(range(len(TSs)), TSs, color='red', label='Tarso Points', marker='^')
plt.scatter(range(len(DSs)), DSs, color='blue', label='Distance Points', marker='o')
plt.xlabel('Number of Cycles')
plt.ylabel('Bad Score')
plt.title('Bad Scores over Cycles')
plt.legend()
plt.show()
retrieved_data = collection.find()

# Iterate through the retrieved documents
# for document in retrieved_data:
#     # Access and print the Trainee Time field
#     trainee_time = document["Trainee Time"]
#     print("Trainee Time:", trainee_time)


# Connect to MongoDB and retrieve trainee time data


retrieved_data = collection.find({}, {"_id": 0, "Trainee Time": 1})

trainee_times = [entry["Trainee Time"] for entry in retrieved_data]

plt.figure(figsize=(10, 6))
plt.plot(trainee_times, marker='o')

plt.title("Trainee Time Graph")
plt.xlabel("Serial Number")
plt.ylabel("Trainee Time (seconds)")
plt.grid(True)

#plt.show()




# Connect to MongoDB and retrieve efficiency data
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["Toyota_PAP"]
collection = db["Pick_and_Place"]

retrieved_data = collection.find({}, {"_id": 0, "Efficiency": 1})
efficiencies = [entry["Efficiency"] for entry in retrieved_data]

# Create and Display Efficiency Graph
plt.figure(figsize=(10, 6))
plt.plot(efficiencies, marker='o', color='green')

plt.title("Efficiency Graph")
plt.xlabel("Serial Number")
plt.ylabel("Efficiency (%)")
plt.grid(True)

#plt.show()
