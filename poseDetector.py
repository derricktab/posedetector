import cv2
import mediapipe as mp
import tensorflow as tf
import time
import pyautogui
import threading
from PIL import Image, ImageOps
import numpy as np
from skimage.transform import resize
from win10toast import ToastNotifier


start_time = 0
stop_time = 0
elapsed_time = 0
sitting_time = "00:00"


# Create a function to start the timer
def start_timer():
    global start_time
    start_time = time.time()
    update_time()


# Create a function to stop the timer
def stop_timer():
    global start_time
    start_time = None


# Create a function to update the time display
def update_time():
    global elapsed_time
    global start_time
    global sitting_time

    if start_time:
        elapsed_time = time.time() - start_time
        time_string = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        sitting_time = time_string
        if(sitting_time == "00:05:00"):
            print("FIVE MINUTES HAS ELAPSED")
            toaster = ToastNotifier()
            toaster.show_toast("You Have Sat For So Long!!","Please Stand Up And Strech Abit", duration=10)


model = tf.keras.models.load_model("model/keras_model.h5", compile=False)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
mp_face_detection = mp.solutions.face_detection
width, height = pyautogui.size()

# Calculate the center of the screen
center_x = width // 2
center_y = height // 2

# initializing the counter to zero
counter = 0

# For webcam input:
cap = cv2.VideoCapture(0)

# face detection
with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:

    # pose detection
    with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue
            
            # new_image = np.expand_dims(image, axis=-1)
            new_image = resize(image, (224, 224))
            new_image = np.expand_dims(new_image, axis=0)            
            
            prediction = model.predict(new_image)
            predicted_class = np.argmax(prediction)

            words = ""

            # CHECKING WHICH CLASS THE PREDICTION BELONGS TO
            if(predicted_class == 0):
                words = "GOOD SITTING POSTURE"
            else:
                words = "BAD SITTING POSTURE"
            

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            results = pose.process(image)
            face_results = face_detection.process(image)
            
            # Define the rectangle coordinates
            pt1 = (0, 10)
            pt2 = (150, 50)

            # Draw the rectangle on the image
            cv2.rectangle(image, pt1, pt2, (0, 255, 0), thickness=-1)

            # PUTTING THE TIMER ON THE IMAGE
            cv2.putText(image, sitting_time, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Draw the face detection annotations on the image.
            image.flags.writeable = True
            if face_results.detections:

                for detection in face_results.detections:
                    x = int(
                        detection.location_data.relative_bounding_box.xmin * 640)
                    y = int(
                        detection.location_data.relative_bounding_box.ymin * 480)
                    w = int(
                        detection.location_data.relative_bounding_box.width * 640)
                    h = int(
                        detection.location_data.relative_bounding_box.height * 480)

                    # DRAWING THE BOUNDING BOX
                    cv2.rectangle(image, pt1=(
                        x-180, y-80), pt2=(x + w + 180, y + h + 180), color=(0, 255, 0), thickness=2)

                    # PUTTING THE PREDICTED POSTURE ON THE IMAGE
                    cv2.putText(image, words, (x-120, y - 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                if start_time == 0:
                    # STARTING THE TIMER
                    start_timer()
                    stop_time = 0
                else:
                    update_time()

            else:
                if stop_time == 0:
                    stop_time = time.time()
                    elapsed_time = stop_time - start_time
                    elapsed_time_str = time.strftime("%M:%S", time.gmtime(elapsed_time))
                    print(elapsed_time_str)

                start_time = 0

            # Draw the pose annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

 
            # CREATING THE WINDOW
            cv2.namedWindow('Pose Detector', cv2.WINDOW_NORMAL)

            # SETTING THE DIMENSIONS OF THE WINDOW
            cv2.resizeWindow("Pose Detector", width=800, height=600)

            # Move the window to the center of the screen
            cv2.moveWindow("Pose Detector", center_x - 400, center_y-400)

            # RENDERING THE IMAGE
            cv2.imshow('Pose Detector', image)

            if cv2.waitKey(5) & 0xFF == 27:
                break
cap.release()
