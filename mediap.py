import cv2
import mediapipe as mp
import tensorflow as tf
import time
import datetime

start_time = 0
stop_time = 0


model = tf.keras.models.load_model("model/keras_model.h5", compile=False)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
mp_face_detection = mp.solutions.face_detection

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

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        face_results = face_detection.process(image)

        # Draw the face detection annotations on the image.
        image.flags.writeable = True
        if face_results.detections:
            if start_time == 0:
                print("running for the first time")
                start_time = datetime.datetime.now()
                print("START TIME")
                print(start_time)
                stop_time = 0

            for detection in face_results.detections:

                x = int(detection.location_data.relative_bounding_box.xmin * 640)
                y = int(detection.location_data.relative_bounding_box.ymin * 480)
                w = int(detection.location_data.relative_bounding_box.width * 640)
                h = int(detection.location_data.relative_bounding_box.height * 480)

                cv2.rectangle(image, pt1=(x-180, y-80), pt2=(x + w + 180, y + h + 180), color=(0, 255, 0), thickness=2)
                cv2.putText(image, "GOOD SITTING POSTURE", (x, y - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            else:
                if stop_time == 0:
                    stop_time = datetime.datetime.now()
                    print(type(stop_time))
                    print(type(start_time))
                    time_elapsed = stop_time - start_time

                    # convert to minutes
                    time_elapsed = time_elapsed.seconds / 60
                    # time_elapsed = datetime.datetime.strptime(time_elapsed).minute
                    print(time_elapsed)
                    start_time = 0


        # Draw the pose annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        cv2.imshow('MediaPipe Pose', image)
        if cv2.waitKey(5) & 0xFF == 27:
          break
cap.release()

