# # from tensorflow import keras
# import numpy as np
# from PIL import Image
# from keras.utils import *
#
# image = load_img("1.jpg", target_size=(224, 224))
# image = img_to_array(image)
# print(image.shape)
# # image = np.asarray(image)
# # print(image.shape)
# #
# # image = Image.open("1.jpg")
# #
# # image_array = np.asarray(image, dtype="float32")
# # print(image_array.shape)
# #
# # image_array.astype(np.float32).reshape([28, 28, 3])/255.0
# # print(image_array.shape)
#


import cv2


video = cv2.VideoCapture(0)

while video.isOpened():
    success, image = video.read()

    if not success:
        continue

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow("OUR VIDEO", image)

video.release()
