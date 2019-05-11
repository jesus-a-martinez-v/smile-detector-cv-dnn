import argparse

import cv2
import imutils
import numpy as np
from keras.models import load_model
from keras_preprocessing.image import img_to_array

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('-c', '--cascade', required=True, help='Path to where the face cascade resides.')
argument_parser.add_argument('-m', '--model', required=True, help='Path to the pre-trained smile detector CNN resides.')
argument_parser.add_argument('-v', '--video', help='Path to the (optional) video file.')
arguments = vars(argument_parser.parse_args())

detector = cv2.CascadeClassifier(arguments['cascade'])
model = load_model(arguments['model'])

if not arguments.get('video', False):
    camera = cv2.VideoCapture(0)
else:
    camera = cv2.VideoCapture(arguments['video'])

while True:
    grabbed, frame = camera.read()

    if arguments.get('video') and not grabbed:
        break

    frame = imutils.resize(frame, width=300)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_copy = frame.copy()

    rectangles = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                           flags=cv2.CASCADE_SCALE_IMAGE)

    for (f_x, f_y, f_w, f_h) in rectangles:
        roi = gray[f_y: f_y + f_h, f_x: f_x + f_w]
        roi = cv2.resize(roi, (28, 28))
        roi = roi.astype('float') / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        not_smiling, smiling = model.predict(roi)[0]
        label = 'Smiling' if smiling > not_smiling else 'Not Smiling'

        cv2.putText(frame_copy, label, (f_x, f_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.rectangle(frame_copy, (f_x, f_y), (f_x + f_w, f_y + f_h), (0, 0, 255), 2)

    cv2.imshow('Face', frame_copy)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
