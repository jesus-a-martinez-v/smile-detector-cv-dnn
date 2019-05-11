import argparse
import os

import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
from imutils import paths
from keras.preprocessing.image import img_to_array
from keras.utils import np_utils
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from nn.conv.lenet import LeNet

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('-d', '--dataset', required=True, help='Path to input dataset of faces.')
argument_parser.add_argument('-m', '--model', required=True, help='Path to output model.')
arguments = vars(argument_parser.parse_args())

data = list()
labels = list()

for image_path in sorted(list(paths.list_images(arguments['dataset']))):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = imutils.resize(image, width=28)
    image = img_to_array(image)
    data.append(image)

    label = image_path.split(os.path.sep)[-3]
    label = 'smiling' if label == 'positives' else 'not_smiling'
    labels.append(label)

data = np.array(data, dtype='float') / 255.0
labels = np.array(labels)

label_encoder = LabelEncoder().fit(labels)
labels = np_utils.to_categorical(label_encoder.transform(labels), 2)

class_totals = labels.sum(axis=0)
class_weight = class_totals.max() / class_totals

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=.2, stratify=labels, random_state=42)

print('[INFO] Compiling model...')
model = LeNet.build(width=28, height=28, depth=1, classes=2)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print('[INFO] Training network...')
H = model.fit(X_train, y_train, validation_data=(X_test, y_test), class_weight=class_weight, batch_size=128, epochs=15)

print('[INFO] Evaluating network...')
predictions = model.predict(X_test, batch_size=128)
print(classification_report(y_test.argmax(axis=1), predictions.argmax(axis=1), target_names=label_encoder.classes_))

print('[INFO] Serializing network...')
model.save(arguments['model'])

plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, 15), H.history['loss'], label='train_loss')
plt.plot(np.arange(0, 15), H.history['val_loss'], label='val_loss')
plt.plot(np.arange(0, 15), H.history['acc'], label='train_acc')
plt.plot(np.arange(0, 15), H.history['val_acc'], label='val_acc')
plt.title('Training Loss and Accuracy')
plt.xlabel('Epoch #')
plt.ylabel('Loss/Accuracy')
plt.legend()
plt.show()
