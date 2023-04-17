import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import os
import cv2

model_path = r'C:\Users\kamte\OneDrive\Documents\Desktop\TB Detection\TB_Detection\mainapp\TB detect model'
model = load_model(model_path)
# model.summary()

def load_img(image_path):
  test_data = []
  image=image_path
  img = cv2.imread(str(image))
  img = cv2.resize(img, (28,28))
  if img.shape[2] ==1:
    img = np.dstack([img, img, img])
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img=np.array(img)
  img = img/255
  test_data.append(img)
  test_data1 = np.array(test_data)
  return test_data1

def predict_label(image_path):
  result="Normal"
  image = load_img(image_path)
  predict_model = model.predict(np.array(image))
  if np.argmax(predict_model) == 1:
    result="Tuberculosis"
    return result
  else:
    return result
