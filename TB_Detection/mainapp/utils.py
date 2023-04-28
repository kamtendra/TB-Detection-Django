import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import os
import cv2
import boto3
from dotenv import load_dotenv

load_dotenv()



s3 = boto3.client(
    's3',
    aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY')
)

s3.download_file('tb-detection', 'TB.h5', 'TB.h5')
s3.download_file('tb-detection', 'tuberculosis.h5', 'tuberculosis.h5')
s3.download_file('tb-detection', 'my_model2.hdf5', 'my_model2.hdf5')

model_path = r'C:\Users\kamte\OneDrive\Documents\Desktop\TB Detection\TB_Detection\mainapp\TB detect model'
model = load_model(model_path)
# model.summary()

model2 = load_model(r'C:\Users\kamte\OneDrive\Documents\Desktop\TB Detection\TB_Detection\TB.h5')
model3 = load_model(r'C:\Users\kamte\OneDrive\Documents\Desktop\TB Detection\TB_Detection\tuberculosis.h5')
model4 = load_model(r'C:\Users\kamte\OneDrive\Documents\Desktop\TB Detection\TB_Detection\my_model2.hdf5')

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

def load_img1(image_path):
  test_data = []
  image=image_path
  img = cv2.imread(str(image))
  img = cv2.resize(img, (512,512))
  if img.shape[2] ==1:
    img = np.dstack([img, img, img])
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img=np.array(img)
  img = img/255
  test_data.append(img)
  test_data1 = np.array(test_data)
  return test_data1

def predict_label1(image_path):
  result="Normal"
  image = load_img1(image_path)
  predict_model = model2.predict(np.array(image))
  if int(predict_model.round()) == 1:
    result="Tuberculosis"
    return result
  else:
    return result
  
def load_img2(image_path):
  test_data = []
  image=image_path
  img = cv2.imread(str(image))
  img = cv2.resize(img, (300,300))
  if img.shape[2] ==1:
    img = np.dstack([img, img, img])
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img=np.array(img)
  img = img/255
  test_data.append(img)
  test_data1 = np.array(test_data)
  return test_data1  

def predict_label2(image_path):
  result="Normal"
  image = load_img2(image_path)
  predict_model = model3.predict(np.array(image))
  if int(predict_model.round()) == 1:
    result="Tuberculosis"
    return result
  else:
    return result
  
def load_img3(image_path):
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

def predict_label3(image_path):
  result="Normal"
  image = load_img3(image_path)
  predict_model = model.predict(np.array(image))
  if np.argmax(predict_model) == 1:
    result="Tuberculosis"
    return result
  else:
    return result
