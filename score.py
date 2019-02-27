import os
import json
from io import BytesIO
import numpy as np
import pickle
import requests
from sklearn.externals import joblib
from sklearn.svm import SVC
from PIL import Image
import utils

from azureml.core.model import Model

def init():
    global model, target_image_size
    model_name='your_model_name'
    target_image_size=128
    # retreive the path to the model file using the model name
    model_path = Model.get_model_path(model_name)
    model = joblib.load(model_path)

def run(raw_data):
    url = json.loads(raw_data)['url']
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img_arr = np.array(utils.processImg(img, target_image_size)).flatten()
    result = model.predict([img_arr])
    # you can return any data type as long as it is JSON-serializable
    return np.array_str(result)
