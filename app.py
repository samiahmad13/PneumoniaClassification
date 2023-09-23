import pickle
import numpy as np
import cv2
import tensorflow as tf
import keras.models
from keras.optimizers import Adam
from flask import Flask, render_template, request
from keras.preprocessing.image import load_img
from keras.models import load_model
from keras.preprocessing.image import img_to_array


app = Flask(__name__)
model = pickle.load(open('models/model.pk1', 'rb'))

@app.route('/', methods=['GET'])
def display():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    imagefile= request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)
    #img = cv2.imread(image_path)
    #resizedimg = tf.image.resize(img, (256,256))
    image = load_img(image_path, target_size=(256,256))
    #yhat = model.predict(np.expand_dims(resizedimg/255, 0))
    yhat = model.predict(np.expand_dims(image/255, 0))
    if yhat > 0.5: 
        classification = "Pneumonia"
    else:
        classification = "Normal"

    return render_template('index.html', prediction=classification)


if __name__ == '__main__':
    app.run(port=3000, debug=True)