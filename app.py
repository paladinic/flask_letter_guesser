from flask import Flask, render_template, flash, request, Response, jsonify
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
import requests

import cv2
from PIL import Image
import io
import base64

import pandas as pd
import numpy as np

from keras.models import load_model


# App config.
DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'

alphabet = np.array(["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"])

def alph_res(pred):
    return alphabet[pred == max(pred)][0]

def stringToImage(base64_string):
    imgdata = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(imgdata))

def base64_to_img(img_string):
    img = stringToImage(img_string)
    img0 = np.array(img)
    resizedImage = cv2.resize(img0, (28,28))
    resizedImage = resizedImage[:,:,3]
    rescaledImage = resizedImage*(16/255)
    rescaledImage = rescaledImage.reshape(-1)
    return rescaledImage

# load neural network
letter_model = load_model("letters_model.h5py")

# home page
@app.route("/", methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route("/_result")
def result():
    url = request.args.get('img_url', "",type=str)

    data = url.replace("ONEPLUS","+").replace   ("ONEEQUALS","=").replace("ONESLASH","/")
    data = data.replace("ONEPLUS","+").replace("ONEEQUALS","=").replace("ONESLASH","/")

    img = base64_to_img(data)
    imgs = np.array([img])

    pred = letter_model.predict(imgs)
    res = alph_res(pred[0])

    return jsonify(result=res)

app.run(debug=True, port=5000, host="0.0.0.0")
