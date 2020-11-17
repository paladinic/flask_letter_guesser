# import libs

# website
from flask import Flask, render_template, flash, request, Response, jsonify
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
import requests

# images and data
import cv2
from PIL import Image
import io
import base64

# ML
import pandas as pd
import numpy as np
from keras.models import load_model

# App config.
DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'

# array used in functions below
alphabet = np.array(["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"])

# from predictions, return the letter guessed
def alph_res(pred):
    return alphabet[pred == max(pred)][0]

# from base64_string of image, get data
def stringToImage(base64_string):
    imgdata = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(imgdata))

# from base64_string of image, get formatted data
def base64_to_img(img_string):
    img = stringToImage(img_string)
    img0 = np.array(img)
    resizedImage = cv2.resize(img0, (28,28))
    resizedImage = resizedImage[:,:,3]
    rescaledImage = resizedImage*(16/255)
    rescaledImage = rescaledImage.reshape(-1)
    return rescaledImage

# load trained model (neural network - Sequential)
letter_model = load_model("C:/Users/44751/Google_Drive/Code/python/DrawingApp/letters_model.h5py")

# home page
@app.route("/", methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route("/_result")
def result():

    # when input go, from js, base64_string
    url = request.args.get('img_url', "",type=str)

    # initial decoding of base64_string
    data = url.replace("ONEPLUS","+").replace("ONEEQUALS","=").replace("ONESLASH","/")
    data = data.replace("ONEPLUS","+").replace("ONEEQUALS","=").replace("ONESLASH","/")

    # from decoded base64_string, get image data
    img = base64_to_img(data)
    imgs = np.array([img])

    # from image data, get model predeition
    pred = letter_model.predict(imgs)

    # from prediction, get result
    res = alph_res(pred[0])

    # return as JSON
    return jsonify(result=res)

app.run(debug=True, port=5000, host="0.0.0.0")
