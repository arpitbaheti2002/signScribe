from flask import Flask, render_template
from flask import request
import os
import wget

from recognition import sign_recognition
from collection import image_collection
from training import train_model

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recognize', methods=['POST'])
def recognition():
    data = request.get_json()
    lang = data.get('language')
    sentence = sign_recognition(lang)
    return sentence
    return lang

@app.route('/collect', methods=['POST'])
def collection():
    data = request.get_json()
    action = data.get('action')
    sentence = image_collection(action)
    return sentence

@app.route('/train')
def train():
    sentence = train_model()
    return sentence


if __name__ == '__main__':
    app.run(debug=True)

