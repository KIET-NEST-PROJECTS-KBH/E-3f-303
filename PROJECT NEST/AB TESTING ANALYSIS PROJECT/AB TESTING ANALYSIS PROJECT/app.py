import numpy as np
import pandas as pd
from flask import Flask, render_template, request
import pickle

app = Flask(__name__,template_folder='templates')
model = pickle.load(open('model_pkl', 'rb'))
@app.route('/')
def home():
    return render_template('home.html')
@app.route('/about.html')
def about():
    return render_template('about.html')
@app.route('/index.html')
def index():
    return render_template('index.html')

@app.route('/variantA.html')
def variant_A():
    return render_template('variantA.html')
@app.route('/variantB.html')
def variant_B():
    return render_template('variantB.html')
@app.route('/result.html')
def result():
    return render_template('result.html')


if __name__ == '__main__':
    app.run(debug=True)

