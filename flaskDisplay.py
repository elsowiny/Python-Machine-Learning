from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn import linear_model
import sklearn
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

app = Flask(__name__)

# add urls for navigation as a dict
urls = {

    "linear regression": "/lr",
    "knn": "/knn",
    "svm": "/svm",
    "k means": "/kmeans"
}


@app.route('/')
def home():
    return render_template('home.html', data=urls.items(), name="home")


@app.route('/lr')
def linear_regression():
    return render_template('LR.html', data=urls.items(), name="Linear Regression")


@app.route('/knn')
def knn():
    return render_template('KNN.html', data=urls.items(), name="K Nearest Neighbors")


@app.route('/svm')
def svm():
    return render_template('SVM.html', data=urls.items(), name="Support Vector Machines")


@app.route('/kmeans')
def kmeans():
    return render_template('KMEANS.html', data=urls.items(), name="K Means")


"""
 ML ALGOS
"""


@app.route('/lr/predict', methods=['POST'])
def linear_regression_results():
    data = pd.read_csv("/linearRegression/student-mat.csv", sep=";")
    print(data.head())
    data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]  # our attributes
    return render_template('LR.html', data=urls.items(), name="Linear Regression")


if __name__ == "__main__":
    app.run(debug=True)
