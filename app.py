"""
Main flask app API to predict house prices
"""

import json

import pickle
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)

model = pickle.load(open("model.pk", "rb"))


def __preprocess(data: list) -> np.array:
    """Proccesess data before predicting"""
    X = np.asarray(data)
    # If there is only one set of data
    if X.shape == (13,):
        X = [X]
    return X


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        # Return docs
        return render_template("index.html")

    if request.method == "POST":
        # Load data
        try:
            data = json.loads(request.data)["data"]
            X = __preprocess(data)
        except:
            return json.dumps({"error": "Bad data format"}), 400
        # Make prediction
        try:
            prediction = model.predict(X)
            return json.dumps({"prediction": prediction.tolist()})
        except:
            return json.dumps({"error": "Prediction failed"}), 500
    return render_template("index.html")


if __name__ == "__main__":
    app.run()