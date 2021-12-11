import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle


# create flask app
app = Flask(__name__)

# Load the pickle model
model = pickle.load(open("model.pkl", "rb"))
@app.route("/")
def index():
    return render_template('index.html')
@app.route("/predict", methods = ["POST","GET"])
def predict():
    output = request.form.to_dict()
    local = output["local"]
    type = output["type"]
    surface = output["surface"]
    nbpiece = output["nbpiece"]
    data = [{
        "local": local,
        "type" : type,
        "surface" : surface,
        "nbpiece" : nbpiece
    }]
    #json_ = request.json
    query_df = pd.DataFrame(data)
    prediction = model.predict(query_df)
    prix = int(prediction)
    return render_template("index.html",prix = prix)

if __name__ == "__main__":
    app.run(debug=True)




