import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create flask app
flask_app = Flask(__name__)
model_IT = pickle.load(open("ProjectModel_IT.pkl", "rb"))
model_CS = pickle.load(open("ProjectModel_CS.pkl", "rb"))

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predictCS")
def CS_view():
    return render_template("predictCS.html")

@flask_app.route("/predictIT")
def IT_view():
    return render_template("predictIT.html")

@flask_app.route("/predict_IT", methods = ["POST"])
def predict_IT():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model_IT.predict(features)
    return render_template("predictIT.html", prediction_text = "You will land your 1st job as members of the {}".format(prediction) if prediction == "Software Engineer / Programmer" or prediction == "Academician" or prediction == "Technical Support Specialist" else "Sad to say your first job is not related to IT/CS.") 

@flask_app.route("/predict_CS", methods = ["POST"])
def predict_CS():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model_CS.predict(features)
    return render_template("predictCS.html", prediction_text = "You will land your 1st job as members of the {}".format(prediction) if prediction == "Software Engineer / Programmer" or prediction == "Academician" or prediction == "Technical Support Specialist" else "Sad to say your first job is not related to IT/CS.") 

if __name__ == "__main__":
    flask_app.run(debug=True)