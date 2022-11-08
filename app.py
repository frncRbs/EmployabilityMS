from tkinter import Y
import numpy as np
import pandas as pd
import pickle
from pandas import DataFrame
from flask import Flask, request, jsonify, render_template
from sklearn.metrics import accuracy_score
import json
import plotly
import plotly.express as px

dataset = pd.read_csv("model/Institute-of-Computer-Studies-Graduate-Tracer-Study-2021-2022-Responses(ALTERED).csv")

IT_FEATURES = [
     'Sex',
     'Shiftee',
     'ComProg_1_1st',
     'ComProg_1_2nd',
     'ComProg_2_1st',
     'ComProg_2_2nd',
     'Intro_to_Computing_1st',
     'Intro_to_Computing_2nd',
     'Info_Management_1st',
     'Info_Management_2nd',
     'Operating_System_1st',
     'Operating_System_2nd',
     'Elective_1_1st',
     'Elective_1_2nd',
     'Elective_2_1st',
     'Elective_2_2nd',
     'Elective_3_1st',
     'Elective_3_2nd',
     'Data_Structures_1st',
     'Data_Structures_2nd',
     'Application_Dev_and_Emerging_Tech_1st',
     'Application_Dev_and_Emerging_Tech_2nd',
     'Human_and_Computer_Integration_1st',
     'Human_and_Computer_Integration_2nd',
     'Practicum_Industry_Immersion_1st',
     'Practicum_Industry_Immersion_2nd',
     'Integrative_Programming_and_Tech_1st',
     'Integrative_Programming_and_Tech_2nd',
     'System_Integration_and_Architecture_1st',
     'System_Integration_and_Architecture_2nd',
     'Information_Assurance_and_Security_1_1st',
     'Information_Assurance_and_Security_1_2nd',
     'Information_Assurance_and_Security_2_1st',
     'Information_Assurance_and_Security_2_2nd',
     'Software_Engineering_1st',
     'Software_Engineering_2nd',
     'Networking_1_1st',
     'Networking_1_2nd',
     'Networking_2_1st',
     'Networking_2_2nd',
     'WebProg_1st',
     'WebProg_2nd'
]
CS_FEATURES = [
     'Sex',
     'Shiftee',
     'ComProg_1_1st',
     'ComProg_1_2nd',
     'ComProg_2_1st',
     'ComProg_2_2nd',
     'Intro_to_Computing_1st',
     'Intro_to_Computing_2nd',
     'Info_Management_1st',
     'Info_Management_2nd',
     'Operating_System_1st',
     'Operating_System_2nd',
     'Elective_1_1st',
     'Elective_1_2nd',
     'Elective_2_1st',
     'Elective_2_2nd',
     'Elective_3_1st',
     'Elective_3_2nd',
     'Data_Structures_1st',
     'Data_Structures_2nd',
     'Application_Dev_and_Emerging_Tech_1st',
     'Application_Dev_and_Emerging_Tech_2nd',
     'Human_and_Computer_Integration_1st',
     'Human_and_Computer_Integration_2nd',
     'Practicum_Industry_Immersion_1st',
     'Practicum_Industry_Immersion_2nd',
     'Digital_Design_1st',
     'Digital_Design_2nd',
     'Architecture_and_Organization_1st',
     'Architecture_and_Organization_2nd',
     'Programming_Languages_1st',
     'Programming_Languages_2nd',
     'Modelling_and_Simulation_1st',
     'Modelling_and_Simulation_2nd',
     'Information_Assurance_and_Security_1st',
     'Information_Assurance_and_Security_1_2nd',
     'Software_Engineering_1_1st',
     'Software_Engineering_1_2nd',
     'Software_Engineering_2_1st',
     'Software_Engineering_2_2nd',
     'Network_Management_1st',
     'Network_Management_2nd',
     'Advance_Database_1st',
     'Advance_Database_2nd',
     'WebProg_1st',
     'WebProg_2nd'
]



TARGET = 'Suggested_job_role'

Cat_Y = dataset[TARGET]
X_IT = dataset[IT_FEATURES]
X_CS = dataset[CS_FEATURES]

X_IT = X_IT.replace(np.nan, 0)
X_CS = X_CS.replace(np.nan, 0)
percent = "%"
#   CREATE FLASK APP
flask_app = Flask(__name__)
#   MAIN JOB ROLE
model_IT = pickle.load(open("model/ProjectModel_IT.pkl", "rb"))
model_CS = pickle.load(open("model/ProjectModel_CS.pkl", "rb"))
#   SECONDARY JOB ROLES
model_CS_1 = pickle.load(open("model/ProjectModel_CS_1.pkl", "rb"))
model_IT_1 = pickle.load(open("model/ProjectModel_IT_1.pkl", "rb"))
model_IT_2 = pickle.load(open("model/ProjectModel_IT_2.pkl", "rb"))
model_CS_2 = pickle.load(open("model/ProjectModel_CS_2.pkl", "rb"))
model_IT_3 = pickle.load(open("model/ProjectModel_IT_3.pkl", "rb"))
model_CS_3 = pickle.load(open("model/ProjectModel_CS_3.pkl", "rb"))
#   TOP 5 COURSES SUGGESTION
model_ITsuggest = pickle.load(open("model/IT_SUGGESTEDcourse.pkl", "rb"))
model_CSsuggest = pickle.load(open("model/CS_SUGGESTEDcourse.pkl", "rb"))


@flask_app.route("/")
def Home():
    # Graph One
    # df = px.data.medals_wide()
    # dfTest = pd.read_csv(dfCon)
    # fig1 = px.bar(df, x="nation", y=["gold", "silver", "bronze"], title="Wide-Form Input")
    # fig1 = px.histogram(dfTest, y="humidity", title="Wide-Form Input")
    # graph1JSON = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)
    dfCon = "model/Institute-of-Computer-Studies-Graduate-Tracer-Study-2021-2022-Responses(ALTERED).csv"
    df = pd.read_csv(dfCon)
    
    # Graph six
    fig6 = px.pie(df['Sex'])
    graph6JSON = json.dumps(fig6, cls=plotly.utils.PlotlyJSONEncoder)
    
    # Graph one
    fig1 = px.bar(df, x="Suggested_job_role", y=["Shiftee", "Units"], title="Respondents from 2018 - 2022")
    graph1JSON = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)
    
    # Graph two
    dataIRIS = px.data.iris()
    fig2 = px.scatter_3d(df, x='Practicum_Industry_Immersion_1st', y='Data_Structures_1st', z='Operating_System_1st',
              color='Suggested_job_role',  title="Scatter Plot")
    graph2JSON = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)

    # Graph three
    fig3 = px.histogram(df, y="Degree_Completed", title="Respondents from 2018 - 2022")
    graph3JSON = json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder)
    
    # Graph four
    fig4 = px.bar(df, x="Sex", y=["Shiftee", "Units"], title="Wide-Form Input")
    graph4JSON = json.dumps(fig4, cls=plotly.utils.PlotlyJSONEncoder)
    
    # Graph five
    fig5 = px.bar(df, x="Suggested_job_role", y=["Curriculum", "Degree_Completed"], title="Curriculum and Degree Completed Frequency for Job Role")
    graph5JSON = json.dumps(fig5, cls=plotly.utils.PlotlyJSONEncoder)
    
    
    
    return render_template("index.html", graph1JSON=graph1JSON, graph2JSON=graph2JSON, graph3JSON=graph3JSON, graph4JSON=graph4JSON, graph5JSON=graph5JSON, graph6JSON=graph6JSON)

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
    new_Xdata_IT = X_IT.sample(1)
    new_Ydata_IT = Cat_Y[new_Xdata_IT.index.values]
    prediction = model_IT.predict(features)
    prediction1 = model_IT_1.predict(features)
    prediction2 = model_IT_2.predict(features)
    prediction3 = model_IT_3.predict(features)
    suggestIT = model_ITsuggest.predict(prediction)
    
    aScore = accuracy_score(new_Ydata_IT, prediction)
    aScore1 = accuracy_score(new_Ydata_IT, prediction1)
    aScore2 = accuracy_score(new_Ydata_IT, prediction2)
    aScore3 = accuracy_score(new_Ydata_IT, prediction3)
    return render_template("predictIT.html", prediction_text = "{}{}".format(prediction, " : {}{}".format(int(aScore), "00%")), 
                           prediction_text1 = "" if prediction == prediction1 or prediction1 == prediction2 or prediction1 == prediction3 else "{}{}".format(prediction1, " : {}{}".format(int(aScore1), "00%")), 
                           prediction_text2 = "" if prediction == prediction2 or prediction2 == prediction or prediction2 == prediction1 or prediction2 == prediction3 else "{}{}".format(prediction2, " : {}{}".format(int(aScore2), "00%")), 
                           prediction_text3 = "" if prediction == prediction3 or prediction3 == prediction or prediction3 == prediction1 or prediction3 == prediction2 else "{}{}".format(prediction3, " : {}{}".format(int(aScore3), "00%")), 
                           course_suggestion = "{}".format(suggestIT.tolist()) if aScore1 == 00 and aScore2 == 00 and aScore3 == 00 or prediction == "Administrative Assistant" else "",
                           course_suggestion1 = "{}".format(suggestIT.tolist()) if aScore1 != 00 and prediction1 == "Administrative Assistant" or aScore2 != 00 and prediction2 == "Administrative Assistant" or aScore3 != 00 and prediction3 == "Administrative Assistant" else "", 
                           showText = "~TOP 5 COURSES NEED TO IMPROVE" if aScore1 == 00 and aScore2 == 00 and aScore3 == 00 or prediction == "Administrative Assistant" else "",
                           showText1 = "TOP 5 COURSES NEED TO IMPROVE~" if aScore1 != 00 and prediction1 == "Administrative Assistant" or aScore2 != 00 and prediction2 == "Administrative Assistant" or aScore3 != 00 and prediction3 == "Administrative Assistant" else "",  
                           showLowProba = "SECONDARY JOB ROLES", 
                           showHiProba = "MAIN JOB ROLE") 

@flask_app.route("/predict_CS", methods = ["POST"])
def predict_CS():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    new_Xdata_CS = X_CS.sample(1)
    new_Ydata_CS = Cat_Y[new_Xdata_CS.index.values]
    prediction = model_CS.predict(features)
    prediction1 = model_CS_1.predict(features)
    prediction2 = model_CS_2.predict(features)
    prediction3 = model_CS_3.predict(features)
    suggestCS = model_CSsuggest.predict(prediction)
    #127.0.0.1:5000
    
    aScore = accuracy_score(new_Ydata_CS, prediction)
    aScore1 = accuracy_score(new_Ydata_CS, prediction1)
    aScore2 = accuracy_score(new_Ydata_CS, prediction2)
    aScore3 = accuracy_score(new_Ydata_CS, prediction3)
    return render_template("predictCS.html", prediction_text = "{}{}".format(prediction, " : {}{}".format(int(aScore), "00%")), 
                           prediction_text1 = "" if prediction == prediction1 or prediction1 == prediction2 or prediction1 == prediction3 else "{}{}".format(prediction1, " : {}{}".format(int(aScore1), "00%")), 
                           prediction_text2 = "" if prediction == prediction2 or prediction2 == prediction or prediction2 == prediction1 or prediction2 == prediction3 else "{}{}".format(prediction2, " : {}{}".format(int(aScore2), "00%")), 
                           prediction_text3 = "" if prediction == prediction3 or prediction3 == prediction or prediction3 == prediction1 or prediction3 == prediction2 else "{}{}".format(prediction3, " : {}{}".format(int(aScore3), "00%")), 
                           course_suggestion = "{}".format(suggestCS.tolist()) if aScore1 == 00 and aScore2 == 00 and aScore3 == 00 or prediction == "Administrative Assistant" else "",
                           course_suggestion1 = "{}".format(suggestCS.tolist()) if aScore1 != 00 and prediction1 == "Administrative Assistant" or aScore2 != 00 and prediction2 == "Administrative Assistant" or aScore3 != 00 and prediction3 == "Administrative Assistant" else "", 
                           showText = "~TOP 5 COURSES NEED TO IMPROVE" if aScore1 == 00 and aScore2 == 00 and aScore3 == 00 or prediction == "Administrative Assistant" else "",
                           showText1 = "TOP 5 COURSES NEED TO IMPROVE~" if aScore1 != 00 and prediction1 == "Administrative Assistant" or aScore2 != 00 and prediction2 == "Administrative Assistant" or aScore3 != 00 and prediction3 == "Administrative Assistant" else "",  
                           showLowProba = "SECONDARY JOB ROLES", 
                           showHiProba = "MAIN JOB ROLE") 

if __name__ == "__main__":
    flask_app.run(debug=True)
