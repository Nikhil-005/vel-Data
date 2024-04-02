from flask import Flask,render_template,request
import pickle

import numpy as np
model=pickle.load(open("model.pkl",'rb'))
app=Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def diabeties_prediction():
    pregnencies=int(request.form.get("pregnencies"))
    Glucose=float(request.form.get("Glucose"))
    BloodPressure=int(request.form.get("BloodPressure"))
    SkinThickness=int(request.form.get("SkinThickness"))
    Insulin=int(request.form.get("Insulin"))
    BMI=int(request.form.get("BMI"))
    DiabetesPedigreeFunction=int(request.form.get("DiabetesPedigreeFunction"))
    Age=int(request.form.get("Age"))

    result=model.predict(np.array([[pregnencies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]]).reshape(1,8))

    return str(result)

if __name__ =="__main__":
    app.run(debug=True)
