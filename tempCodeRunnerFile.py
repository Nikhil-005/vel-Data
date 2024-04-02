from flask import Flask,render_template,request
import pickle

import numpy as np

app=Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def diabeties_prediction():
    data=request.form
    print(data)
    model=pickle.load(open("model.pkl",'rb'))
    print(model)

    user_data=[[float(data["pregnencies"],
                float(["Glucose"])),
                float(["BloodPressure"]),
                float(["SkinThickness"]),
                float(["Insulin"]),
                float(["BMI"]),
                float(["DiabetesPedigreeFunction"]),
                float(["Age"])]]
    
    print(user_data)

    result = model.predict(user_data)
    print(result)
    return target[result[0]]
    

if __name__ =="__main__":
    app.run(debug=True)
