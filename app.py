from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.prediction_pipeline import CustomData, PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata', methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data = CustomData(
            Pregnancies = float(request.form.get('Pregnancies')),
            Glucose=float(request.form.get('Glucose')),
            BloodPressure=float(request.form.get('BloodPressure')),
            SkinThickness = float(request.form.get('SkinThickness')),
            Insulin = float(request.form.get('Insulin')),
            BMI = float(request.form.get('BMI')),
            DiabetesPedigreeFunction = float(request.form.get('DiabetesPedigreeFunction')),
            Age = float(request.form.get('Age'))

        )
        
        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")
        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results, proba = predict_pipeline.predict(pred_df)
        print("after Prediction")
        print(predict_pipeline.predict(pred_df))
        if results == 1:
            return render_template('home.html', results=f'Positive to Diabete with probability {proba}')
        else:
            return render_template('home.html', results=f'Negtaive to Diabete with probability {proba}')
    

if __name__=="__main__":
    app.run(host="0.0.0.0")        

