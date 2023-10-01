import numpy as np
import pandas as pd
from flask import Flask,request,render_template
import joblib

app = Flask(__name__, template_folder ='templates',static_url_path='/static')
model = joblib.load(r"C:\Users\Lenovo\OneDrive\Desktop\heart_disea_prediction\heart.pkl")
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Retrieve data from the HTML form
        age = int(request.form['age'])
        sex = int(request.form['sex'])
        chest_pain = int(request.form['chestpain'])
        resting_bp = int(request.form['restingbp'])
        cholesterol = int(request.form['cholesterol'])
        fasting_bs = int(request.form['fastingbs'])
        resting_ecg = int(request.form['ratingecg'])
        max_hr = int(request.form['maxhr'])
        exercise_angina = int(request.form['exerciseangina'])
        old_peak = float(request.form['oldpeak'])
        st_slope = int(request.form['stslope'])

        # Create a list of input features
        inp_features = [age, sex, chest_pain, resting_bp, cholesterol, fasting_bs,
                        resting_ecg, max_hr, exercise_angina, old_peak, st_slope]

        # Make a prediction using the loaded model
        prediction = model.predict([inp_features])

        # Prepare the prediction result message
        if prediction[0] == 1:
            result = "Heart Disease Detected"
        else:
            result = "No Heart Disease Detected"
        
        return render_template('index.html', prediction_result=result)  
@app.route('/result/<prediction_result>')
def result(prediction_result):
    return render_template('result.html', prediction_result=prediction_result)
      
 
if __name__ == '__main__':
    app.run(debug = True)

