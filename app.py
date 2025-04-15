from flask import Flask,render_template,request
import os
import numpy as np
import pandas as pd
from src.datascience.pipeline.prediction_pipeline import PredictionPipeline

app = Flask(__name__)


@app.route('/',methods=['GET']) ## route to display the home page
def homepage():
    return render_template("index.html") # index.html should be created inside template folder

@app.route('/train',methods=['GET']) # route to train the pipeline
def training():
    os.system("python main.py")
    return "Training Succesful"

@app.route('/predict', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        try:
            sequence_data = []

            for i in range(10):
                LATITUDE = float(request.form[f'LATITUDE{i}'])
                LONGITUDE = float(request.form[f'LONGITUDE{i}'])
                ELEVATION = float(request.form[f'ELEVATION{i}'])
                TAVG = float(request.form[f'TAVG{i}'])
                YEAR = float(request.form[f'YEAR{i}'])
                MONTH = float(request.form[f'MONTH{i}'])
                DAY = float(request.form[f'DAY{i}'])
                DAY_OF_WEEK = float(request.form[f'DAY_OF_WEEK{i}'])
                DAY_OF_YEAR = float(request.form[f'DAY_OF_YEAR{i}'])

                row = [LATITUDE, LONGITUDE, ELEVATION, TAVG, YEAR, MONTH, DAY, DAY_OF_WEEK, DAY_OF_YEAR]
                sequence_data.append(row)

            sequence_data = np.array(sequence_data).reshape(1, 10, 9)  # (1, time_steps, features)

            obj = PredictionPipeline()
            predict = obj.predict(sequence_data)

            return render_template('results.html', prediction=str(predict))

        except Exception as e:
            print('The Exception message is:', e)
            return 'Something went wrong'

    else:
        return render_template('index.html')



if __name__ == "__main__":
	
	app.run(host="0.0.0.0", port = 8080)