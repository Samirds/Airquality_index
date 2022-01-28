# ================================== IMPORT ===================================================>
import pandas as pd
import combine_dataset
import pickle
from datetime import datetime
from flask import Flask, render_template, request, send_file
import requests
import numpy as np
import os

# =====================================  Initialize           ================================>

app = Flask(__name__)
model = pickle.load(open('AirQuality.pkl', 'rb'))

# =====================================  Pic Upload           ==============================

picfolder = os.path.join('static', 'pics')
app.config['UPLOAD_FOLDER'] = picfolder


# ================================= Routes ============================================================>

@app.route('/', methods=['GET'])
def home():
    picture = os.path.join(app.config['UPLOAD_FOLDER'], 'airquality_img.jpeg')
    return render_template('AIrQuality_index.html')


@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == "POST":
        # Date_of_Prediction
        form_date = request.form["date"]
        end_of_tr_date = datetime.strptime("2020-04-01", "%Y-%m-%d")
        date = datetime.strptime(form_date, "%Y-%m-%d")
        period_time = abs(date - end_of_tr_date).days

        future = model.make_future_dataframe(periods=period_time)
        future['PM2.5'] = combine_dataset.final_df['PM2.5']
        future['PM10'] = combine_dataset.final_df['PM10']
        future['NOx'] = combine_dataset.final_df['NOx']
        future['temp_st'] = combine_dataset.final_df['temp_st']

        forecast = model.predict(future)
        index = forecast[forecast["ds"] == form_date]["yhat"].values[0]

        # ============================ Category ===========================================
        category = []

        if index < 41:
            category.append("Good")

        elif (index >= 41) & (index < 84):
            category.append("Satisfactory")

        elif (index >= 84) & (index < 148):
            category.append("Moderate")

        elif (index >= 148) & (index < 251):
            category.append("Poor")

        elif (index >= 251) & (index < 347):
            category.append("Very Poor")

        elif index >= 347:
            category.append("Severe")

        return render_template("AIrQuality_index.html", date_given=form_date, predict_index=round(index, 2),
                               predict_category=category[0])

    return render_template("AIrQuality_index.html")


if __name__ == "__main__":
    app.run(debug=True)
