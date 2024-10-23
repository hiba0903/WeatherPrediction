import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the encoders, scaler, and model
cloud_cover_encoder = joblib.load('cloud_cover_encoder.pkl')
season_encoder = joblib.load('season_encoder.pkl')
location_encoder = joblib.load('location_encoder.pkl')
scaler = joblib.load('scaler.pkl')
model = joblib.load('model.pkl')

app = Flask(__name__)

# Mapping prediction numbers to weather conditions
weather_conditions = {
    0: 'Cloudy',
    1: 'Rainy',
    2: 'Snowy',
    3: 'Sunny'
}

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None  # Initialize prediction variable
    weather_prediction = None  # Initialize weather prediction variable
    if request.method == 'POST':
        # Get user input
        cloud_cover = request.form.get('cloud_cover')
        season = request.form.get('season')
        location = request.form.get('location')
        temperature = float(request.form.get('temperature'))
        humidity = float(request.form.get('humidity'))
        wind_speed = float(request.form.get('wind_speed'))
        precipitation = float(request.form.get('precipitation'))
        atmospheric_pressure = float(request.form.get('atmospheric_pressure'))
        uv_index = float(request.form.get('uv_index'))
        visibility = float(request.form.get('visibility'))

        # Encode categorical variables
        cloud_cover_encoded = cloud_cover_encoder.transform([cloud_cover])[0]
        season_encoded = season_encoder.transform([season])[0]
        location_encoded = location_encoder.transform([location])[0]

        # Create array for input features
        input_features = np.array([[temperature, humidity, wind_speed, 
                                    precipitation, atmospheric_pressure, 
                                    uv_index, visibility, 
                                    cloud_cover_encoded, season_encoded, 
                                    location_encoded]])

        # Scale numerical features
        numerical_data_scaled = scaler.transform(input_features)

        # Make prediction
        prediction = model.predict(numerical_data_scaled)
        prediction = prediction[0]  # Extract the single prediction

        # Get the corresponding weather condition
        weather_prediction = weather_conditions.get(prediction, 'Unknown')

    return render_template('index.html', prediction=weather_prediction)

if __name__ == '__main__':
    app.run(debug=True)
