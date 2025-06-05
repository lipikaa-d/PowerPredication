import pandas as pd
import numpy as np
from flask import Flask, render_template, request
import joblib
import os
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Load trained model
model = joblib.load('model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get inputs from form
        wind_speed = float(request.form['wind_speed'])
        wind_direction = float(request.form['wind_direction'])

        # Prepare input for model
        X_input = np.array([[wind_speed, wind_direction]])
        prediction = model.predict(X_input)

        # Extract predictions
        pred_lv = prediction[0][0]
        pred_theory = prediction[0][1]
        abs_error = abs(pred_theory - pred_lv)

        # Log prediction to CSV
        row = {
            'Wind Speed': wind_speed,
            'Wind Direction': wind_direction,
            'Predicted LV Active Power': pred_lv,
            'Predicted Theoretical Power Curve': pred_theory,
            'Absolute Error': abs_error
        }

        log_file = 'prediction_log.csv'
        file_exists = os.path.isfile(log_file)
        pd.DataFrame([row]).to_csv(log_file, mode='a', header=not file_exists, index=False)

        # Generate chart
        fig, ax = plt.subplots()
        labels = ['LV Power', 'Theoretical Power']
        values = [pred_lv, pred_theory]

        ax.bar(labels, values, color=['skyblue', 'orange'])
        ax.set_ylabel('Power (kW)')
        ax.set_title('Predicted LV vs Theoretical Power')
        plt.tight_layout()

        # Convert plot to base64 string
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()

        return render_template('result.html',
                               wind_speed=wind_speed,
                               wind_direction=wind_direction,
                               pred_lv=round(pred_lv, 2),
                               pred_theory=round(pred_theory, 2),
                               abs_error=round(abs_error, 2),
                               plot_url=plot_url)

    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)
