from flask import Flask, render_template, request
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model (update the path as needed)
model = pickle.load(open('model.pkl', 'rb'))

# Example accuracy value
accuracy = 96.4

@app.route('/')
def home():
    return render_template('index.html', accuracy=accuracy)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract data from form and ensure they are floats
        features = []
        for feature in ['temperature', 'humidity', 'pm25', 'pm10', 'no2', 'so2', 'co', 'proximity', 'population']:
            value = request.form[feature]
            try:
                features.append(float(value))
            except ValueError:
                raise ValueError(f"Invalid value for {feature}: {value} is not a number.")

        # Convert to NumPy array for prediction
        data = np.array([features])
        
        # Get the prediction from the model
        prediction = model.predict(data)
        
        # Map prediction to air quality levels (without using the fixed list)
        if prediction[0] == 0:
            result = "Good"
        elif prediction[0] == 1:
            result = "Moderate"
        elif prediction[0] == 2:
            result = "Poor"
        elif prediction[0] == 3:
            result = "Hazardous"
        else:
            result = "Invalid Prediction"
        
        return render_template('index.html', accuracy=accuracy, prediction=result)
    
    except Exception as e:
        # Handle errors gracefully
        return render_template('index.html', accuracy=accuracy, error=str(e))


if __name__ == '__main__':
    app.run(debug=True)
