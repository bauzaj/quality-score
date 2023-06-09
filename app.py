from flask import Flask, render_template, request
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load your model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collect all form inputs in a list
    features = [float(x) for x in request.form.values()]
    
    # Convert features to array and reshape
    array_features = np.array(features).reshape(1, -1)
    
    # Apply the scaler to array_features before prediction
    array_features = scaler.transform(array_features)
    
    # Predict
    prediction = model.predict(array_features)
    
    output = round(prediction[0],2)
    
    return render_template('results.html', prediction_text='predicted quality score: {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)

