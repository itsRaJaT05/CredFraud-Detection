from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import pickle
import os
from sklearn.impute import SimpleImputer

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load models
models = {
    "Logistic Regression": "models/logistic_regression_model.pkl",
    "Random Forest": "models/random_forest_model.pkl",
    "SVC": "models/svc_model.pkl",
    "KNN": "models/knn_model.pkl",
    "GBM": "models/gbm_model.pkl"
}

def load_model(model_path):
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except (OSError, IOError, pickle.UnpicklingError) as e:
        print(f"Error loading model: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        file = request.files['file']
        if not file:
            flash('No file selected!', 'danger')
            return redirect(url_for('index'))

        model_name = request.form['model']
        if model_name not in models:
            flash('Invalid model selected!', 'danger')
            return redirect(url_for('index'))

        # Save file to uploads folder
        filepath = os.path.join('uploads', file.filename)
        file.save(filepath)

        # Load and preprocess data
        data = pd.read_csv(filepath)
        features = data.drop(columns=['Class'])
        target = data['Class']

        # Handle missing values
        imputer = SimpleImputer(strategy='mean')
        features = imputer.fit_transform(features)

        # Load model and predict
        model = load_model(models[model_name])
        if model is None:
            flash('Error loading model!', 'danger')
            return redirect(url_for('index'))

        predictions = model.predict(features)
        
        # Calculate fraud transactions
        fraud_count = sum(predictions)
        
        return render_template('prediction.html', model_name=model_name, fraud_count=fraud_count, total_count=len(predictions))
    return redirect(url_for('index'))

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)
