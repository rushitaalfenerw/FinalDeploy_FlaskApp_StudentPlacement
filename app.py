from flask import Flask, request, render_template
import pickle
import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("global_student_migration_New_cleaned.csv")

# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def home():
    test_options = df['language_proficiency_test'].dropna().unique().tolist()
    return render_template('index.html', test_options=test_options)

@app.route('/predict', methods=['POST'])
def predict():
    gpa_or_score = float(request.form['gpa_or_score'])
    language_proficiency_test = request.form['language_proficiency_test']
    test_score = float(request.form['test_score'])
    input_df = pd.DataFrame({
        'gpa_or_score': [gpa_or_score],
        'language_proficiency_test': [language_proficiency_test],
        'test_score': [test_score]
    })
    prediction = model.predict(input_df)[0]
    return render_template('index.html', prediction_text=f"Predicted Placement Status: {prediction}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

