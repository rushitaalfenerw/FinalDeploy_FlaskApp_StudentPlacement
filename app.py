from flask import Flask, request, render_template
import pickle
import pandas as pd
import socket
import pandas as pd


app = Flask(__name__)
df = pd.read_csv("global_student_migration_New_cleaned.txt")

# Load trained pipeline
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route('/')
def home():
   # Get unique test types from your dataset
    test_options = df['language_proficiency_test'].dropna().unique().tolist()
    return render_template('index.html', test_options=test_options)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        gpa_or_score = float(request.form['gpa_or_score'])
        language_proficiency_test = request.form['language_proficiency_test']
        test_score = float(request.form['test_score'])

        # Prepare DataFrame
        input_df = pd.DataFrame({
            'gpa_or_score': [gpa_or_score],
            'language_proficiency_test': [language_proficiency_test],
            'test_score': [test_score]
        })

        prediction = model.predict(input_df)[0]

        return render_template('index.html', prediction_text=f"Predicted Placement Status: {prediction}")
    except Exception as e:
        return str(e)

def find_free_port(start_port=5000, max_port=5100):
    """Finds a free port between start_port and max_port"""
    for port in range(start_port, max_port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('127.0.0.1', port)) != 0:
                return port
    return start_port  # fallback

if __name__ == '__main__':
    port = find_free_port()
    print(f"Starting Flask app on http://127.0.0.1:{port}")
    # Set debug=True for development; set to False in production
    app.run(debug=True, port=port, use_reloader=False)