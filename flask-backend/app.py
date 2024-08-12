from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        iris = load_iris()
        df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
        df.to_csv('iris.csv', index=False)
        df = pd.read_csv('iris.csv')
        result = analyze_data(df)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def analyze_data(df):
    # Basic analysis
    description = replace_invalid_json_values(df.describe(include='all').to_dict())

    # Basic Machine Learning Model
    if 'species' in df.columns:
        X = df.drop('species', axis=1)
        y = df['species']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = LogisticRegression(max_iter=200)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        return {
            "description": description,
            "accuracy": accuracy,
            "classification_report": report
        }
    return {"description": description}

def replace_invalid_json_values(obj):
    if isinstance(obj, dict):
        return {k: replace_invalid_json_values(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [replace_invalid_json_values(i) for i in obj]
    elif isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj

if __name__ == '__main__':
    app.run(debug=True)
