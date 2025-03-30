from flask import Flask, request, jsonify, render_template, url_for
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import numpy as np
import pandas as pd
import requests
from io import StringIO
import os

app = Flask(__name__, 
    static_url_path='',
    static_folder='static',
    template_folder='templates')

def load_default_data():
    """Load and return the Iris dataset as a default option"""
    try:
        data = load_iris()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target, name='target')
        
        # target names
        target_names = pd.Series(data.target).map(dict(enumerate(data.target_names)))
        y = pd.Series(target_names, name='target')
        
        return X, y
    except Exception as e:
        raise ValueError(f"Error loading default dataset: {str(e)}")

#data
def load_data():
    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    return X, y

def load_data_from_file(file):
    if file.filename.endswith('.csv'):
        df = pd.read_csv(file)
    elif file.filename.endswith('.xlsx'):
        df = pd.read_excel(file)
    else:
        raise ValueError("Unsupported file format")
    
    # target label
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return X, y

def load_data_from_url(url):
    try:
        response = requests.get(url)
        if url.endswith('.csv'):
            df = pd.read_csv(StringIO(response.text))
        elif url.endswith('.xlsx'):
            df = pd.read_excel(response.content)
        else:
            raise ValueError("Unsupported file format")
        
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        return X, y
    except Exception as e:
        raise ValueError(f"Error loading data from URL: {str(e)}")

#ppt
def preprocess_data(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

#fs/e
def feature_selection(X):
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)
    return X_reduced

#model selection
def get_model(algorithm):
    if algorithm == "logistic_regression":
        return LogisticRegression()
    elif algorithm == "decision_tree":
        return DecisionTreeClassifier()
    elif algorithm == "random_forest":
        return RandomForestClassifier()
    elif algorithm == "svm":
        return SVC()
    elif algorithm == "knn":
        return KNeighborsClassifier()

# train and eval 
def train_and_evaluate_model(algorithm, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = get_model(algorithm)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    #calc metrics
    metrics_data = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, average='weighted'), 4),
        "recall": round(recall_score(y_test, y_pred, average='weighted'), 4),
        "f1_score": round(f1_score(y_test, y_pred, average='weighted'), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }
    return metrics_data

#flask routes
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/metrics', methods=['POST'])
def metrics():
    try:
        algorithm = request.form.get('algorithm')
        if not algorithm:
            raise ValueError("Algorithm not specified")
        
        # file upload
        if 'file' in request.files and request.files['file'].filename != '':
            file = request.files['file']
            if not file.filename.endswith(('.csv', '.xlsx')):
                raise ValueError("Unsupported file format. Please use CSV or XLSX files.")
            X, y = load_data_from_file(file)
        # urL import
        elif request.form.get('url'):
            url = request.form.get('url')
            X, y = load_data_from_url(url)
        # exception for no dataset
        else:
            X, y = load_default_data()
        
        # data format validation
        if not isinstance(X, (pd.DataFrame, np.ndarray)) or not isinstance(y, (pd.Series, np.ndarray)):
            raise ValueError("Invalid data format")
            
        X = preprocess_data(X)
        X = feature_selection(X)
        metrics_data = train_and_evaluate_model(algorithm, X, y)
        
        response_data = {
            "success": True,
            "metrics": metrics_data,
            "dataset_info": {
                "n_samples": len(X),
                "n_features": X.shape[1],
                "n_classes": len(np.unique(y)),
                "is_default": isinstance(X, np.ndarray) and X.shape[1] == 2  # After PCA
            }
        }
        
        return jsonify(response_data)
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400

if __name__ == '__main__':
    app.run(debug=True)