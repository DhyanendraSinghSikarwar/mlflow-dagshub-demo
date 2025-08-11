# Auto logging an MLflow model with DagsHub

import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import dagshub
import os
import pandas as pd

# Initialize DagsHub with explicit MLflow tracking
dagshub.init(
    repo_owner='dhyanendra.manit',
    repo_name='mlflow-dagshub-demo',
    mlflow=True
)

# Set the tracking URI explicitly
mlflow.set_tracking_uri('https://dagshub.com/dhyanendra.manit/mlflow-dagshub-demo.mlflow')

# Load data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# Set experiment (create if doesn't exist)
experiment_name = "iris-rf"
if not mlflow.get_experiment_by_name(experiment_name):
    mlflow.create_experiment(experiment_name)
mlflow.set_experiment(experiment_name)

mlflow.autolog()  # Enable autologging
# Start run
with mlflow.start_run():
    # Model parameters
    params = {
        "max_depth": 5,
        "n_estimators": 100
    }
    
    # Create and train model
    rf = RandomForestClassifier(**params)
    rf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Create and save confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        confusion_matrix(y_test, y_pred),
        annot=True, fmt='d',
        xticklabels=iris.target_names,
        yticklabels=iris.target_names
    )
    plt.title("Confusion Matrix")
    confusion_matrix_path = "confusion_matrix.png"
    plt.savefig(confusion_matrix_path)
    plt.close()
    
    # Log artifacts
   # mlflow.log_artifact(__file__)

    # Add tags
    mlflow.set_tags({
        "author": "dhyanendra",
        "project": "iris-classification",
        "framework": "scikit-learn",
        "model_type": "random-forest",
    })


    print(f"Run completed successfully! Accuracy: {accuracy:.2f}")
    print("View your run at: https://dagshub.com/dhyanendra.manit/mlflow-dagshub-demo.mlflow")
