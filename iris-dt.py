# logging an MLflow model with DagsHub integration
# This script demonstrates how to log a Decision Tree model using MLflow with DagsHub integration.
# It includes hyperparameter tuning, model evaluation, and logging of artifacts and datasets.
# It also logs the confusion matrix and model artifacts to DagsHub.
# The script uses the Iris dataset for classification.
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import dagshub
import os

# Initialize DagsHub with explicit MLflow tracking
dagshub.init(
    repo_owner='dhyanendra.manit',
    repo_name='mlflow-dagshub-demo',
    mlflow=True
)

# Set the tracking URI explicitly (sometimes needed)
mlflow.set_tracking_uri('https://dagshub.com/dhyanendra.manit/mlflow-dagshub-demo.mlflow')

# Load data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Set experiment (create if doesn't exist)
experiment_name = "iris-dt"
if not mlflow.get_experiment_by_name(experiment_name):
    mlflow.create_experiment(experiment_name)
mlflow.set_experiment(experiment_name)

# Start run
with mlflow.start_run(run_name="decision-tree-run"):
    # Model parameters
    params = {
        "max_depth": 10,
        "random_state": 42
    }
    
    # Create and train model
    dt = DecisionTreeClassifier(**params)
    dt.fit(X_train, y_train)
    
    # Evaluate
    y_pred = dt.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Log parameters and metrics
    mlflow.log_params(params)
    mlflow.log_metric("accuracy", accuracy)
    
    # Create and save confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred),
                annot=True, fmt='d',
                xticklabels=iris.target_names,
                yticklabels=iris.target_names)
    plt.title("Confusion Matrix")
    confusion_matrix_path = "confusion_matrix.png"
    plt.savefig(confusion_matrix_path)
    plt.close()
    
    # Log artifacts
    mlflow.log_artifact(confusion_matrix_path)
    mlflow.log_artifact(__file__)
    
    # Model logging compatible with DagsHub
    run_id = mlflow.active_run().info.run_id
    model_dir = f"model_{run_id}"
    mlflow.sklearn.save_model(dt, model_dir)
    mlflow.log_artifacts(model_dir, artifact_path="model")
    
    # Add tags
    mlflow.set_tags({
        "project": "iris-classification",
        "framework": "scikit-learn",
        "model_type": "decision-tree"
    })
    
    print(f"Run completed successfully! Accuracy: {accuracy:.2f}")
    print(f"View your run at: https://dagshub.com/dhyanendra.manit/mlflow-dagshub-demo.mlflow")