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
    
    # Alternative model logging that should work with DagsHub
    try:
        # Try the standard way first
        mlflow.sklearn.log_model(dt, "model")
    except Exception as e:
        print(f"Standard model logging failed: {e}")
        # print("Trying alternative approach...")
        # Fallback: save model locally and log as artifact
        import time
        times = time.time()  # Ensure the model path is unique
        model_path = f"iris_model_{times}.pkl"
        mlflow.sklearn.save_model(dt, model_path)
        mlflow.log_artifact(model_path)
    
    # Add tags
    mlflow.set_tags({
        "project": "iris-classification",
        "framework": "scikit-learn",
        "model_type": "decision-tree"
    })
    
    print(f"Run completed successfully! Accuracy: {accuracy:.2f}")
    print(f"View your run at: https://dagshub.com/dhyanendra.manit/mlflow-dagshub-demo.mlflow")