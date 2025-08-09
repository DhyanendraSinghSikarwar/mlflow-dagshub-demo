import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Set the MLflow tracking URI to log artifacts
mlflow.set_tracking_uri("http://localhost:5000")

# Load the iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameters for the Decision Tree model
max_depth = 4

# apply MLflow tracking
mlflow.set_experiment("iris-dt") # this will create a new experiment if it doesn't exist
# or with mlflow.start_run(experiment_id="595542530452145360", run_name="DecisionTree-Iris-experiment"): 
with mlflow.start_run():
    # Train the Decision Tree model
    model = DecisionTreeClassifier(max_depth=max_depth)
    model.fit(X_train, y_train)
    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate accuracy and confusion matrix
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    # Log parameters, metrics, and model
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_metric("accuracy", accuracy)
    # mlflow.sklearn.log_model(model, "model")

    # Create a confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')

    # Save the plot as an artifact
    plt.savefig("confusion_matrix.png")


    mlflow.log_artifact("confusion_matrix.png")
    mlflow.log_artifact(__file__)
    mlflow.sklearn.log_model(model, "Decision Tree Model")

    mlflow.set_tag("model_type", "Decision Tree")
    mlflow.set_tag("author", "Dhyanendra")

    print(f"Accuracy: {accuracy:.2f}")