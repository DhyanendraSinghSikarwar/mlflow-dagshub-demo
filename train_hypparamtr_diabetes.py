from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import mlflow

df = pd.read_csv("diabetes.csv")

# Split the data into features and target variable
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
model = RandomForestClassifier(random_state=42)

# Define the hyperparameter grid
param_grid = {
    'n_estimators': [10, 50, 100],
    'max_depth': [None, 10, 20, 30],
    # 'min_samples_split': [2, 5, 10],
    # 'min_samples_leaf': [1, 2, 4]
}
# Perform grid search with cross-validation
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# mlflow.set_tracking_uri('https://dagshub.com/dhyanendra.manit/mlflow-dagshub-demo.mlflow')
mlflow.set_experiment("diabetes-rf")
with mlflow.start_run():
    # Fit the model
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    # Log the best parameters
    mlflow.log_params(best_params)
    

    # Log the metrics
    mlflow.log_metric("best_accuracy", best_score)

    # Log the data
    train_df = X_train
    train_df['Outcome'] = y_train
    train_df = mlflow.data.from_pandas(train_df)
    mlflow.log_input(train_df, "training")

    test_df = X_test
    test_df['Outcome'] = y_test
    test_df = mlflow.data.from_pandas(test_df)
    mlflow.log_input(test_df, "Validation")


    # Log the source code
    mlflow.log_artifact(__file__)
    
    # Log the model
    mlflow.sklearn.log_model(grid_search.best_estimator_, "Random Forest")

    # set tag
    mlflow.set_tag("author", "Dhyanendra")

    mlflow.log_params(best_params)
    mlflow.log_metric("best_score", best_score)

    print("Best parameters found: ", best_params)
    print("Best cross-validation score: ", best_score)

