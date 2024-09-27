from utils.utils import load_data, split_data, evaluate_model, get_scaler
from models.models import RandomForestModel, LogisticRegressionModel, MLTrainingPipeline
from icecream import ic

# Load the dataset (features and labels)
ic("Loading data...")
X, y = load_data()

# Split the data into training and testing sets
ic("Splitting data...")
X_train, X_test, y_train, y_test = split_data(X, y)

# Get a StandardScaler instance for preprocessing
ic("Getting the scaler...")
scaler = get_scaler()

# Using Dependency Injection with RandomForest model
ic("Training and testing the RandomForest model...")
random_forest_pipeline = MLTrainingPipeline(RandomForestModel(), scaler)  # Dependency Injection here
rf_predictions = random_forest_pipeline.run_pipeline(X_train, X_test, y_train)
ic(random_forest_pipeline, rf_predictions)

# Evaluate the RandomForest model
evaluate_model(y_test, rf_predictions)

# Using Dependency Injection with LogisticRegression model
ic("Training and testing the Logistic Regression model...")
logistic_regression_pipeline = MLTrainingPipeline(LogisticRegressionModel(), scaler)  # Dependency Injection here
lr_predictions = logistic_regression_pipeline.run_pipeline(X_train, X_test, y_train)
ic(logistic_regression_pipeline, lr_predictions)

# Evaluate the Logistic Regression model
evaluate_model(y_test, lr_predictions)

ic("Done.")
