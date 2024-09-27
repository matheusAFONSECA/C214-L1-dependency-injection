from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


# Model interface
class BaseModel(ABC):
    """Abstract base class for machine learning models"""

    @abstractmethod
    def train(self, X_train, y_train):
        """Train the model on the training data"""
        pass

    @abstractmethod
    def predict(self, X_test):
        """Make predictions using the trained model"""
        pass


# Random Forest implementation
class RandomForestModel(BaseModel):
    """RandomForest model that implements the BaseModel interface"""

    def __init__(self):
        # Initialize RandomForestClassifier from scikit-learn
        self.model = RandomForestClassifier()

    def train(self, X_train, y_train):
        # Train the RandomForest model on the training data
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        # Predict the output for the test data
        return self.model.predict(X_test)


# Logistic Regression implementation
class LogisticRegressionModel(BaseModel):
    """LogisticRegression model that implements the BaseModel interface"""

    def __init__(self):
        # Initialize LogisticRegression from scikit-learn
        self.model = LogisticRegression()

    def train(self, X_train, y_train):
        # Train the Logistic Regression model on the training data
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        # Predict the output for the test data
        return self.model.predict(X_test)


# Machine Learning pipeline class
class MLTrainingPipeline:
    """Pipeline for training and testing machine learning models"""

    def __init__(self, model: BaseModel, scaler):
        """
        Initialize the pipeline with a model and a scaler.
        
        Dependency Injection: The model (RandomForest or LogisticRegression)
        and scaler (StandardScaler) are injected via the constructor.
        """
        self.model = model  # Dependency Injection for the model
        self.scaler = scaler  # Dependency Injection for the scaler

    def run_pipeline(self, X_train, X_test, y_train):
        """
        Execute the pipeline steps: scaling, training, and predicting.
        
        :param X_train: Training features
        :param X_test: Testing features
        :param y_train: Training labels
        :return: Predictions for the test set
        """
        # Preprocess the training and testing data using the injected scaler
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        # Train the model using the training data
        self.model.train(X_train, y_train)

        # Make predictions on the test data
        return self.model.predict(X_test)
