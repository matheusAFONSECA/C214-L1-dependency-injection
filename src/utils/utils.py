from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from icecream import ic
from sklearn.datasets import make_classification


def load_data():
    """
    Load synthetic classification data for demonstration purposes.

    :return: Features (X) and labels (y)
    """

    return make_classification(n_samples=1000, n_features=20, random_state=42)


def split_data(X, y, test_size=0.3, random_state=42):
    """
    Split data into training and testing sets.

    :param X: Features
    :param y: Labels
    :param test_size: Proportion of the data to include in the test split
    :param random_state: Random state for reproducibility
    :return: X_train, X_test, y_train, y_test
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def evaluate_model(y_test, predictions):
    """
    Evaluate the model's accuracy based on predictions.

    :param y_test: True labels for the test set
    :param predictions: Predicted labels from the model
    :return: Prints the accuracy of the model
    """
    accuracy = accuracy_score(y_test, predictions)
    ic(accuracy)


def get_scaler():
    """
    Get a StandardScaler instance for data normalization.

    :return: A StandardScaler object
    """
    return StandardScaler()
