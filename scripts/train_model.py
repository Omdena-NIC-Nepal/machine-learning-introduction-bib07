# scripts/train_model.py

from sklearn.linear_model import LinearRegression

def train_model(X_train, y_train):
    """
    Trains a linear regression model using the training data.
    
    Parameters:
    - X_train: Training features
    - y_train: Training target

    Returns:
    - model: Trained linear regression model
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model
