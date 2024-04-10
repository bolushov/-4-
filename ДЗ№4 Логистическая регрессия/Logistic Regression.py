
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0

        for i in range(self.num_iterations):
            z = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(z)

            dw = (1 / m) * np.dot(X.T, (predictions - y))
            db = (1 / m) * np.sum(predictions - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        predictions = self.sigmoid(z)
        return np.round(predictions)

# Загрузка данных Iris
iris = load_iris()
X = iris.data
y = (iris.target == 0).astype(int)  # 1 if Setosa, else 0

# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Эксперимент с логистической регрессией
learning_rates = [0.1, 0.01, 0.001]
for learning_rate in learning_rates:
    print(f"Logistic Regression (learning_rate={learning_rate}):")
    logistic_regression = LogisticRegression(learning_rate=learning_rate)
    logistic_regression.fit(X_train, y_train)
    lr_predictions = logistic_regression.predict(X_test)
    lr_accuracy = accuracy_score(y_test, lr_predictions)
    print(f"Accuracy: {lr_accuracy}")

# Эксперимент с kNN
k_values = [1, 3, 5, 7, 9]
for k in k_values:
    print(f"kNN (k={k}):")
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    knn_predictions = knn.predict(X_test)
    knn_accuracy = accuracy_score(y_test, knn_predictions)
    print(f"Accuracy: {knn_accuracy}")
