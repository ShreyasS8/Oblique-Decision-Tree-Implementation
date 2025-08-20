import sys
import numpy as np
import pandas as pd
import csv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# CLASS OBLIQUE DECISION TREE
class Oblique_Decision_Tree:
    def __init__(self, max_depth=20, min_samples_split=5, regularization=0.01, solver='lbfgs', l1_ratio=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.regularization = regularization
        self.solver = solver
        self.l1_ratio = l1_ratio
        self.tree = None
    # NODE STRUCTURE
    class Node:
        def __init__(self, node_id, weights=None, threshold=None, left=None, right=None, prediction=None):
            self.node_id = node_id
            self.weights = weights
            self.threshold = threshold
            self.left = left
            self.right = right
            self.prediction = prediction

    def fit(self, X, y):
        self.tree = self.build_tree_func(X, y, depth=1, node_id=1)

    def build_tree_func(self, X, y, depth, node_id):
        if depth > self.max_depth or len(y) < self.min_samples_split or np.all(y == y[0]):
            prediction = np.bincount(y).argmax()
            return self.Node(node_id=node_id, prediction=prediction)

        model_solver = 'saga' if self.l1_ratio is not None else 'lbfgs'
        model = LogisticRegression(
            max_iter=1000,
            C=self.regularization,
            solver=model_solver,
            l1_ratio=self.l1_ratio,
            penalty='elasticnet' if self.l1_ratio is not None else 'l2'
        )
        
        model.fit(X, y)
        weights = model.coef_[0]
        threshold = self.best_threshold(X, y, weights)

        left_indices = X.dot(weights) <= threshold
        right_indices = ~left_indices

        if sum(left_indices) == 0 or sum(right_indices) == 0:
            prediction = np.bincount(y).argmax()
            return self.Node(node_id=node_id, prediction=prediction)

        left_child = self.build_tree_func(X[left_indices], y[left_indices], depth + 1, node_id * 2)
        right_child = self.build_tree_func(X[right_indices], y[right_indices], depth + 1, node_id * 2 + 1)
        return self.Node(node_id=node_id, weights=weights, threshold=threshold, left=left_child, right=right_child)

    def best_threshold(self, X, y, weights):
        projections = X.dot(weights)
        unique_projections = np.unique(projections)
        midpoints = (unique_projections[:-1] + unique_projections[1:]) / 2

        best_threshold, min_weighted_entropy = unique_projections[0], float("inf")
        for split in midpoints:
            left, right = y[projections < split], y[projections >= split]
            weighted_entropy = self.weighted_entropy_impurity_func(left, right)
            if weighted_entropy < min_weighted_entropy:
                min_weighted_entropy, best_threshold = weighted_entropy, split
        return best_threshold

    def weighted_entropy_impurity_func(self, left_y, right_y):
        def entropy(labels):
            if len(labels) == 0:
                return 0
            p = np.bincount(labels) / len(labels)
            return -np.sum(p * np.log2(p + 1e-9))

        total = len(left_y) + len(right_y)
        left_weight = len(left_y) / total
        right_weight = len(right_y) / total
        return left_weight * entropy(left_y) + right_weight * entropy(right_y)

    def predict(self, X):
        return np.array([self.predict_sample_func(sample, self.tree) for sample in X])

    def predict_sample_func(self, sample, node):
        if node.prediction is not None:
            return node.prediction
        if sample.dot(node.weights) < node.threshold:
            return self.predict_sample_func(sample, node.left)
        else:
            return self.predict_sample_func(sample, node.right)

    def prune(self, X_val, y_val):
        self.prune_node_func(self.tree, X_val, y_val)

    def prune_node_func(self, node, X_val, y_val):
        if node is None or node.prediction is not None:
            return np.mean(y_val == node.prediction) if len(y_val) > 0 else 0

        left_X, right_X, left_y, right_y = self.split_data_func(X_val, y_val, node.weights, node.threshold)
        left_accuracy = self.prune_node_func(node.left, left_X, left_y)
        right_accuracy = self.prune_node_func(node.right, right_X, right_y)

        subtree_accuracy = (len(left_y) * left_accuracy + len(right_y) * right_accuracy) / len(y_val) if len(y_val) > 0 else 0
        majority_class = np.bincount(y_val).argmax() if len(y_val) > 0 else 0
        leaf_accuracy = np.mean(y_val == majority_class) if len(y_val) > 0 else 0

        if leaf_accuracy >= subtree_accuracy:
            node.prediction = majority_class
            node.left = node.right = node.weights = node.threshold = None
            return leaf_accuracy
        return subtree_accuracy

    def split_data_func(self, X, y, weights, threshold):
        left_indices = X.dot(weights) <= threshold
        right_indices = ~left_indices
        return X[left_indices], X[right_indices], y[left_indices], y[right_indices]

    def save_weights_func(self, filename):
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            self.write_node_weights(writer, self.tree)

    def write_node_weights(self, writer, node):
        if node is None or node.prediction is not None:
            return
        writer.writerow([node.node_id] + node.weights.tolist() + [node.threshold])
        self.write_node_weights(writer, node.left)
        self.write_node_weights(writer, node.right)

    def save_predictions(self, X, filename):
        predictions = self.predict(X)
        pd.DataFrame({'Prediction': predictions}).to_csv(filename, index=False, header=False)

    def accuracy(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)

def main():

    _,_,train_file, val_file, test_file, prediction_file, weight_file = sys.argv

    train_data = pd.read_csv(train_file)
    val_data = pd.read_csv(val_file)
    test_data = pd.read_csv(test_file)

    X_train, y_train = train_data.iloc[:, :-1].values, train_data.iloc[:, -1].values
    X_val, y_val = val_data.iloc[:, :-1].values, val_data.iloc[:, -1].values
    X_test = test_data.iloc[:, :-1].values  # Exclude the last column as it's the target

    # Parameters for the decision tree
    max_depth = 20
    min_samples_split = 10
    regularization = 0.01
    solver = 'saga'  # Use 'saga' to support elasticnet if needed
    l1_ratio = 0.5  # Only relevant for 'elasticnet'

    # Train the tree
    tree = Oblique_Decision_Tree(max_depth=max_depth, min_samples_split=min_samples_split, regularization=regularization, solver=solver, l1_ratio=l1_ratio)
    tree.fit(X_train, y_train)
    tree.prune(X_val, y_val)

    # #print training, validation, and test accuracy
    train_accuracy = tree.accuracy(X_train, y_train)
    val_accuracy = tree.accuracy(X_val, y_val)
    #print(f"Training Accuracy: {train_accuracy:.4f}")
    #print(f"Validation Accuracy: {val_accuracy:.4f}")

    # Test predictions and accuracy (if test labels are provided)
    test_predictions = tree.predict(X_test)
    if 'target' in test_data.columns:
        test_accuracy = np.mean(test_predictions == test_data.iloc[:, -1].values)
        #print(f"Test Accuracy: {test_accuracy:.4f}")

    # Save weights and predictions
    tree.save_weights_func(weight_file)
    tree.save_predictions(X_test, prediction_file)

if __name__ == "__main__":
    main()
