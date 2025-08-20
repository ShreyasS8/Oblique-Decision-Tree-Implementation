import numpy as np
import pandas as pd
import sys
import csv
import logistic_regression as lgr  # Assuming logistic_regression has a method `logistic_regression`

class ObliqueDecisionTree_class:
    def __init__(self, max_depth, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    class Node:
        def __init__(self, node_id, weights=None, threshold=None, left=None, right=None, prediction=None):
            self.node_id = node_id
            self.weights = weights
            self.threshold = threshold
            self.left = left
            self.right = right
            self.prediction = prediction

    def fit(self, X, y):
        y = y.astype(int)
        self.tree = self._build_tree_func(X, y, depth=1, node_id=1)

    def _build_tree_func(self, X, y, depth, node_id):
        if depth > self.max_depth or len(y) < self.min_samples_split or np.all(y == y[0]):
            prediction = np.bincount(y).argmax()
            return self.Node(node_id=node_id, prediction=prediction)

        weights = lgr.logistic_regression(X, y)
        threshold = self._find_best_threshold_func(X, y, weights)

        left_indices = X.dot(weights) <= threshold
        right_indices = ~left_indices

        if sum(left_indices) == 0 or sum(right_indices) == 0:
            prediction = np.bincount(y).argmax()
            return self.Node(node_id=node_id, prediction=prediction)

        left_child = self._build_tree_func(X[left_indices], y[left_indices], depth + 1, node_id * 2)
        right_child = self._build_tree_func(X[right_indices], y[right_indices], depth + 1, node_id * 2 + 1)
        return self.Node(node_id=node_id, weights=weights, threshold=threshold, left=left_child, right=right_child)

    def _find_best_threshold_func(self, X, y, weights):
        projections = X.dot(weights)
        unique_projections = np.unique(projections)
        midpoints = (unique_projections[:-1] + unique_projections[1:]) / 2

        best_threshold, min_gini = unique_projections[0], float("inf")
        for split in midpoints:
            left, right = y[projections < split], y[projections >= split]
            gini = self._gini_impurity_func(left, right)
            if gini < min_gini:
                min_gini, best_threshold = gini, split
        return best_threshold

    def _gini_impurity_func(self, left_y, right_y):
        def gini(labels):
            if len(labels) == 0:
                return 0
            # Ensure the labels are integers
            labels = labels.astype(int)
            p = np.bincount(labels) / len(labels)
            return 1 - np.sum(p ** 2)

        total = len(left_y) + len(right_y)
        return (len(left_y) * gini(left_y.astype(int)) + len(right_y) * gini(right_y.astype(int))) / total

    def predict(self, X):
        return np.array([self._predict_sample_func(sample, self.tree) for sample in X])

    def _predict_sample_func(self, sample, node):
        if node.prediction is not None:
            return node.prediction
        if sample.dot(node.weights) < node.threshold:
            return self._predict_sample_func(sample, node.left)
        else:
            return self._predict_sample_func(sample, node.right)

    def prune(self, X_val, y_val):
        self._prune_node_func(self.tree, X_val, y_val)

    def _prune_node_func(self, node, X_val, y_val):
        if node is None or node.prediction is not None:
            return node.prediction if node else None

        # Split the validation set using the current node's weights and threshold
        left_indices = X_val.dot(node.weights) < node.threshold
        right_indices = ~left_indices

        # Prune left child first
        left_prediction = self._prune_node_func(node.left, X_val[left_indices], y_val[left_indices])

        # Prune right child
        right_prediction = self._prune_node_func(node.right, X_val[right_indices], y_val[right_indices])

        # If both children are leaves, evaluate if pruning the subtree improves validation accuracy
        if node.left and node.left.prediction is not None and node.right and node.right.prediction is not None:
            # Calculate original accuracy before pruning
            original_accuracy = np.mean(y_val == self.predict(X_val)) if len(y_val) > 0 else 0

            # Assign the majority class in the current validation set to this node
            if len(y_val) > 0:
                majority_class = np.bincount(y_val.astype(int)).argmax()
                node.prediction = majority_class
            else:
                node.prediction = 0  # Default to class 0 if validation set is empty

            # Calculate pruned accuracy
            pruned_accuracy = np.mean(y_val == self.predict(X_val)) if len(y_val) > 0 else 0

            # If pruning doesn't improve accuracy, revert to split node
            if pruned_accuracy < original_accuracy:
                node.prediction = None
            else:
                # If pruning is beneficial, remove the children nodes
                node.left = node.right = node.weights = node.threshold = None

        return node.prediction if node.prediction is not None else None

    def _split_data(self, X, y, weights, threshold):
        left_indices = X.dot(weights) <= threshold
        right_indices = ~left_indices
        return X[left_indices], X[right_indices], y[left_indices], y[right_indices]

    def save_weights_func(self, filename):
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            self._write_node_weights_func(writer, self.tree)

    def _write_node_weights_func(self, writer, node):
        if node is None or node.prediction is not None:
            return
        writer.writerow([node.node_id] + node.weights.tolist() + [node.threshold])
        self._write_node_weights_func(writer, node.left)
        self._write_node_weights_func(writer, node.right)

    def save_predictions_func(self, X, filename):
        predictions = self.predict(X)
        pd.DataFrame({'Prediction': predictions}).to_csv(filename, index=False, header=False)

    def accuracy(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)

    def count_non_leaf_nodes_func(self, node):
        if node is None or node.prediction is not None:
            return 0
        return 1 + self.count_non_leaf_nodes_func(node.left) + self.count_non_leaf_nodes_func(node.right)

def _accuracies(tree, X_train, y_train, X_val, y_val, X_test, y_test, pruned=False):
    tree_type = "Pruned" if pruned else "Unpruned"
    #print(f"\n{tree_type} Tree Results:")
    train_accuracy = tree.accuracy(X_train, y_train) * 100
    val_accuracy = tree.accuracy(X_val, y_val) * 100
    test_accuracy = tree.accuracy(X_test, y_test) * 100
    non_leaf_nodes = tree.count_non_leaf_nodes_func(tree.tree)

    #print(f"Train Accuracy: {train_accuracy:.2f}%")
    #print(f"Validation Accuracy: {val_accuracy:.2f}%")
    #print(f"Test Accuracy: {test_accuracy:.2f}%")
    #print(f"Number of Non-Leaf Nodes: {non_leaf_nodes}")

def main():
    args = sys.argv[1:]
    mode = args[0]

    if mode == "train" and args[1] == "unpruned":
        # Unpruned training
        train_data = pd.read_csv(args[2])
        max_depth = int(args[3])
        weight_file = args[4]

        X_train, y_train = train_data.iloc[:, :-1].values, train_data.iloc[:, -1].values
        oblique_tree = ObliqueDecisionTree_class(max_depth=max_depth)
        oblique_tree.fit(X_train, y_train)
        oblique_tree.save_weights_func(weight_file)
        #print("Unpruned tree training completed and weights saved.")
        #_accuracies(oblique_tree, X_train, y_train, X_train, y_train, pruned=False)


    elif mode == "train" and args[1] == "pruned":
        # Pruned training
        train_data = pd.read_csv(args[2])
        val_data = pd.read_csv(args[3])
        max_depth = int(args[4])
        weight_file = args[5]

        X_train, y_train = train_data.iloc[:, :-1].values, train_data.iloc[:, -1].values
        X_val, y_val = val_data.iloc[:, :-1].values, val_data.iloc[:, -1].values
        oblique_tree = ObliqueDecisionTree_class(max_depth=max_depth)
        oblique_tree.fit(X_train, y_train)
        oblique_tree.prune(X_val, y_val)
        oblique_tree.save_weights_func(weight_file)
        #print("Pruned tree training and pruning completed and weights saved.")
        # _accuracies(oblique_tree, X_train, y_train, X_val, y_val, X_train, y_train, pruned=False)
        train_accuracy = oblique_tree.accuracy(X_train, y_train) * 100
        #print(f"Train Accuracy: {train_accuracy:.2f}%")

    elif mode == "test":
        # Test and predictions
        train_data = pd.read_csv(args[1])
        val_data = pd.read_csv(args[2])
        test_data = pd.read_csv(args[3])
        max_depth = int(args[4])
        prediction_file = args[5]

        X_train, y_train = train_data.iloc[:, :-1].values, train_data.iloc[:, -1].values
        X_val, y_val = val_data.iloc[:, :-1].values, val_data.iloc[:, -1].values
        X_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values  # Assuming test labels are available here

        # Train unpruned tree and #print results
        oblique_tree = ObliqueDecisionTree_class(max_depth=max_depth)
        oblique_tree.fit(X_train, y_train)
        _accuracies(oblique_tree, X_train, y_train, X_val, y_val, X_test, y_test, pruned=False)

        # Prune and evaluate pruned tree
        oblique_tree.prune(X_val, y_val)
        _accuracies(oblique_tree, X_train, y_train, X_val, y_val, X_test, y_test, pruned=True)

        # Save predictions
        oblique_tree.save_predictions_func(X_test, prediction_file)
        #print("Predictions saved.")

if __name__ == "__main__":
    main()
