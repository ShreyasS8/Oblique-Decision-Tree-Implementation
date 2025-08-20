# Oblique Decision Tree Implementation

This repository contains a Python implementation of an Oblique Decision Tree from scratch. Unlike traditional axis-aligned decision trees that split nodes based on a single feature, an oblique decision tree creates splits based on a linear combination of multiple features. This allows for more complex decision boundaries and can lead to smaller, more accurate trees.

The core splitting mechanism at each node is a two-step process:

1. A Logistic Regression model is trained on the node's data to find an optimal separating hyperplane.

2. The Gini Impurity or Information Gain is then used to find the best threshold along the hyperplane's normal to partition the data.

The project also includes a bottom-up, post-pruning algorithm (Reduced Error Pruning) to prevent overfitting by simplifying the tree based on its performance on a validation set.

## 📋 Project Structure
This project is structured to fulfill the requirements of the original assignment, which is divided into four main parts:

### Part A: Unpruned Tree
Implementation of the ObliqueDecisionTree class.

- The tree is grown to its maximum depth to achieve 100% accuracy on the training data, demonstrating the model's ability to perfectly fit the training set.

### Part B: Reduced Error Pruning
Implementation of a post-pruning algorithm.

- After a full tree is grown, it is pruned from the bottom up. At each node, the algorithm decides whether to keep the subtree or replace it with a leaf node by comparing their respective accuracies on a separate validation dataset.

### Part C: Real-World Application
Application of the complete pipeline (training and pruning) to a real-world dataset.

This part involves:

- Training a full, unpruned tree.
- Pruning the tree using validation data.
- Generating predictions on a test set using the final pruned tree.

### Part D: Competitive Challenge
An advanced and optimized version of the tree designed to achieve the best possible generalization on the test set.

This implementation introduces enhancements such as:

- **Regularization**: Using L2 and Elastic Net regularization in the logistic regression splits to create more robust and simpler hyperplanes.
- **Impurity Metric**: Using Information Gain (Entropy) as an alternative to Gini Impurity for selecting the optimal split threshold.
- **Hyperparameter Tuning**: Fine-tuning parameters like max_depth, min_samples_split, and regularization strength to balance the bias-variance tradeoff.

## ⚙️ Implementation Details

### Node Splitting Mechanism
Each non-leaf node in the tree represents a decision rule of the form:

$$ \mathbf{w}^T \mathbf{x} \le \theta $$

where  
**w** is a vector of weights,  
**x** is a feature vector, and  
**θ** is a threshold.

The process to find **w** and **θ** at a given node is as follows:

1. **Find the Hyperplane (w)**:  
A `LogisticRegression` model from Scikit-learn is trained on all data points (X,y) present at the node. The coefficients of the fitted model serve as the weight vector **w**.

```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver='lbfgs')
model.fit(X, y)
weights = model.coef_[0]
```

2. **Find the Threshold (θ)**:  
The data points are projected onto the 1-dimensional line defined by the weight vector **w**. The algorithm then searches for the best threshold **θ** on these projected values (`projections = X.dot(weights)`) that minimizes the weighted Gini Impurity or maximizes the Information Gain of the resulting child nodes.

### Tree Pruning
The pruning is performed using Reduced Error Pruning, a post-pruning technique that works as follows:

1. The tree is built completely on the training data.  
2. It is then traversed in a bottom-up manner.  
3. For each internal node, we evaluate two scenarios using a validation set:  
   a. The accuracy of the subtree rooted at this node.  
   b. The accuracy of converting this node into a leaf, predicting the majority class of the training data that reached it.  

If the accuracy of converting the node to a leaf is greater than or equal to the accuracy of its subtree, the subtree is pruned, and the node becomes a leaf.

## 🚀 How to Run

### Dependencies
The project requires the following Python libraries. You can install them using pip:

```bash
pip install numpy pandas scikit-learn
```

### Execution
The scripts are designed to be run from the command line. The primary script `main.py` handles training, pruning, and prediction.

Example command for the competitive solution (Part D):  
This command trains a tree on `train_real.csv`, prunes it using `val_real.csv`, and then generates predictions for `test_real.csv`.

```bash
python main.py train_real.csv val_real.csv test_real.csv prediction_real_competitive.csv weights_real_competitive.csv
```
(Note: This assumes a `main.py` adapted from the competitive code provided.)

## 📁 Output File Formats

The program generates two types of output files: weights and predictions.

### 1. Weights File (`weights_*.csv`)
This file stores the structure of the trained decision tree. Each row represents a non-leaf (internal) node and follows this format:

```
node_id,weight_1,weight_2,...,weight_n,threshold
```

- **node_id**: An integer identifying the node. The root is 1, and the children of node x are 2x (left) and 2x+1 (right).  
- **weight_i**: The coefficient for the i-th feature in the linear combination.  
- **threshold**: The threshold value for the split.  

For the sample tree shown above, the first line in the weights file would be:

```
1,1,-0.03,-0.02,0.01
```

### 2. Prediction File (`prediction_*.csv`)
This file contains the final class label predictions for the test dataset. It is a single-column CSV with no header, where each row corresponds to a test instance in the original order.

Example:

```
1
0
1
1
0
...
```
