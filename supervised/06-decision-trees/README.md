# Decision Trees

## Introduction

Decision Tree is a supervised machine learning algorithm used for both classification and regression tasks. It predicts outcomes by learning simple decision rules inferred from data features.

Decision Trees are highly interpretable because their structure resembles a flowchart where each internal node represents a feature, each branch represents a decision, and each leaf node represents an outcome.

---

## Learning Objectives

After completing this module, you will be able to:

- Understand Decision Tree fundamentals.
- Explain tree structure and terminology.
- Learn Entropy, Information Gain, and Gini Index.
- Understand tree construction.
- Learn pruning techniques.
- Implement Decision Trees using Scikit-learn.
- Evaluate classification performance.
- Answer common interview questions.

---

## What is a Decision Tree?

A Decision Tree is a tree-like model that recursively splits data into subsets based on feature values.

The goal is to create pure subsets where data points belong to the same class.

---

## Tree Terminology

### Root Node

The top-most node of the tree.

### Parent Node

A node that splits into one or more child nodes.

### Child Node

A node generated after splitting a parent node.

### Leaf Node

The final node that produces predictions.

### Branch

Represents the outcome of a decision rule.

---

## Entropy

Entropy measures impurity or randomness in a dataset.

Formula:

Entropy(S) = -Σ p(x) log₂ p(x)

Interpretation:

- Entropy = 0 → Pure node
- Higher entropy → More disorder

---

## Information Gain

Information Gain measures how much uncertainty is reduced after a split.

Formula:

Information Gain = Entropy(Parent) - Weighted Entropy(Children)

The feature with the highest Information Gain is selected for splitting.

---

## Gini Index

Gini Index measures node impurity.

Formula:

Gini = 1 - Σ p(x)²

Interpretation:

- Lower Gini → Better split
- Gini = 0 → Pure node

---

## Tree Construction Process

1. Select best feature.
2. Split dataset.
3. Calculate Entropy or Gini.
4. Repeat recursively.
5. Stop when stopping criteria are met.

---

## Overfitting in Decision Trees

Decision Trees can memorize training data and create overly complex structures.

Symptoms:

- High training accuracy
- Low testing accuracy

---

## Pruning Techniques

### Pre-Pruning

Stops tree growth early.

Examples:

- Maximum depth
- Minimum samples split

### Post-Pruning

Builds full tree first and removes unnecessary branches later.

---

## Advantages

- Easy to understand.
- Requires minimal preprocessing.
- Handles numerical and categorical data.
- Provides feature importance.

---

## Limitations

- Prone to overfitting.
- Sensitive to small dataset changes.
- Can create biased trees.

---

## Applications

- Medical Diagnosis
- Credit Risk Analysis
- Customer Segmentation
- Fraud Detection
- Loan Approval Systems

---

## Dataset Information

Dataset: Titanic Dataset

Problem Type: Binary Classification

Objective:

Predict passenger survival based on demographic and travel-related features.

---

## Practical Implementation

The implementation demonstrates:

- Data Loading
- Data Exploration
- Data Preprocessing
- Train-Test Split
- Decision Tree Training
- Prediction
- Evaluation

---

## Workflow Diagram

![Workflow Diagram](images/workflow.png)

---

## Interview Questions

1. What is a Decision Tree?

2. What is Entropy?

3. What is Information Gain?

4. What is the Gini Index?

5. Difference between Gini and Entropy?

6. Why do Decision Trees overfit?

7. What is pruning?

8. What is a leaf node?

9. What is a root node?

10. Advantages and limitations of Decision Trees?

---

## Key Takeaways

- Decision Trees are interpretable models.
- Entropy and Gini measure impurity.
- Information Gain selects the best feature.
- Pruning helps reduce overfitting.
- Decision Trees support classification and regression.

---

## References

- Scikit-learn Documentation
- GeeksforGeeks
- Towards Data Science
- Medium Machine Learning Blogs