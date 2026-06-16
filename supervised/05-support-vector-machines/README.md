# Support Vector Machines (SVM)

## Introduction

Support Vector Machine (SVM) is a powerful supervised machine learning algorithm used for classification and regression tasks. It is particularly effective in high-dimensional datasets and works by finding the optimal decision boundary that separates different classes.

SVM aims to maximize the margin between classes, making it one of the most robust classification algorithms. Due to its strong mathematical foundation and excellent performance on structured datasets, SVM is widely used in machine learning applications.

---

## Learning Objectives

After completing this module, you will be able to:

* Understand the concept of Support Vector Machines.
* Explain hyperplanes, margins, and support vectors.
* Differentiate between linear and non-linear SVMs.
* Understand the kernel trick and various kernel functions.
* Implement SVM using Scikit-learn.
* Evaluate classification performance.
* Answer common interview questions related to SVM.

---

## What is Support Vector Machine?

A Support Vector Machine is a supervised learning algorithm that separates data points into different classes by identifying the optimal hyperplane.

The objective is not only to separate classes but also to maximize the distance between the nearest data points and the decision boundary.

This maximum distance is known as the margin.

---

## Why SVM?

Traditional classifiers may create decision boundaries that are sensitive to noise.

SVM addresses this issue by:

* Maximizing margin.
* Improving generalization.
* Handling high-dimensional data effectively.
* Supporting non-linear classification through kernels.

---

## Hyperplane

A hyperplane is the decision boundary that separates classes.

For:

* 2D data → Hyperplane is a line.
* 3D data → Hyperplane is a plane.
* Higher dimensions → Hyperplane is a higher-dimensional boundary.

The optimal hyperplane is the one that maximizes the margin between classes.

---

## Margin

Margin is the distance between the hyperplane and the nearest data points from each class.

Two types:

### Hard Margin

Assumes perfectly separable data.

### Soft Margin

Allows some misclassification to improve generalization.

Larger margins generally lead to better model performance.

---

## Support Vectors

Support vectors are the data points closest to the hyperplane.

These points directly influence the position of the decision boundary.

If support vectors change, the hyperplane changes.

This is why they are called "support vectors."

---

## Linear SVM

Used when data is linearly separable.

Characteristics:

* Simple model.
* Faster training.
* Easy interpretation.

Applications:

* Text classification.
* Binary classification problems.

---

## Non-Linear SVM

Used when classes cannot be separated using a straight line.

To solve this problem, SVM uses the Kernel Trick.

---

## Kernel Trick

The kernel trick transforms data into a higher-dimensional space where it becomes linearly separable.

Advantages:

* Avoids explicit feature transformation.
* Handles complex patterns.
* Improves classification performance.

---

## Types of Kernels

### Linear Kernel

Best for linearly separable data.

### Polynomial Kernel

Useful when relationships between classes are polynomial.

### RBF (Radial Basis Function) Kernel

Most commonly used kernel.

Handles complex non-linear decision boundaries.

### Sigmoid Kernel

Behaves similarly to neural networks.

Used less frequently in practical applications.

---

## Advantages

* Effective in high-dimensional spaces.
* Works well with small datasets.
* Robust against overfitting.
* Supports linear and non-linear classification.
* Strong theoretical foundation.

---

## Limitations

* Computationally expensive for large datasets.
* Sensitive to parameter selection.
* Difficult to interpret compared to Decision Trees.
* Training time increases with dataset size.

---

## Applications

* Face Recognition
* Spam Detection
* Text Classification
* Bioinformatics
* Medical Diagnosis
* Image Classification

---

## Dataset Information

Dataset: Breast Cancer Wisconsin Dataset

Problem Type: Binary Classification

Objective:
Predict whether a tumor is benign or malignant using medical features.

---

## Practical Implementation

The implementation demonstrates:

* Data Loading
* Data Exploration
* Data Preprocessing
* Feature Scaling
* Train-Test Split
* SVM Training
* Prediction
* Evaluation

---

## Workflow Diagram

![Workflow Diagram](images/workflow.png)

---

## Interview Questions

1. What is a Support Vector Machine?

2. What is a hyperplane?

3. What are support vectors?

4. What is margin in SVM?

5. Difference between hard margin and soft margin?

6. What is the kernel trick?

7. Difference between linear and non-linear SVM?

8. Why is feature scaling important in SVM?

9. Which kernel is most commonly used?

10. What are the advantages and limitations of SVM?

---

## Key Takeaways

* SVM is a supervised learning algorithm.
* It finds an optimal hyperplane for classification.
* Support vectors determine the position of the decision boundary.
* Kernel functions enable non-linear classification.
* Feature scaling is important for SVM performance.
* SVM is widely used in classification problems.

---

## References

* Scikit-learn Documentation
* GeeksforGeeks
* Towards Data Science
* Medium Machine Learning Blogs
