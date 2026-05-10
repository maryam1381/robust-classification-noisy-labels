Comparative Study
This project investigates the impact of label noise on various machine learning models. In real-world datasets, labels are often imperfect due to human error or automated labeling glitches. This study evaluates how different architectures—from linear models to ensemble methods and neural networks—behave when trained on noisy data.

## Project Overview
The goal is to analyze the robustness of four popular classifiers when the training labels are randomly corrupted at different rates (from 0% to 40%).

## Dataset

The MNIST dataset used in this project can be downloaded from OpenML:

https://www.openml.org/d/554

After downloading, place the file in the root directory:


## Key Features:
Dataset: MNIST (Handwritten digits).
Noise Injection: Synthetic label noise added at controlled levels (0.0, 0.1, 0.2, 0.3, 0.4).
Models Evaluated:
1) Logistic Regression (Baseline)
2) Random Forest (Ensemble Learning)
3) Multi-Layer Perceptron (Neural Network)
Evaluation Metric: Classification Accuracy on a clean test set.


## Key Findings
Random Forest demonstrated the highest robustness. Its ensemble nature allows it to “average out” the effect of individual noisy labels better than other models.
LR showed a steady decline in performance as noise increased, indicating sensitivity to outliers in the decision boundary.
MLP performs exceptionally well on clean data but its accuracy drops faster under high noise, likely due to overfitting on incorrect labels (unless regularization is heavily applied).


![Results Plot](results\model_noise_comparison.png)