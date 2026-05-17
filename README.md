# Sales Prediction from Advertising

A beginner machine learning project built using PyTorch to predict product sales from advertising data using linear regression.

---

## Project Overview

This project demonstrates an end-to-end machine learning workflow using PyTorch:

- Loading data using Pandas
- Data preprocessing
- Converting data into tensors
- Building a linear regression model
- Training using gradient descent and Adam optimizer
- Testing predictions on unseen data

---

## Technologies Used

- Python
- PyTorch
- Pandas
- NumPy

---

## Model

The model uses Linear Regression:

\[
\hat{y} = XW + b
\]

where:
- \(X\) = advertising features
- \(W\) = learned weights
- \(b\) = bias
- \(\hat{y}\) = predicted sales

---

## Dataset

The dataset contains advertising spending data and corresponding sales values.

Features include:
- TV advertising budget
- Radio advertising budget
- Newspaper advertising budget

Target:
- Sales

---

## Training

Loss Function:
- Mean Squared Error (MSE)

Optimizer:
- Adam Optimizer

---

## Example Workflow

1. Load dataset
2. Preprocess data
3. Convert to tensors
4. Train model
5. Predict sales on test data

---

## Run the Project

```bash
python3 advertising.py
```

---

## Future Improvements

- Add train/test split evaluation metrics
- Add data normalization
- Build deeper neural networks
- Compare with other ML models

---

## Author

Aditya Robin
