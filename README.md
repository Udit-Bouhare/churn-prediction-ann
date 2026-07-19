# 📊 Customer Churn Prediction using Artificial Neural Network (ANN)

Predict whether a bank customer is likely to leave the bank using a Deep Learning model built with PyTorch and deployed with Streamlit.

## 🚀 Live Demo

Add your Streamlit URL here:

[Live App](https://churn-prediction-ann-jbnbocbxdphxo3ymeqtlsr.streamlit.app/)

---

## 📌 Project Overview

Customer churn prediction is a classification problem where the goal is to identify customers who are likely to stop using a company's services.

In this project, an Artificial Neural Network (ANN) is trained on customer banking data to predict whether a customer will churn or stay.

---

## 🎯 Problem Statement

Given customer information such as:

- Credit Score
- Geography
- Gender
- Age
- Tenure
- Balance
- Number of Products
- Credit Card Status
- Active Membership
- Estimated Salary

Predict:

- **0 → Customer Stays**
- **1 → Customer Churns**

---

## 🛠️ Tech Stack

### Deep Learning

- PyTorch
- Torchvision

### Data Processing

- Pandas
- NumPy
- Scikit-Learn

### Deployment

- Streamlit

### Model Persistence

- Pickle
- PyTorch State Dictionary

---

## 📂 Dataset

Dataset contains banking customer information and churn labels.

Target Variable:

```text
Exited
```

- 1 = Customer Churned
- 0 = Customer Retained

---

## ⚙️ Data Preprocessing

### 1. Label Encoding

Gender column:

```text
Female → 0
Male → 1
```

### 2. One-Hot Encoding

Geography column:

```text
France
Germany
Spain
```

Converted into:

```text
Geography_France
Geography_Germany
Geography_Spain
```

### 3. Feature Scaling

StandardScaler was used to normalize numerical features before training.

---

## 🧠 ANN Architecture

```text
Input Layer (12 Features)
        ↓
Linear (12 → 64)
        ↓
BatchNorm1D
        ↓
ReLU
        ↓
Dropout (0.3)

        ↓

Linear (64 → 32)
        ↓
BatchNorm1D
        ↓
ReLU
        ↓
Dropout (0.3)

        ↓

Linear (32 → 1)
        ↓
Sigmoid
```

---

## 📈 Training Configuration

| Parameter | Value |
|------------|---------|
| Loss Function | BCELoss |
| Optimizer | Adam |
| Batch Size | 32 |
| Epochs | 100 |
| Activation | ReLU |
| Output Activation | Sigmoid |

---

## 📁 Project Structure

```text
ANN_Classification/
│
├── app.py
├── model.py
├── model.pth
├── scaler.pkl
├── label_encoder_gender.pkl
├── onehot_encoder_geo.pkl
│
├── experiments.ipynb
├── Churn_Modelling.csv
│
├── README.md
├── pyproject.toml
└── .gitignore
```

---

## 💾 Model Saving

Model weights were saved using PyTorch:

```python
torch.save({
    "model_state": model.state_dict(),
    "num_features": 12
}, "model.pth")
```

Preprocessing objects were saved using Pickle:

```python
pickle.dump(scaler, file)
pickle.dump(label_encoder_gender, file)
pickle.dump(onehot_encoder_geo, file)
```

---

## 🌐 Streamlit Application

The deployed application allows users to:

- Enter customer information
- Process inputs automatically
- Generate churn probability
- Display prediction results

Example Output:

```text
Churn Probability: 0.78

⚠️ Customer is likely to churn
```

---

## 📊 Skills Demonstrated

- Data Preprocessing
- Feature Engineering
- Label Encoding
- One-Hot Encoding
- Feature Scaling
- PyTorch Dataset & DataLoader
- ANN Development
- Batch Normalization
- Dropout Regularization
- Binary Classification
- Model Serialization
- Streamlit Deployment

---

## 📸 Application Screenshot

Add screenshot here:

```text
screenshots/app.png
```

---

## 🔮 Future Improvements

- Hyperparameter Tuning
- Early Stopping
- Model Monitoring
- MLflow Integration
- Docker Deployment
- CI/CD Pipeline
- Explainable AI (SHAP)

---

## 👨‍💻 Author

**Udit Bauhare**

- GitHub: https://github.com/YOUR_USERNAME
- LinkedIn: https://linkedin.com/in/YOUR_PROFILE

---

⭐ If you found this project useful, consider giving it a star.
