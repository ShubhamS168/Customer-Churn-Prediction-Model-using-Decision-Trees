# Customer Churn Prediction Model - Streaming Entertainment Industry

<!-- **🔗 Deployed Prediction Model:** [Customer Churn Prediction Model](https://customer-churn-prediction-model-using-decision-trees-hfkmnsgxm.streamlit.app/) -->

**🔗 Deployed Prediction Model:** <a href="https://customer-churn-prediction-model-using-decision-trees-hfkmnsgxm.streamlit.app/" target="_blank">
 Customer Churn Prediction Model
</a>

## 📋 Project Overview

This is a **complete, production-ready Machine Learning project** that predicts customer churn for a streaming entertainment platform using **Decision Tree Classification**. The project includes data generation, preprocessing, exploratory analysis, model training with hyperparameter tuning, and an interactive web application.

### 🎯 Business Context

Customer churn (customers leaving a platform) is a critical metric for streaming services. This model helps identify at-risk customers so that targeted retention strategies can be implemented.

---

## 📁 Project Structure

```
churn-prediction-project/
├── churn_prediction_main.py          # Main ML pipeline (standalone script)
├── churn_streamlit_app.py            # Interactive web dashboard
├── eda_visualization.png             # EDA charts
├── model_performance.png             # Confusion matrix & accuracy
├── decision_tree.png                 # Decision tree structure
├── feature_importance.png            # Feature importance chart
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

---

## 📊 Dataset Features

The model uses **8 customer attributes** to predict churn:

| Feature | Type | Description | Range |
|---------|------|-------------|-------|
| **Age** | Numerical | Customer's age | 18-80 years |
| **Gender** | Categorical | Customer's gender | Male/Female |
| **Tenure** | Numerical | Time as customer | 0-60 months |
| **SubscriptionType** | Categorical | Plan type | Basic/Standard/Premium |
| **MonthlyCharges** | Numerical | Monthly subscription cost | $5-$25 |
| **TotalWatchHours** | Numerical | Annual content consumption | 0-1000 hours |
| **PaymentMethod** | Categorical | Payment type | Credit Card/Debit/Digital Wallet |
| **SupportTickets** | Numerical | Support requests filed | 0-10 tickets |
| **Churn** | Binary | Target variable (1=Churned, 0=Retained) | 0/1 |

---

## 🛠️ Installation & Setup

### 1. **Clone or Download the Project**
```bash
# Navigate to project directory
cd churn-prediction-project
```

### 2. **Install Dependencies**
```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### 3. **Required Libraries**
```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=0.24.0
matplotlib>=3.3.0
seaborn>=0.11.0
streamlit>=1.0.0
```

---

## 🚀 How to Use

### **Option 1: Run the Complete ML Pipeline**

This runs the entire project from scratch: data generation → preprocessing → EDA → training → evaluation.

```bash
python churn_prediction_main.py
```

**Output:**
- Console output with detailed step-by-step explanations
- 4 visualization files:
  - `eda_visualization.png` - Exploratory analysis charts
  - `model_performance.png` - Confusion matrix and accuracy
  - `decision_tree.png` - Tree structure visualization
  - `feature_importance.png` - Feature importance chart

### **Option 2: Launch Interactive Dashboard**

This starts a web application where you can input customer data and get real-time predictions.

```bash
streamlit run churn_streamlit_app.py
```

**Features:**
- 🏠 **Home** - Project overview and quick stats
- 🔮 **Make Prediction** - Input customer data and predict churn
- 📈 **Model Info** - Model parameters and performance metrics
- 📋 **Dataset Info** - Feature descriptions and statistics

---

## 📈 Project Workflow

### **Step 1: Data Generation**
```python
# Creates 1,000 synthetic customer records with realistic patterns
# Churn probability based on business logic:
# - Lower tenure → Higher churn
# - More support tickets → Higher churn
# - More watch hours → Lower churn
```

### **Step 2: Data Preprocessing**
```python
# Missing value handling
# Categorical encoding (LabelEncoder)
# Feature normalization (StandardScaler)
```

### **Step 3: Exploratory Data Analysis (EDA)**
- Churn distribution
- Churn by subscription type
- Tenure vs churn patterns
- Monthly charges vs age
- Support tickets vs churn
- Watch hours vs churn

### **Step 4: Data Splitting**
```python
# 80% Training (800 samples)
# 20% Testing (200 samples)
# Stratified split to maintain churn proportion
```

### **Step 5: Model Training**
```python
# Initial model with default parameters
# GridSearchCV for hyperparameter tuning
# Tested combinations:
#   - Criterion: ['gini', 'entropy']
#   - Max Depth: [5, 10, 15, 20, None]
#   - Min Samples Split: [2, 5, 10, 15]
#   - Min Samples Leaf: [1, 2, 4, 8]
```

### **Step 6: Model Evaluation**
```python
Metrics Calculated:
- Accuracy    (% correct predictions)
- Precision   (% predicted churn that actually churned)
- Recall      (% actual churn captured)
- F1-Score    (harmonic mean of precision & recall)
- Confusion Matrix (TP, TN, FP, FN)
```

### **Step 7: Visualization**
- Decision tree structure (max depth 3 for readability)
- Feature importance ranking
- Confusion matrix heatmap

---

## 📊 Key Results

### **Model Performance** (Typical Results)

| Metric | Score |
|--------|-------|
| **Accuracy** | ~0.82 |
| **Precision** | ~0.79 |
| **Recall** | ~0.68 |
| **F1-Score** | ~0.73 |

### **Top Predictive Features** (Example)

1. **Tenure** (23.5%) - Strong negative correlation with churn
2. **TotalWatchHours** (18.2%) - Engagement indicator
3. **SupportTickets** (15.7%) - Customer satisfaction proxy
4. **MonthlyCharges** (12.1%) - Price sensitivity
5. **SubscriptionType** (10.4%) - Plan type differences

---

## 💻 Code Examples

### **Example 1: Make Predictions Programmatically**

```python
from churn_prediction_main import (
    generate_dataset, preprocess_data, train_decision_tree
)
import pandas as pd

# Generate data
df = generate_dataset(1000)

# Preprocess
df_processed, encoders = preprocess_data(df)

# Train model
X = df_processed.drop('Churn', axis=1)
y = df_processed['Churn']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

best_model, X_test, y_test, y_pred = train_decision_tree(
    X_train, y_train, X_test, y_test
)

# Make prediction for a new customer
new_customer = pd.DataFrame({
    'Age': [35],
    'Gender': [1],  # Encoded
    'Tenure': [24],
    'SubscriptionType': [2],  # Premium
    'MonthlyCharges': [15.5],
    'TotalWatchHours': [500],
    'PaymentMethod': [0],
    'SupportTickets': [2]
})

prediction = best_model.predict(new_customer)
probability = best_model.predict_proba(new_customer)

print(f"Churn Risk: {probability[0][1]:.2%}")
```

### **Example 2: Feature Importance Analysis**

```python
import pandas as pd
import matplotlib.pyplot as plt

# Extract feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': best_model.feature_importances_
}).sort_values('Importance', ascending=False)

print(feature_importance)

# Visualize
plt.barh(feature_importance['Feature'], feature_importance['Importance'])
plt.xlabel('Importance Score')
plt.title('Feature Importance')
plt.show()
```

---

## 🔍 Understanding the Decision Tree

### **How Decision Trees Work**

The model makes binary decisions at each node:

```
                         Root Node
                    (All customers)
                            |
                ____________|____________
               |                        |
         Is Tenure < 12 months?         |
          /            \               |
        Yes             No             |
        |               |              |
     Churn Risk    Is Tickets > 5?     |
                  /          \         |
                Yes           No       |
                |             |        |
           High Risk      Check Age    |
```

**Advantages:**
✅ Easy to interpret and visualize
✅ Handles both numerical and categorical data
✅ No feature scaling required
✅ Fast predictions
✅ Identifies most important features

**Limitations:**
❌ May overfit without proper tuning
❌ Can be unstable with small data changes
❌ Biased toward dominant classes

---

## 🎯 Business Applications

### **Use Cases**

1. **Retention Campaigns**
   - Identify high-churn-risk customers
   - Target with retention offers

2. **Customer Segmentation**
   - Group customers by churn probability
   - Tailor engagement strategies

3. **Pricing Strategy**
   - Analyze charge sensitivity
   - Optimize pricing for retention

4. **Product Development**
   - Identify features that reduce churn
   - Improve content recommendations

---

## ⚙️ Advanced: Custom Configuration

### **Modify Dataset Size**
```python
# In churn_prediction_main.py
df = generate_dataset(n_samples=5000)  # Default: 1000
```

### **Change Train/Test Split**
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42  # 25% test instead of 20%
)
```

### **Adjust Hyperparameter Search**
```python
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 6, 9, 12],  # Shallower trees
    'min_samples_split': [2, 10, 20],
    'min_samples_leaf': [1, 5, 10]
}
```

---

## 📝 Troubleshooting

### **Issue: "ModuleNotFoundError: No module named 'sklearn'"**
**Solution:** Install scikit-learn
```bash
pip install scikit-learn
```

### **Issue: Streamlit app not launching**
**Solution:** Ensure Streamlit is installed and use correct command
```bash
pip install streamlit
streamlit run churn_streamlit_app.py
```

### **Issue: Visualizations not displaying**
**Solution:** Ensure matplotlib and seaborn are installed
```bash
pip install matplotlib seaborn
```

---

## 📚 Learning Resources

### **Machine Learning Concepts**
- Decision Trees: https://scikit-learn.org/stable/modules/tree.html
- Classification Metrics: https://scikit-learn.org/stable/modules/model_evaluation.html
- GridSearchCV: https://scikit-learn.org/stable/modules/grid_search.html

### **Libraries**
- Pandas Documentation: https://pandas.pydata.org/docs/
- Scikit-learn: https://scikit-learn.org/stable/
- Streamlit: https://docs.streamlit.io/

---

## 🤝 Contributing & Improvements

**Potential Enhancements:**
- [ ] Add more customer features (geographic, behavioral)
- [ ] Implement ensemble methods (Random Forest, Gradient Boosting)
- [ ] Deploy model as REST API (Flask/FastAPI)
- [ ] Add database integration for real customer data
- [ ] Implement model retraining pipeline
- [ ] Add data drift detection
- [ ] Deploy to cloud (AWS/GCP/Azure)

---

## ✨ Key Takeaways

1. **Data Quality Matters** - Good preprocessing improves model significantly
2. **Hyperparameter Tuning Works** - GridSearchCV often improves accuracy 5-10%
3. **Interpretability is Important** - Decision trees show clear decision rules
4. **Business Context is Key** - Understanding business logic improves features
5. **Monitoring is Critical** - Real models need continuous performance tracking

---

## 📬 Contact

For any queries, feedback, or collaboration, feel free to connect:

📧 **Email:** [shubhamsourav475@gmail.com](mailto:shubhamsourav475@gmail.com)

---

> 📝 **Note:**  
> This repository is maintained as part of the Capstone Project and is intended for educational use.

## 🪪 License

Distributed under the MIT License.  
© 2025 Shubham Sourav. All rights reserved.

---



**Happy Learning! 🚀**
