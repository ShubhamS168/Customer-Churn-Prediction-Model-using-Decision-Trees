# ==============================================================================
# CUSTOMER CHURN PREDICTION MODEL - STREAMING ENTERTAINMENT INDUSTRY
# Decision Tree Classification Project
# ==============================================================================

"""
This is a complete machine learning project for predicting customer churn
in a streaming entertainment platform using Decision Tree Classifier.

Features:
- Synthetic data generation
- Data preprocessing and cleaning
- Exploratory Data Analysis (EDA)
- Hyperparameter tuning with GridSearchCV
- Model evaluation with multiple metrics
- Visualization of decision tree and feature importance
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (accuracy_score, classification_report, 
                            confusion_matrix, f1_score, precision_score, 
                            recall_score)
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# ==============================================================================
# STEP 1: GENERATE SYNTHETIC DATASET
# ==============================================================================
def generate_dataset(n_samples=1000):
    """
    Generate synthetic customer data for a streaming platform.
    
    Parameters:
    -----------
    n_samples : int
        Number of customer records to generate
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing customer data with churn label
    """
    print("[STEP 1] Generating Synthetic Dataset for Streaming Platform")
    print("-" * 80)
    
    # Create base customer features
    data = {
        'CustomerID': [f'CUST_{i:05d}' for i in range(1, n_samples + 1)],
        'Age': np.random.randint(18, 75, n_samples),
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'Tenure': np.random.randint(0, 60, n_samples),  # in months
        'SubscriptionType': np.random.choice(['Basic', 'Premium', 'Standard'], n_samples),
        'MonthlyCharges': np.random.uniform(5, 25, n_samples),
        'TotalWatchHours': np.random.randint(0, 1000, n_samples),
        'PaymentMethod': np.random.choice(['Credit Card', 'Debit Card', 'Digital Wallet'], n_samples),
        'SupportTickets': np.random.randint(0, 10, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Generate churn based on business logic
    # Business Rules:
    # - Lower tenure -> higher churn probability
    # - Lower watch hours -> higher churn probability
    # - More support tickets -> higher churn probability (indicates dissatisfaction)
    churn_probability = []
    for idx, row in df.iterrows():
        base_prob = 0.2
        base_prob -= (row['Tenure'] / 100)  # Longer tenure reduces churn
        base_prob += (row['SupportTickets'] / 20)  # More support increases churn
        base_prob -= (row['TotalWatchHours'] / 2000)  # More watch hours reduces churn
        base_prob = np.clip(base_prob, 0, 1)  # Keep probability between 0 and 1
        churn_probability.append(base_prob)
    
    df['Churn'] = [1 if np.random.random() < prob else 0 for prob in churn_probability]
    
    # Reorder columns for clarity
    churn_col = df.pop('Churn')
    df.insert(9, 'Churn', churn_col)
    
    print(f"✓ Dataset created with {n_samples} samples")
    print(f"\nDataset Shape: {df.shape}")
    print(f"\nFirst 10 rows:")
    print(df.head(10))
    print(f"\nChurn Distribution:")
    print(df['Churn'].value_counts())
    print(f"Churn Rate: {df['Churn'].mean()*100:.2f}%")
    
    return df

# ==============================================================================
# STEP 2: DATA PREPROCESSING
# ==============================================================================
def preprocess_data(df):
    """
    Clean and preprocess the dataset.
    
    Performs:
    - Missing value handling
    - Categorical encoding
    - Feature normalization
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw dataset
        
    Returns:
    --------
    tuple : (pd.DataFrame, dict)
        Processed DataFrame and encoders dictionary
    """
    print("\n[STEP 2] Data Preprocessing & Cleaning")
    print("-" * 80)
    
    df_processed = df.copy()
    
    # Check for missing values
    print("Missing Values Check:")
    print(df_processed.isnull().sum())
    
    # Drop CustomerID (not useful for prediction)
    df_processed = df_processed.drop('CustomerID', axis=1)
    print("✓ Dropped CustomerID column")
    
    # Encode categorical variables
    print("\nEncoding Categorical Variables:")
    encoders = {}
    
    le_gender = LabelEncoder()
    le_subscription = LabelEncoder()
    le_payment = LabelEncoder()
    
    df_processed['Gender'] = le_gender.fit_transform(df_processed['Gender'])
    encoders['Gender'] = le_gender
    print(f"  - Gender: {dict(zip(le_gender.classes_, le_gender.transform(le_gender.classes_)))}")
    
    df_processed['SubscriptionType'] = le_subscription.fit_transform(df_processed['SubscriptionType'])
    encoders['SubscriptionType'] = le_subscription
    print(f"  - SubscriptionType: {dict(zip(le_subscription.classes_, le_subscription.transform(le_subscription.classes_)))}")
    
    df_processed['PaymentMethod'] = le_payment.fit_transform(df_processed['PaymentMethod'])
    encoders['PaymentMethod'] = le_payment
    print(f"  - PaymentMethod: {dict(zip(le_payment.classes_, le_payment.transform(le_payment.classes_)))}")
    
    # Normalize numerical features
    print("\nNormalizing Numerical Features:")
    scaler = StandardScaler()
    numerical_cols = ['Age', 'Tenure', 'MonthlyCharges', 'TotalWatchHours', 'SupportTickets']
    df_processed[numerical_cols] = scaler.fit_transform(df_processed[numerical_cols])
    encoders['scaler'] = scaler
    encoders['numerical_cols'] = numerical_cols
    print(f"✓ Normalized columns: {numerical_cols}")
    
    print(f"\nProcessed Dataset (first 5 rows):")
    print(df_processed.head())
    
    return df_processed, encoders

# ==============================================================================
# STEP 3: EXPLORATORY DATA ANALYSIS
# ==============================================================================
def perform_eda(df):
    """
    Perform exploratory data analysis and create visualizations.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Original dataset (before preprocessing)
    """
    print("\n[STEP 3] Exploratory Data Analysis (EDA)")
    print("-" * 80)
    
    print("\n📊 Creating EDA Visualizations...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Customer Churn - Exploratory Data Analysis', fontsize=16, fontweight='bold')
    
    # 1. Churn Distribution
    churn_counts = df['Churn'].value_counts()
    axes[0, 0].bar(['No Churn', 'Churn'], churn_counts.values, color=['#2ecc71', '#e74c3c'])
    axes[0, 0].set_title('Churn Distribution', fontweight='bold')
    axes[0, 0].set_ylabel('Count')
    for i, v in enumerate(churn_counts.values):
        axes[0, 0].text(i, v + 10, str(v), ha='center', fontweight='bold')
    
    # 2. Churn by Subscription Type
    churn_by_sub = pd.crosstab(df['SubscriptionType'], df['Churn'])
    churn_by_sub.plot(kind='bar', ax=axes[0, 1], color=['#2ecc71', '#e74c3c'])
    axes[0, 1].set_title('Churn by Subscription Type', fontweight='bold')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].legend(['No Churn', 'Churn'])
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Tenure vs Churn
    axes[0, 2].hist([df[df['Churn']==0]['Tenure'], df[df['Churn']==1]['Tenure']], 
                    label=['No Churn', 'Churn'], bins=20, color=['#2ecc71', '#e74c3c'])
    axes[0, 2].set_title('Tenure vs Churn', fontweight='bold')
    axes[0, 2].set_xlabel('Tenure (months)')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].legend()
    
    # 4. Monthly Charges vs Age
    axes[1, 0].scatter(df[df['Churn']==0]['MonthlyCharges'], 
                       df[df['Churn']==0]['Age'], alpha=0.5, label='No Churn', color='#2ecc71')
    axes[1, 0].scatter(df[df['Churn']==1]['MonthlyCharges'], 
                       df[df['Churn']==1]['Age'], alpha=0.5, label='Churn', color='#e74c3c')
    axes[1, 0].set_title('Monthly Charges vs Age', fontweight='bold')
    axes[1, 0].set_xlabel('Monthly Charges ($)')
    axes[1, 0].set_ylabel('Age')
    axes[1, 0].legend()
    
    # 5. Support Tickets vs Churn
    axes[1, 1].boxplot([df[df['Churn']==0]['SupportTickets'], 
                         df[df['Churn']==1]['SupportTickets']], 
                        labels=['No Churn', 'Churn'],
                        patch_artist=True,
                        boxprops=dict(facecolor='#3498db', alpha=0.7))
    axes[1, 1].set_title('Support Tickets vs Churn', fontweight='bold')
    axes[1, 1].set_ylabel('Number of Tickets')
    
    # 6. Total Watch Hours vs Churn
    axes[1, 2].boxplot([df[df['Churn']==0]['TotalWatchHours'], 
                         df[df['Churn']==1]['TotalWatchHours']], 
                        labels=['No Churn', 'Churn'],
                        patch_artist=True,
                        boxprops=dict(facecolor='#9b59b6', alpha=0.7))
    axes[1, 2].set_title('Total Watch Hours vs Churn', fontweight='bold')
    axes[1, 2].set_ylabel('Watch Hours')
    
    plt.tight_layout()
    plt.savefig('eda_visualization.png', dpi=300, bbox_inches='tight')
    print("✓ EDA visualization saved as 'eda_visualization.png'")
    plt.show()
    
    # Correlation analysis
    print("\n📊 Feature Correlations with Churn:")
    correlation_data = df[['Age', 'Tenure', 'MonthlyCharges', 'TotalWatchHours', 'SupportTickets', 'Churn']].corr()
    print(correlation_data['Churn'].sort_values(ascending=False))

# ==============================================================================
# STEP 4: MODEL TRAINING
# ==============================================================================
def train_decision_tree(X_train, y_train, X_test, y_test):
    """
    Train and tune decision tree model using GridSearchCV.
    
    Parameters:
    -----------
    X_train, y_train : Training data
    X_test, y_test : Test data
    
    Returns:
    --------
    tuple : (best_model, X_test, y_test, y_pred_test)
    """
    print("\n[STEP 4] Training Initial Decision Tree Model")
    print("-" * 80)
    
    # Train initial model
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, y_train)
    y_pred_initial = dt_model.predict(X_test)
    
    print(f"✓ Initial Decision Tree Model trained")
    print(f"Initial Testing Accuracy: {accuracy_score(y_test, y_pred_initial):.4f}")
    
    # Hyperparameter tuning
    print("\n[STEP 5] Hyperparameter Tuning using GridSearchCV")
    print("-" * 80)
    
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [5, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10, 15],
        'min_samples_leaf': [1, 2, 4, 8]
    }
    
    print("🔍 Searching for optimal hyperparameters...")
    
    gs_cv = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=0
    )
    
    gs_cv.fit(X_train, y_train)
    
    print(f"✓ Grid Search completed")
    print(f"\nBest Parameters Found:")
    for param, value in gs_cv.best_params_.items():
        print(f"  - {param}: {value}")
    print(f"Best Cross-validation F1 Score: {gs_cv.best_score_:.4f}")
    
    best_model = gs_cv.best_estimator_
    y_pred_test = best_model.predict(X_test)
    
    return best_model, X_test, y_test, y_pred_test

# ==============================================================================
# STEP 5: MODEL EVALUATION
# ==============================================================================
def evaluate_model(y_test, y_pred, model_name="Model"):
    """
    Evaluate model performance with multiple metrics.
    
    Parameters:
    -----------
    y_test : true labels
    y_pred : predicted labels
    model_name : str, name for display
    """
    print(f"\n[STEP 6] Model Performance Evaluation - {model_name}")
    print("-" * 80)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"\n📊 Key Metrics:")
    print(f"  - Accuracy:  {acc:.4f}")
    print(f"  - Precision: {prec:.4f}")
    print(f"  - Recall:    {rec:.4f}")
    print(f"  - F1-Score:  {f1:.4f}")
    
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))
    
    print(f"\n🔲 Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print(f"\n  True Negatives:  {cm[0, 0]}")
    print(f"  False Positives: {cm[0, 1]}")
    print(f"  False Negatives: {cm[1, 0]}")
    print(f"  True Positives:  {cm[1, 1]}")
    
    return cm

# ==============================================================================
# STEP 6: VISUALIZATIONS
# ==============================================================================
def create_visualizations(best_model, X_test, y_test, y_pred, cm):
    """
    Create model visualization plots.
    
    Parameters:
    -----------
    best_model : trained model
    X_test : test features
    y_test : test labels
    y_pred : predictions
    cm : confusion matrix
    """
    print("\n[STEP 7] Creating Visualizations")
    print("-" * 80)
    
    # 1. Confusion Matrix and Model Comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[0],
                xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
    axes[0].set_title('Confusion Matrix (Test Set)', fontweight='bold', fontsize=12)
    axes[0].set_ylabel('Actual')
    axes[0].set_xlabel('Predicted')
    
    acc = accuracy_score(y_test, y_pred)
    axes[1].bar(['Model Accuracy'], [acc], color='#2ecc71', alpha=0.7, edgecolor='black', linewidth=2)
    axes[1].set_title('Model Accuracy on Test Set', fontweight='bold', fontsize=12)
    axes[1].set_ylabel('Accuracy')
    axes[1].set_ylim([0, 1])
    axes[1].text(0, acc + 0.02, f'{acc:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('model_performance.png', dpi=300, bbox_inches='tight')
    print("✓ Model performance visualization saved")
    plt.show()
    
    # 2. Decision Tree Visualization
    print("\n🌳 Generating Decision Tree Visualization...")
    
    fig, ax = plt.subplots(figsize=(20, 12))
    plot_tree(best_model, 
              feature_names=X_test.columns.tolist(),
              class_names=['No Churn', 'Churn'],
              filled=True,
              ax=ax,
              rounded=True,
              fontsize=10,
              max_depth=3)
    plt.title('Decision Tree Classifier (Optimized Model - Max Depth: 3)', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig('decision_tree.png', dpi=300, bbox_inches='tight')
    print("✓ Decision tree visualization saved")
    plt.show()
    
    # 3. Feature Importance
    print("\n📊 Extracting Feature Importance...")
    
    feature_importance = pd.DataFrame({
        'Feature': X_test.columns,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nFeature Importance (Top 10):")
    print(feature_importance.head(10).to_string(index=False))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors_importance = plt.cm.viridis(np.linspace(0, 1, len(feature_importance)))
    bars = ax.barh(feature_importance['Feature'], feature_importance['Importance'], color=colors_importance)
    ax.set_xlabel('Importance Score', fontweight='bold')
    ax.set_title('Feature Importance in Decision Tree Model', fontweight='bold', fontsize=12)
    ax.invert_yaxis()
    
    for i, (bar, value) in enumerate(zip(bars, feature_importance['Importance'])):
        ax.text(value + 0.01, bar.get_y() + bar.get_height()/2, f'{value:.4f}', 
                va='center', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    print("✓ Feature importance chart saved")
    plt.show()
    
    return feature_importance

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("CUSTOMER CHURN PREDICTION MODEL - STREAMING ENTERTAINMENT INDUSTRY")
    print("=" * 80)
    
    # Step 1: Generate dataset
    df = generate_dataset(n_samples=1000)
    
    # Step 2: Preprocess data
    df_processed, encoders = preprocess_data(df)
    
    # Step 3: EDA
    perform_eda(df)
    
    # Step 4: Prepare data for modeling
    print("\n[STEP 4] Splitting Data into Training and Test Sets")
    print("-" * 80)
    
    X = df_processed.drop('Churn', axis=1)
    y = df_processed['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"✓ Data split completed:")
    print(f"  - Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
    print(f"  - Testing set: {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
    print(f"  - Features: {X_train.shape[1]}")
    
    # Step 5: Train model
    best_model, X_test, y_test, y_pred = train_decision_tree(X_train, y_train, X_test, y_test)
    
    # Step 6: Evaluate model
    cm = evaluate_model(y_test, y_pred, "Optimized Decision Tree")
    
    # Step 7: Create visualizations
    feature_importance = create_visualizations(best_model, X_test, y_test, y_pred, cm)
    
    print("\n" + "=" * 80)
    print("✅ MODEL TRAINING AND EVALUATION COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("\nGenerated Files:")
    print("  1. eda_visualization.png - Exploratory Data Analysis plots")
    print("  2. model_performance.png - Confusion matrix and accuracy")
    print("  3. decision_tree.png - Decision tree structure visualization")
    print("  4. feature_importance.png - Feature importance bar chart")
