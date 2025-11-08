"""
Streamlit App for Customer Churn Prediction
============================================

This is a simple interactive web application built with Streamlit that allows users
to input customer data and get real-time churn predictions from the trained model.

To run this app:
    streamlit run churn_prediction_streamlit.py

Features:
- Interactive input form for customer data
- Real-time predictions
- Probability visualization
- Model explanation
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Set page configuration
st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .stButton > button {
        width: 100%;
        padding: 0.75rem;
        font-size: 1rem;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# Helper Functions
# ============================================================================

@st.cache_resource
def train_model():
    """
    Train the decision tree model with synthetic data.
    This is cached to improve performance.
    """
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'Age': np.random.randint(18, 75, n_samples),
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'Tenure': np.random.randint(0, 60, n_samples),
        'SubscriptionType': np.random.choice(['Basic', 'Premium', 'Standard'], n_samples),
        'MonthlyCharges': np.random.uniform(5, 25, n_samples),
        'TotalWatchHours': np.random.randint(0, 1000, n_samples),
        'PaymentMethod': np.random.choice(['Credit Card', 'Debit Card', 'Digital Wallet'], n_samples),
        'SupportTickets': np.random.randint(0, 10, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Generate churn labels
    churn_probability = []
    for idx, row in df.iterrows():
        base_prob = 0.2
        base_prob -= (row['Tenure'] / 100)
        base_prob += (row['SupportTickets'] / 20)
        base_prob -= (row['TotalWatchHours'] / 2000)
        base_prob = np.clip(base_prob, 0, 1)
        churn_probability.append(base_prob)
    
    df['Churn'] = [1 if np.random.random() < prob else 0 for prob in churn_probability]
    
    # Preprocess data
    le_gender = LabelEncoder()
    le_subscription = LabelEncoder()
    le_payment = LabelEncoder()
    
    df['Gender'] = le_gender.fit_transform(df['Gender'])
    df['SubscriptionType'] = le_subscription.fit_transform(df['SubscriptionType'])
    df['PaymentMethod'] = le_payment.fit_transform(df['PaymentMethod'])
    
    scaler = StandardScaler()
    numerical_cols = ['Age', 'Tenure', 'MonthlyCharges', 'TotalWatchHours', 'SupportTickets']
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    # Split and train
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model with hyperparameter tuning
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [5, 10, 15, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    gs_cv = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1
    )
    
    gs_cv.fit(X_train, y_train)
    model = gs_cv.best_estimator_
    
    return model, le_gender, le_subscription, le_payment, scaler, X.columns, X_test, y_test

def preprocess_input(age, gender, tenure, subscription, charges, watch_hours, payment, tickets,
                     le_gender, le_subscription, le_payment, scaler):
    """Preprocess user input for prediction."""
    
    # Encode categorical variables
    gender_encoded = le_gender.transform([gender])[0]
    subscription_encoded = le_subscription.transform([subscription])[0]
    payment_encoded = le_payment.transform([payment])[0]
    
    # Create DataFrame
    input_data = pd.DataFrame({
        'Age': [age],
        'Gender': [gender_encoded],
        'Tenure': [tenure],
        'SubscriptionType': [subscription_encoded],
        'MonthlyCharges': [charges],
        'TotalWatchHours': [watch_hours],
        'PaymentMethod': [payment_encoded],
        'SupportTickets': [tickets]
    })
    
    # Normalize numerical features
    numerical_cols = ['Age', 'Tenure', 'MonthlyCharges', 'TotalWatchHours', 'SupportTickets']
    input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])
    
    return input_data

# ============================================================================
# Main Streamlit App
# ============================================================================

def main():
    # Header
    st.title("📊 Customer Churn Prediction Dashboard")
    st.markdown("---")
    
    # Load model
    model, le_gender, le_subscription, le_payment, scaler, feature_names, X_test, y_test = train_model()
    
    # Sidebar for navigation
    page = st.sidebar.radio("Navigation", ["🏠 Home", "🔮 Make Prediction", "📈 Model Info", "📋 Dataset Info"])
    
    # ========================================================================
    # HOME PAGE
    # ========================================================================
    if page == "🏠 Home":
        st.header("Welcome to Customer Churn Prediction System")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Project Overview")
            st.write("""
            This application uses **Decision Tree Classification** to predict 
            whether a customer will churn (leave) from a streaming platform.
            
            **Key Features:**
            - Analyzes 8 customer attributes
            - Provides real-time predictions
            - Shows prediction confidence
            - Explains key factors affecting churn
            """)
        
        with col2:
            st.subheader("📊 Quick Stats")
            model_accuracy = model.score(X_test, y_test)
            st.metric("Model Accuracy", f"{model_accuracy:.2%}")
            
            y_pred = model.predict(X_test)
            from sklearn.metrics import precision_score, recall_score, f1_score
            st.metric("Precision", f"{precision_score(y_test, y_pred):.2%}")
            st.metric("Recall", f"{recall_score(y_test, y_pred):.2%}")
            st.metric("F1-Score", f"{f1_score(y_test, y_pred):.2%}")
    
    # ========================================================================
    # PREDICTION PAGE
    # ========================================================================
    elif page == "🔮 Make Prediction":
        st.header("Make a Churn Prediction")
        
        # Create form
        with st.form("prediction_form"):
            st.subheader("Customer Information")
            
            col1, col2 = st.columns(2)
            
            with col1:
                age = st.number_input("Age", min_value=18, max_value=80, value=35, step=1)
                gender = st.selectbox("Gender", ["Male", "Female"])
                tenure = st.slider("Tenure (months)", 0, 60, 24)
                subscription = st.selectbox("Subscription Type", ["Basic", "Premium", "Standard"])
            
            with col2:
                charges = st.number_input("Monthly Charges ($)", min_value=5.0, max_value=25.0, value=15.0, step=0.5)
                watch_hours = st.slider("Total Watch Hours (annual)", 0, 1000, 300)
                payment = st.selectbox("Payment Method", ["Credit Card", "Debit Card", "Digital Wallet"])
                tickets = st.slider("Support Tickets (last 12 months)", 0, 10, 2)
            
            # Prediction button
            if st.form_submit_button("🎯 Predict Churn", use_container_width=True):
                # Preprocess input
                input_data = preprocess_input(
                    age, gender, tenure, subscription, charges, watch_hours, payment, tickets,
                    le_gender, le_subscription, le_payment, scaler
                )
                
                # Make prediction
                prediction = model.predict(input_data)[0]
                prediction_prob = model.predict_proba(input_data)[0]
                
                # Display results
                st.markdown("---")
                st.subheader("📊 Prediction Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if prediction == 0:
                        st.success("✅ No Churn Expected", icon="✅")
                    else:
                        st.error("⚠️ Churn Risk Detected", icon="⚠️")
                
                with col2:
                    st.metric("No Churn Probability", f"{prediction_prob[0]:.2%}")
                
                with col3:
                    st.metric("Churn Probability", f"{prediction_prob[1]:.2%}")
                
                # Visualization
                fig, ax = plt.subplots(figsize=(8, 5))
                categories = ['No Churn', 'Churn']
                colors = ['#2ecc71', '#e74c3c']
                bars = ax.bar(categories, prediction_prob, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
                ax.set_ylabel('Probability', fontweight='bold')
                ax.set_title('Churn Prediction Probability', fontweight='bold', fontsize=12)
                ax.set_ylim([0, 1])
                
                for bar, prob in zip(bars, prediction_prob):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                           f'{prob:.2%}', ha='center', va='bottom', fontweight='bold', fontsize=11)
                
                st.pyplot(fig)
                
                # Key insights
                st.subheader("🔍 Key Factors")
                feature_importance = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                st.write("**Top Features Influencing Churn:**")
                for idx, row in feature_importance.head(5).iterrows():
                    st.write(f"  • {row['Feature']}: {row['Importance']:.4f}")
    
    # ========================================================================
    # MODEL INFO PAGE
    # ========================================================================
    elif page == "📈 Model Info":
        st.header("Model Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🤖 Model Details")
            st.write(f"""
            **Algorithm:** Decision Tree Classifier
            
            **Features Used:** {model.n_features_in_}
            - Age
            - Gender
            - Tenure
            - Subscription Type
            - Monthly Charges
            - Total Watch Hours
            - Payment Method
            - Support Tickets
            
            **Tree Depth:** {model.get_depth()}
            **Number of Leaves:** {model.get_n_leaves()}
            """)
        
        with col2:
            st.subheader("📊 Performance Metrics")
            y_pred = model.predict(X_test)
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            st.write(f"""
            **Accuracy:** {accuracy_score(y_test, y_pred):.4f}
            **Precision:** {precision_score(y_test, y_pred):.4f}
            **Recall:** {recall_score(y_test, y_pred):.4f}
            **F1-Score:** {f1_score(y_test, y_pred):.4f}
            """)
        
        # Feature importance
        st.subheader("🎯 Feature Importance")
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.viridis(np.linspace(0, 1, len(feature_importance)))
        bars = ax.barh(feature_importance['Feature'], feature_importance['Importance'], color=colors)
        ax.set_xlabel('Importance Score', fontweight='bold')
        ax.set_title('Feature Importance in Decision Tree Model', fontweight='bold')
        ax.invert_yaxis()
        
        for bar, value in zip(bars, feature_importance['Importance']):
            ax.text(value + 0.01, bar.get_y() + bar.get_height()/2, f'{value:.4f}', 
                   va='center', fontweight='bold')
        
        st.pyplot(fig)
        
        st.dataframe(feature_importance, use_container_width=True)
    
    # ========================================================================
    # DATASET INFO PAGE
    # ========================================================================
    elif page == "📋 Dataset Info":
        st.header("Dataset Information")
        
        st.subheader("📊 Feature Descriptions")
        
        feature_descriptions = {
            "Age": "Customer's age in years (18-80)",
            "Gender": "Customer's gender (Male/Female)",
            "Tenure": "How long the customer has been with the platform (0-60 months)",
            "SubscriptionType": "Type of subscription (Basic/Standard/Premium)",
            "MonthlyCharges": "Monthly subscription cost in dollars ($5-$25)",
            "TotalWatchHours": "Total hours watched annually (0-1000 hours)",
            "PaymentMethod": "Payment method used (Credit Card/Debit Card/Digital Wallet)",
            "SupportTickets": "Number of support tickets filed (0-10)",
            "Churn": "Target variable - whether customer churned (Yes/No)"
        }
        
        for feature, description in feature_descriptions.items():
            st.write(f"**{feature}:** {description}")
        
        st.subheader("📈 Dataset Statistics")
        
        # Generate sample dataset for display
        np.random.seed(42)
        n_samples = 100
        
        data = {
            'Age': np.random.randint(18, 75, n_samples),
            'Gender': np.random.choice(['Male', 'Female'], n_samples),
            'Tenure': np.random.randint(0, 60, n_samples),
            'SubscriptionType': np.random.choice(['Basic', 'Premium', 'Standard'], n_samples),
            'MonthlyCharges': np.random.uniform(5, 25, n_samples),
            'TotalWatchHours': np.random.randint(0, 1000, n_samples),
            'PaymentMethod': np.random.choice(['Credit Card', 'Debit Card', 'Digital Wallet'], n_samples),
            'SupportTickets': np.random.randint(0, 10, n_samples),
        }
        
        df_sample = pd.DataFrame(data)
        
        st.write(df_sample.describe())
        
        st.subheader("💡 Business Insights")
        st.write("""
        - **Tenure Impact:** Longer-tenured customers are less likely to churn
        - **Support Tickets:** High number of support tickets indicates potential dissatisfaction
        - **Engagement:** Total watch hours show customer engagement level
        - **Subscription Type:** Different subscription types may have different churn rates
        - **Payment Method:** Some payment methods may correlate with churn behavior
        """)

if __name__ == "__main__":
    main()
