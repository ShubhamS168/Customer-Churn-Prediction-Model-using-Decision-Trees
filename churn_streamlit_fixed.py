"""
Streamlit App for Customer Churn Prediction
============================================

FIXED VERSION - Resolves blank screen issue

This is an interactive web application built with Streamlit that allows users
to input customer data and get real-time churn predictions from the trained model.

To run this app:
    streamlit run churn_streamlit_fixed.py

If you still see a blank screen:
    1. Run: streamlit run churn_streamlit_fixed.py --logger.level=debug
    2. Check the terminal for any error messages
    3. Try: streamlit cache clear (to clear cache)
    4. Restart the app
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION - Must be first Streamlit command
# ============================================================================
st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM STYLING
# ============================================================================
st.markdown("""
    <style>
    .main {
        padding: 2rem;
        background-color: #f8f9fa;
    }
    .stMetric {
        background-color: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton > button {
        width: 100%;
        padding: 0.75rem;
        font-size: 1rem;
        font-weight: bold;
        background-color: #2ecc71;
        color: white;
        border: none;
        border-radius: 0.5rem;
    }
    .stButton > button:hover {
        background-color: #27ae60;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_resource
def train_model():
    """
    Train the decision tree model with synthetic data.
    This is cached to improve performance and avoid retraining.
    """
    try:
        # Generate synthetic data
        np.random.seed(42)
        n_samples = 1000

        # Step 1: Generate data
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

        # Step 2: Generate churn based on business logic
        churn_probability = []
        for idx, row in df.iterrows():
            base_prob = 0.2
            base_prob -= (row['Tenure'] / 100)
            base_prob += (row['SupportTickets'] / 20)
            base_prob -= (row['TotalWatchHours'] / 2000)
            base_prob = np.clip(base_prob, 0, 1)
            churn_probability.append(base_prob)

        df['Churn'] = [1 if np.random.random() < prob else 0 for prob in churn_probability]

        # Step 3: Encode categorical variables
        le_gender = LabelEncoder()
        le_subscription = LabelEncoder()
        le_payment = LabelEncoder()

        df['Gender'] = le_gender.fit_transform(df['Gender'])
        df['SubscriptionType'] = le_subscription.fit_transform(df['SubscriptionType'])
        df['PaymentMethod'] = le_payment.fit_transform(df['PaymentMethod'])

        # Step 4: Normalize numerical features
        scaler = StandardScaler()
        numerical_cols = ['Age', 'Tenure', 'MonthlyCharges', 'TotalWatchHours', 'SupportTickets']
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

        # Step 5: Split data
        X = df.drop('Churn', axis=1)
        y = df['Churn']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Step 6: Train model with hyperparameter tuning
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

        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        return model, le_gender, le_subscription, le_payment, scaler, X.columns, X_test, y_test, {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    except Exception as e:
        st.error(f"❌ Error training model: {str(e)}")
        st.stop()

def preprocess_input(age, gender, tenure, subscription, charges, watch_hours, payment, tickets,
                     le_gender, le_subscription, le_payment, scaler):
    """Preprocess user input for prediction."""
    try:
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

    except Exception as e:
        st.error(f"❌ Error preprocessing input: {str(e)}")
        return None

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Header
    st.markdown("<h1 style='text-align: center; color: #2c3e50;'>📊 Customer Churn Prediction</h1>", 
                unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #7f8c8d;'>Streaming Entertainment Industry</h3>", 
                unsafe_allow_html=True)
    st.markdown("---")

    # Load model with spinner
    with st.spinner("⏳ Loading and training model... This may take 30-60 seconds on first run"):
        model, le_gender, le_subscription, le_payment, scaler, feature_names, X_test, y_test, metrics = train_model()

    st.success("✅ Model loaded successfully!")

    # Sidebar Navigation
    st.sidebar.markdown("### 📍 Navigation")
    page = st.sidebar.radio(
        "Select a page:",
        ["🏠 Home", "🔮 Make Prediction", "📈 Model Info", "📋 Dataset Info"],
        label_visibility="collapsed"
    )

    # ====================================================================
    # HOME PAGE
    # ====================================================================
    if page == "🏠 Home":
        st.header("Welcome to the Churn Prediction Dashboard!")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Accuracy", f"{metrics['accuracy']:.1%}")
        with col2:
            st.metric("🎯 Precision", f"{metrics['precision']:.1%}")
        with col3:
            st.metric("📈 Recall", f"{metrics['recall']:.1%}")
        with col4:
            st.metric("🔄 F1-Score", f"{metrics['f1']:.1%}")

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🎯 About This Project")
            st.write("""
            This application uses **Decision Tree Classification** to predict 
            whether a customer will churn (leave) from a streaming platform.

            **Key Features:**
            - Analyzes 8 customer attributes
            - Provides real-time predictions with confidence scores
            - Shows prediction probabilities
            - Explains key factors affecting churn
            """)

        with col2:
            st.subheader("📋 What is Churn?")
            st.write("""
            **Customer Churn** = When a customer cancels or stops using a service

            **Why It Matters:**
            - Losing customers reduces revenue
            - Acquiring new customers costs 5-25x more
            - Retaining customers increases lifetime value
            """)

        st.markdown("---")
        st.info("👉 Go to **Make Prediction** page to test the model!")

    # ====================================================================
    # PREDICTION PAGE
    # ====================================================================
    elif page == "🔮 Make Prediction":
        st.header("Make a Churn Prediction")
        st.write("Enter customer information below and click 'Predict' to get churn risk assessment.")

        with st.form("prediction_form"):
            st.subheader("📝 Customer Information")

            col1, col2 = st.columns(2)

            with col1:
                age = st.number_input("👤 Age", min_value=18, max_value=80, value=35, step=1)
                gender = st.selectbox("👫 Gender", ["Male", "Female"])
                tenure = st.slider("📅 Tenure (months)", 0, 60, 24)
                subscription = st.selectbox("💳 Subscription Type", ["Basic", "Premium", "Standard"])

            with col2:
                charges = st.number_input("💰 Monthly Charges ($)", min_value=5.0, max_value=25.0, value=15.0, step=0.5)
                watch_hours = st.slider("🎬 Total Watch Hours (annual)", 0, 1000, 300)
                payment = st.selectbox("💳 Payment Method", ["Credit Card", "Debit Card", "Digital Wallet"])
                tickets = st.slider("📞 Support Tickets (last 12 months)", 0, 10, 2)

            # Prediction button
            if st.form_submit_button("🎯 Predict Churn", use_container_width=True):
                input_data = preprocess_input(
                    age, gender, tenure, subscription, charges, watch_hours, payment, tickets,
                    le_gender, le_subscription, le_payment, scaler
                )

                if input_data is not None:
                    prediction = model.predict(input_data)[0]
                    prediction_prob = model.predict_proba(input_data)[0]

                    st.markdown("---")
                    st.subheader("📊 Prediction Results")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if prediction == 0:
                            st.success("✅ LOW CHURN RISK")
                        else:
                            st.error("⚠️ HIGH CHURN RISK")
                    with col2:
                        st.metric("No Churn Probability", f"{prediction_prob[0]:.1%}")
                    with col3:
                        st.metric("Churn Probability", f"{prediction_prob[1]:.1%}")

                    # Visualization
                    fig, ax = plt.subplots(figsize=(8, 5))
                    categories = ['No Churn', 'Churn']
                    colors = ['#2ecc71', '#e74c3c']
                    bars = ax.bar(categories, prediction_prob, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
                    ax.set_ylabel('Probability', fontweight='bold')
                    ax.set_title('Churn Prediction Probability', fontweight='bold')
                    ax.set_ylim([0, 1])

                    for bar, prob in zip(bars, prediction_prob):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                               f'{prob:.1%}', ha='center', va='bottom', fontweight='bold')

                    plt.tight_layout()
                    st.pyplot(fig)

    # ====================================================================
    # MODEL INFO PAGE
    # ====================================================================
    elif page == "📈 Model Info":
        st.header("Model Information & Performance")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🤖 Model Specifications")
            st.write(f"""
            **Algorithm:** Decision Tree Classifier
            **Features Used:** {model.n_features_in_} features
            **Tree Depth:** {model.get_depth()}
            **Number of Leaves:** {model.get_n_leaves()}
            """)

        with col2:
            st.subheader("📊 Test Set Performance")
            st.write(f"""
            **Accuracy:** {metrics['accuracy']:.2%}
            **Precision:** {metrics['precision']:.2%}
            **Recall:** {metrics['recall']:.2%}
            **F1-Score:** {metrics['f1']:.2%}
            """)

        st.markdown("---")
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
            ax.text(value + 0.01, bar.get_y() + bar.get_height()/2, f'{value:.3f}',
                   va='center', fontweight='bold')

        plt.tight_layout()
        st.pyplot(fig)

    # ====================================================================
    # DATASET INFO PAGE
    # ====================================================================
    elif page == "📋 Dataset Info":
        st.header("Dataset Information")

        st.subheader("📊 Features Description")

        features_info = {
            "Age": "Customer's age (18-80 years)",
            "Gender": "Customer's gender (Male/Female)",
            "Tenure": "Months as customer (0-60)",
            "SubscriptionType": "Plan type (Basic/Standard/Premium)",
            "MonthlyCharges": "Monthly cost ($5-$25)",
            "TotalWatchHours": "Annual watch hours (0-1000)",
            "PaymentMethod": "Payment type (3 options)",
            "SupportTickets": "Support requests (0-10)",
        }

        for feature, description in features_info.items():
            st.write(f"**{feature}:** {description}")

        st.markdown("---")
        st.subheader("💡 Key Insights")
        st.write("""
        - **Tenure Impact:** Longer-tenured customers are less likely to churn
        - **Engagement:** Higher watch hours = better retention
        - **Support Issues:** Many tickets may indicate problems
        - **Pricing:** Affects churn probability
        """)

if __name__ == "__main__":
    main()
