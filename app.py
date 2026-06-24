import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import plotly.express as px
import plotly.figure_factory as ff

# Set page config
st.set_page_config(page_title="Customer Churn Predictor", layout="wide")

# ==========================================
# STEP 1: DATASET COLLECTION AND LOADING
# ==========================================
st.title("📊 Customer Churn Analytics & Prediction App")
st.markdown("This application analyzes customer data, trains machine learning models, and predicts potential churners.")

@st.cache_data # Caches data so it doesn't reload on every click
def load_data():
    # Replace 'churn.csv' with your actual Kaggle dataset path
    # Example assumes columns: Gender, SeniorCitizen, Partner, Dependents, Tenure, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges, Churn
    try:
        df = pd.read_csv("churn.csv")
        return df
    except FileNotFoundError:
        # Creating a dummy dataset if the file isn't present yet so the app doesn't crash
        np.random.seed(42)
        n = 1000
        dummy_data = {
            'gender': np.random.choice(['Male', 'Female'], n),
            'SeniorCitizen': np.random.choice([0, 1], n),
            'Partner': np.random.choice(['Yes', 'No'], n),
            'tenure': np.random.randint(1, 72, n),
            'PhoneService': np.random.choice(['Yes', 'No'], n),
            'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n),
            'PaperlessBilling': np.random.choice(['Yes', 'No'], n),
            'MonthlyCharges': np.random.uniform(18.25, 118.75, n),
            'TotalCharges': np.random.uniform(18.25, 8000, n),
            'Churn': np.random.choice(['Yes', 'No'], n, p=[0.2, 0.8])
        }
        return pd.DataFrame(dummy_data)

df_raw = load_data()

# ==========================================
# STEP 2: DATA PREPROCESSING
# ==========================================
# Handle missing values
df = df_raw.dropna().copy()
if 'TotalCharges' in df.columns:
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

# Separate target and features
target_col = 'Churn'
feature_cols = [col for col in df.columns if col != target_col and col != 'customerID']

X = df[feature_cols].copy()
y = df[target_col].copy()

# Encode Categorical Variables
le_dict = {}
for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    le_dict[col] = le

# Encode Target
target_le = LabelEncoder()
y = target_le.fit_transform(y) # No = 0, Yes = 1

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numeric data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# STEP 3 & 4: MODEL BUILDING & EVALUATION
# ==========================================
@st.cache_resource # Cache models to prevent retraining on user inputs
def train_models(X_t, X_v, y_t, y_v):
    models = {
        "Logistic Regression": LogisticRegression(random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_t, y_t)
        preds = model.predict(X_v)
        acc = accuracy_score(y_v, preds)
        cm = confusion_matrix(y_v, preds)
        results[name] = {"model": model, "accuracy": acc, "cm": cm}
    
    return results

results = train_models(X_train_scaled, X_test_scaled, y_train, y_test)

# Determine best model dynamically
best_model_name = max(results, key=lambda k: results[k]['accuracy'])
best_model = results[best_model_name]['model']

# ==========================================
# STEP 5: STREAMLIT INTERFACE DESIGN
# ==========================================
tabs = st.tabs(["📋 Dataset Overview", "📈 Model Performance", "🔮 Predict Churn"])

# --- TAB 1: DATASET OVERVIEW ---
with tabs[0]:
    st.subheader("Dataset Preview")
    st.dataframe(df_raw.head(10), use_container_width=True)
    
    st.subheader("Data Insights")
    col1, col2 = st.columns(2)
    with col1:
        churn_counts = df_raw[target_col].value_counts().reset_index()
        fig_pie = px.pie(churn_counts, names=target_col, values='count', title="Overall Churn Distribution", color=target_col, color_discrete_sequence=["#2ecc71", "#e74c3c"])
        st.plotly_chart(fig_pie, use_container_width=True)
    with col2:
        if 'tenure' in df_raw.columns:
            fig_hist = px.histogram(df_raw, x="tenure", color=target_col, barmode="group", title="Tenure vs Churn")
            st.plotly_chart(fig_hist, use_container_width=True)

# --- TAB 2: MODEL PERFORMANCE (STEP 7 VISUALIZATIONS) ---
with tabs[1]:
    st.subheader("Model Comparison")
    
    # Accuracy Comparison Chart
    acc_df = pd.DataFrame({
        "Model": results.keys(),
        "Accuracy": [results[m]["accuracy"] for m in results.keys()]
    })
    fig_acc = px.bar(acc_df, x="Model", y="Accuracy", text_auto='.2%', color="Model", title="Model Accuracy Comparison")
    st.plotly_chart(fig_acc, use_container_width=True)
    
    st.success(f"🏆 **Best Performing Model:** {best_model_name} ({results[best_model_name]['accuracy']:.2%} Accuracy)")
    
    # Confusion Matrix Heatmap for Best Model
    st.subheader(f"Confusion Matrix: {best_model_name}")
    cm = results[best_model_name]["cm"]
    fig_cm = ff.create_annotated_heatmap(
        z=cm, 
        x=["Predicted No Churn", "Predicted Churn"], 
        y=["Actual No Churn", "Actual Churn"], 
        colorscale='Viridis'
    )
    st.plotly_chart(fig_cm, use_container_width=True)
    
    # Feature Importance (For Tree models or coefficients for LR)
    st.subheader("Feature Importance")
    if best_model_name in ["Decision Tree", "Random Forest"]:
        importances = best_model.feature_importances_
    else:
        importances = np.abs(best_model.coef_[0])
        
    feat_df = pd.DataFrame({"Feature": feature_cols, "Importance": importances}).sort_values(by="Importance", ascending=False)
    fig_feat = px.bar(feat_df, x="Importance", y="Feature", orientation='h', title=f"Feature Significance ({best_model_name})")
    st.plotly_chart(fig_feat, use_container_width=True)

# --- TAB 3: PREDICTION SYSTEM (STEP 6) ---
with tabs[2]:
    st.subheader("Predict Customer Churn Risk")
    st.markdown("Adjust the inputs below to calculate the real-time probability of a customer leaving.")
    
    # Create input form dynamically based on features
    input_data = {}
    form_cols = st.columns(3)
    
    for idx, col in enumerate(feature_cols):
        col_side = form_cols[idx % 3]
        if col in le_dict: # Categorical feature
            options = list(le_dict[col].classes_)
            input_data[col] = col_side.selectbox(f"{col}", options)
        else: # Numeric feature
            min_val = float(df[col].min())
            max_val = float(df[col].max())
            mean_val = float(df[col].mean())
            input_data[col] = col_side.slider(f"{col}", min_val, max_val, mean_val)
            
    if st.button("Run Churn Diagnostic", type="primary"):
        # Process input payload
        input_df = pd.DataFrame([input_data])
        
        # Encode inputs exactly like training
        for col, le in le_dict.items():
            input_df[col] = le.transform(input_df[col])
            
        # Scale inputs
        input_scaled = scaler.transform(input_df)
        
        # Predict
        prediction = best_model.predict(input_scaled)[0]
        probability = best_model.predict_proba(input_scaled)[0][1]
        
        # Display Diagnostic Outcome
        st.markdown("---")
        if prediction == 1:
            st.error(f"🚨 **High Risk Warning:** This customer is likely to churn. (Probability: {probability:.1%})")
        else:
            st.success(f"✅ **Low Risk:** This customer is stable. (Retention Probability: {(1-probability):.1%})")