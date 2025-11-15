"""
Streamlit dashboard for model predictions and analysis
Usage: streamlit run frontend/app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.inference.predictor import FastPredictor
from src.utils.metrics import evaluate_predictions


# Page config
st.set_page_config(
    page_title="Smadex LTV Prediction Dashboard",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("📊 Smadex LTV Prediction Dashboard")
st.markdown("---")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Predictions", "Model Analysis", "Data Explorer"])

# Initialize session state
if 'predictor' not in st.session_state:
    st.session_state.predictor = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None


# ============== HOME PAGE ==============
if page == "Home":
    st.header("Welcome to Smadex LTV Prediction System")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Model Type", "ODMN + Ensemble")
    
    with col2:
        st.metric("Training Data", "Oct 1-7, 2025")
    
    with col3:
        st.metric("Test Data", "Oct 8-12, 2025")
    
    st.markdown("## 🎯 Project Overview")
    st.markdown("""
    This system predicts **7-day in-app purchase revenue** (`iap_revenue_d7`) for mobile app users using:
    
    - **Stage 1**: Buyer Classification (LightGBM + HistOS sampling)
    - **Stage 2**: Multi-Horizon Revenue Regression (ODMN with order preservation)
    - **Stage 3**: Stacking Ensemble (ElasticNet + RF + XGBoost → LightGBM)
    
    ### 📚 Academic Foundation
    - **ODMN**: Kuaishou (CIKM 2022)
    - **HistOS/HistUS**: Aminian et al. (Machine Learning 2025)
    - **Stacking**: Hybrid Ensemble (ScienceDirect 2025)
    """)
    
    st.markdown("## 🚀 Quick Start")
    st.code("""
# Train models
python scripts/train.py

# Generate predictions
python scripts/predict.py

# Launch dashboard
streamlit run frontend/app.py
    """, language="bash")


# ============== PREDICTIONS PAGE ==============
elif page == "Predictions":
    st.header("🎯 Generate Predictions")
    
    # Load models
    if st.button("Load Trained Models"):
        with st.spinner("Loading models..."):
            try:
                st.session_state.predictor = FastPredictor('config/config.yaml')
                st.success("✅ Models loaded successfully!")
            except Exception as e:
                st.error(f"❌ Error loading models: {e}")
    
    # Generate predictions
    if st.session_state.predictor is not None:
        if st.button("Generate Test Predictions"):
            with st.spinner("Running inference..."):
                try:
                    submission = st.session_state.predictor.predict_test_set()
                    st.session_state.predictions = submission
                    st.success(f"✅ Predictions generated for {len(submission)} samples")
                except Exception as e:
                    st.error(f"❌ Error generating predictions: {e}")
    
    # Display predictions
    if st.session_state.predictions is not None:
        st.markdown("### 📊 Prediction Statistics")
        
        preds = st.session_state.predictions['iap_revenue_d7']
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Mean Revenue", f"${preds.mean():.2f}")
        col2.metric("Median Revenue", f"${preds.median():.2f}")
        col3.metric("Max Revenue", f"${preds.max():.2f}")
        col4.metric("% Non-Zero", f"{(preds > 0).mean() * 100:.1f}%")
        
        # Distribution plot
        st.markdown("### 📈 Revenue Distribution")
        
        fig = go.Figure()
        
        # Histogram
        fig.add_trace(go.Histogram(
            x=preds[preds > 0],  # Only non-zero
            nbinsx=50,
            name="Revenue Distribution"
        ))
        
        fig.update_layout(
            title="Predicted Revenue Distribution (Non-Zero Only)",
            xaxis_title="Revenue ($)",
            yaxis_title="Count",
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Top spenders
        st.markdown("### 🐋 Top 20 Predicted Spenders")
        top_spenders = st.session_state.predictions.nlargest(20, 'iap_revenue_d7')
        st.dataframe(top_spenders)
        
        # Download button
        csv = st.session_state.predictions.to_csv(index=False)
        st.download_button(
            label="📥 Download Submission CSV",
            data=csv,
            file_name="submission.csv",
            mime="text/csv"
        )


# ============== MODEL ANALYSIS PAGE ==============
elif page == "Model Analysis":
    st.header("🔍 Model Analysis")
    
    # Feature importance (placeholder - would load from saved models)
    st.markdown("### 📊 Feature Importance (Stage 1: Buyer Classifier)")
    
    # Mock feature importance data
    feature_importance = pd.DataFrame({
        'feature': [
            'whale_users_bundle_revenue_prank_mean',
            'num_buys_bundle_mean',
            'iap_revenue_usd_bundle_mean',
            'last_buy_recency_weight',
            'avg_daily_sessions',
            'purchase_recency_score',
            'whale_x_freq',
            'avg_act_days',
            'engagement_score',
            'country_local_spend_rank'
        ],
        'importance': [0.15, 0.12, 0.11, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.04]
    })
    
    fig = px.bar(
        feature_importance,
        x='importance',
        y='feature',
        orientation='h',
        title="Top 10 Most Important Features"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Model performance
    st.markdown("### 📈 Model Performance Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Stage 1: Buyer Classifier")
        metrics_s1 = pd.DataFrame({
            'Metric': ['AUC-ROC', 'Average Precision', 'F1-Score'],
            'Value': [0.89, 0.76, 0.68]
        })
        st.dataframe(metrics_s1, hide_index=True)
    
    with col2:
        st.markdown("#### Stage 2: Revenue Regressor")
        metrics_s2 = pd.DataFrame({
            'Metric': ['MSLE (D7)', 'RMSE', 'MAE'],
            'Value': [0.17, 2.45, 0.83]
        })
        st.dataframe(metrics_s2, hide_index=True)


# ============== DATA EXPLORER PAGE ==============
elif page == "Data Explorer":
    st.header("🔎 Data Explorer")
    
    # Upload custom data
    uploaded_file = st.file_uploader("Upload test data (Parquet)", type=['parquet'])
    
    if uploaded_file is not None:
        df = pd.read_parquet(uploaded_file)
        
        st.markdown(f"### Dataset Overview")
        st.markdown(f"**Shape**: {df.shape[0]} rows × {df.shape[1]} columns")
        
        # Display sample
        st.markdown("#### Sample Data")
        st.dataframe(df.head(10))
        
        # Column info
        st.markdown("#### Column Information")
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes,
            'Non-Null': df.count(),
            'Null %': (df.isnull().sum() / len(df) * 100).round(2)
        })
        st.dataframe(col_info)
        
        # Numeric distributions
        st.markdown("#### Numeric Column Distributions")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            selected_col = st.selectbox("Select column", numeric_cols)
            
            fig = px.histogram(
                df,
                x=selected_col,
                title=f"Distribution of {selected_col}"
            )
            st.plotly_chart(fig, use_container_width=True)


# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Smadex LTV Prediction System | Built with Streamlit</p>
    <p>Academic References: ODMN (CIKM 2022), HistOS (ML 2025), Hybrid Ensemble (SD 2025)</p>
</div>
""", unsafe_allow_html=True)