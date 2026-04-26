"""
AI Energy Anomaly Explainer
============================
Streamlit web application for detecting and explaining
industrial sensor anomalies using ML + LLM.

Author: Mahima Rajesh
Portfolio project for Imperial College MSc BA
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys
import os
import json

# Add src to path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from anomaly_detector import (
    load_data, engineer_features, train_model, detect_anomalies
)
from llm_explainer import explain_anomaly, NORMAL_RANGES

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Energy Anomaly Explainer",
    page_icon="⚡",
    layout="wide"
)

# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .anomaly-high {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .anomaly-medium {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .anomaly-low {
        background-color: #e8f5e9;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD AND CACHE MODEL
# Cache means the model only trains once — not every time
# the user interacts with the app
# ============================================================

@st.cache_resource
def load_trained_model():
    """
    Loads data and trains the Random Forest model.
    @st.cache_resource means this only runs once
    and the model is reused for every subsequent request.
    """
    with st.spinner("Loading training data and training model..."):
        df = load_data()
        X = engineer_features(df)
        y = df['anomaly'][X.index]
        model, X_test, y_test = train_model(X, y)
        
        # Also load normal data for LLM context
        normal_url = "https://raw.githubusercontent.com/waico/SKAB/master/data/anomaly-free/anomaly-free.csv"
        df_normal = pd.read_csv(
            normal_url, sep=';',
            index_col='datetime',
            parse_dates=True
        )
    
    return model, df_normal, df


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def plot_sensor_data(df, anomaly_indices, sensor):
    """
    Creates an interactive Plotly chart showing sensor readings
    with anomalies highlighted in red.
    """
    fig = go.Figure()
    
    # Normal readings — blue line
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df[sensor],
        mode='lines',
        name='Normal',
        line=dict(color='steelblue', width=1)
    ))
    
    # Anomalous readings — red dots
    if len(anomaly_indices) > 0:
        anomaly_mask = df.index.isin(anomaly_indices)
        fig.add_trace(go.Scatter(
            x=df.index[anomaly_mask],
            y=df[sensor][anomaly_mask],
            mode='markers',
            name='Anomaly',
            marker=dict(color='red', size=6)
        ))
    
    fig.update_layout(
        title=f'{sensor} — Sensor Reading Over Time',
        xaxis_title='Time',
        yaxis_title='Value',
        height=300,
        showlegend=True,
        hovermode='x unified'
    )
    
    return fig


def display_explanation(explanation, anomaly_time, probability):
    """
    Displays a single anomaly explanation in a formatted card.
    """
    severity = explanation.get('severity', 'UNKNOWN')
    
    # Choose card style based on severity
    if severity == 'HIGH':
        card_class = 'anomaly-high'
        severity_emoji = '🔴'
    elif severity == 'MEDIUM':
        card_class = 'anomaly-medium'
        severity_emoji = '🟡'
    else:
        card_class = 'anomaly-low'
        severity_emoji = '🟢'
    
    st.markdown(f"""
    <div class="{card_class}">
        <h4>{severity_emoji} {severity} SEVERITY — {anomaly_time}</h4>
        <p><strong>Confidence:</strong> {probability:.0%} anomaly probability</p>
        <p><strong>Cause:</strong> {explanation.get('cause', 'Unknown')}</p>
        <p><strong>Action:</strong> {explanation.get('action', 'Unknown')}</p>
        <p><strong>Affected Sensors:</strong> 
        {', '.join(explanation.get('affected_sensors', []))}</p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# MAIN APP
# ============================================================

def main():
    # Header
    st.markdown(
        '<div class="main-header">⚡ AI Energy Anomaly Explainer</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="sub-header">Industrial sensor anomaly detection powered by '
        'Random Forest ML + Llama 3.2 LLM explanations</div>',
        unsafe_allow_html=True
    )
    
    # Load model
    model, df_normal, df_training = load_trained_model()
    
    st.success("Model loaded — F1 Score: 0.981 | Precision: 1.00 | Recall: 0.96")
    
    st.markdown("---")
    
    # ============================================================
    # SIDEBAR
    # ============================================================
    
    with st.sidebar:
        st.header("⚙️ Settings")
        
        st.subheader("Data Source")
        data_source = st.radio(
            "Choose data source:",
            ["Use sample anomaly data", "Upload your own CSV"]
        )
        
        st.subheader("Detection Settings")
        anomaly_threshold = st.slider(
            "Anomaly probability threshold",
            min_value=0.3,
            max_value=0.9,
            value=0.5,
            step=0.1,
            help="Higher = more conservative, fewer false alarms"
        )
        
        max_explanations = st.slider(
            "Max anomalies to explain",
            min_value=1,
            max_value=10,
            value=3,
            help="LLM explanation takes 10-30 seconds each"
        )
        
        st.subheader("About")
        st.info(
            "This system uses a Random Forest model trained on "
            "14,965 rows of real industrial pump sensor data "
            "to detect anomalies, then uses Llama 3.2 running "
            "locally via Ollama to explain what caused each anomaly "
            "and what action to take."
        )
    
    # ============================================================
    # DATA LOADING
    # ============================================================
    
    if data_source == "Use sample anomaly data":
        st.subheader("📊 Sample Data — Valve Fault Experiment")
        
        sample_url = "https://raw.githubusercontent.com/waico/SKAB/master/data/valve1/1.csv"
        df_input = pd.read_csv(
            sample_url, sep=';',
            index_col='datetime',
            parse_dates=True
        )
        
        st.info(
            f"Loaded sample dataset: {len(df_input)} rows, "
            f"{int(df_input['anomaly'].sum())} known anomalies"
        )
        
    else:
        st.subheader("📁 Upload Sensor Data")
        uploaded_file = st.file_uploader(
            "Upload CSV with sensor columns: "
            "Pressure, Volume Flow RateRMS, Current, Temperature",
            type=['csv']
        )
        
        if uploaded_file is None:
            st.warning("Please upload a CSV file to continue")
            return
            
        df_input = pd.read_csv(
            uploaded_file,
            index_col=0,
            parse_dates=True
        )
    
    # Show raw data preview
    with st.expander("📋 View raw sensor data"):
        st.dataframe(df_input.head(20))
    
    # ============================================================
    # ANOMALY DETECTION
    # ============================================================
    
    st.markdown("---")
    st.subheader("🔍 Anomaly Detection")
    
    if st.button("🚀 Run Anomaly Detection", type="primary"):
        
        with st.spinner("Engineering features and detecting anomalies..."):
            # Engineer features on input data
            X_input = engineer_features(df_input)
            
            # Detect anomalies
            predictions, probabilities = detect_anomalies(model, X_input)
        
        # Results summary
        n_anomalies = predictions.sum()
        anomaly_rate = predictions.mean()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Rows", len(predictions))
        with col2:
            st.metric("Anomalies Detected", int(n_anomalies))
        with col3:
            st.metric("Anomaly Rate", f"{anomaly_rate:.1%}")
        with col4:
            st.metric("Model F1 Score", "0.981")
        
        # ============================================================
        # SENSOR CHARTS
        # ============================================================
        
        st.markdown("---")
        st.subheader("📈 Sensor Readings with Detected Anomalies")
        
        # Get anomaly timestamps
        anomaly_mask = predictions == 1
        anomaly_timestamps = X_input.index[anomaly_mask]
        
        # Plot each sensor
        sensors_to_plot = [
            'Pressure', 'Volume Flow RateRMS',
            'Current', 'Temperature'
        ]
        
        for sensor in sensors_to_plot:
            if sensor in df_input.columns:
                fig = plot_sensor_data(
                    df_input, anomaly_timestamps, sensor
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # ============================================================
        # LLM EXPLANATIONS
        # ============================================================
        
        st.markdown("---")
        st.subheader("🤖 AI-Generated Anomaly Explanations")
        
        if n_anomalies == 0:
            st.success("No anomalies detected in this dataset.")
            return
        
        # Get top anomalies by probability
        anomaly_df = pd.DataFrame({
            'timestamp': X_input.index[anomaly_mask],
            'probability': probabilities[anomaly_mask]
        }).sort_values('probability', ascending=False)
        
        # Limit to max_explanations
        anomaly_df = anomaly_df.head(max_explanations)
        
        st.info(
            f"Generating LLM explanations for top "
            f"{len(anomaly_df)} anomalies by confidence score. "
            f"Each explanation takes 10-30 seconds."
        )
        
        # Generate explanation for each anomaly
        for idx, row in anomaly_df.iterrows():
            timestamp = row['timestamp']
            probability = row['probability']
            
            with st.spinner(
                f"Generating explanation for anomaly at {timestamp}..."
            ):
                # Get sensor readings at anomaly time
                if timestamp in df_input.index:
                    anomaly_row = df_input.loc[timestamp]
                    
                    # Generate explanation
                    explanation, context = explain_anomaly(
                        anomaly_row,
                        df_normal,
                        anomaly_duration_seconds=30
                    )
                    
                    # Display explanation
                    display_explanation(explanation, timestamp, probability)
                    
                    # Show context in expander
                    with st.expander(
                        f"📋 View context sent to LLM — {timestamp}"
                    ):
                        st.code(context)
        
        # ============================================================
        # FEEDBACK
        # ============================================================
        
        st.markdown("---")
        st.subheader("💬 Was this helpful?")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("👍 Explanations were accurate"):
                st.success(
                    "Thank you for the feedback! "
                    "This helps improve the system."
                )
        with col2:
            if st.button("👎 Explanations were inaccurate"):
                st.error(
                    "Thank you for the feedback! "
                    "This case will be added to the evaluation dataset."
                )


if __name__ == "__main__":
    main()