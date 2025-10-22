"""
Data Upload and Preprocessing Page
"""

import streamlit as st

# Page config MUST be first
st.set_page_config(page_title="Data Upload", page_icon="📤", layout="wide")

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
sys.path.append('..')

from utils.session_state import init_session_state, save_data
from utils.data_utils import preprocess_data, engineer_features, split_data

# Initialize session state
init_session_state()

st.markdown('<h1 style="color: #1E3A8A;">📤 Data Upload & Preprocessing</h1>', unsafe_allow_html=True)

# Instructions
st.markdown("""
<div style="background-color: #EFF6FF; border-left: 4px solid #3B82F6; padding: 1rem; border-radius: 5px; margin: 1rem 0;">
    <h3 style="margin-top: 0;">📋 Instructions</h3>
    <p>Upload your customer dataset in CSV format. The app will automatically:</p>
    <ul>
        <li>Clean and preprocess the data</li>
        <li>Engineer relevant features</li>
        <li>Split into training and test sets</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# File uploader
col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type="csv",
        help="Upload your telecom churn dataset"
    )

with col2:
    use_sample = st.button("📁 Use Sample Dataset", type="primary")

# Load data
data = None

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        st.success(f"✅ File uploaded successfully! {data.shape[0]} rows and {data.shape[1]} columns")
    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")

elif use_sample:
    try:
        data = pd.read_csv('../data/churndata.csv')
        st.success(f"✅ Sample dataset loaded! {data.shape[0]} rows and {data.shape[1]} columns")
    except Exception as e:
        st.error(f"❌ Error loading sample dataset: {str(e)}")

# Process data if loaded
if data is not None:
    st.markdown("---")
    
    # Show raw data preview
    with st.expander("👁️ View Raw Data (First 10 rows)"):
        st.dataframe(data.head(10), width="stretch")
    
    # Data info
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Rows", f"{data.shape[0]:,}")
    with col2:
        st.metric("Total Columns", data.shape[1])
    with col3:
        missing = data.isnull().sum().sum()
        st.metric("Missing Values", missing)
    with col4:
        if 'Churn' in data.columns:
            churn_rate = (data['Churn'] == 'Yes').sum() / len(data) * 100
            st.metric("Churn Rate", f"{churn_rate:.1f}%")
    
    st.markdown("---")
    
    # Preprocessing button
    if st.button("🔄 Preprocess Data", type="primary"):
        with st.spinner("Processing data..."):
            # Save raw data
            st.session_state.data_raw = data.copy()
            
            # Preprocess
            processed = preprocess_data(data)
            processed = engineer_features(processed)
            
            # Split data
            X_train, X_test, y_train, y_test = split_data(processed)
            
            # Save to session state
            save_data(processed, 'data_processed')
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            st.session_state.data_uploaded = True
            
            st.success("✅ Data preprocessed successfully!")
            st.balloons()
    
    # Show processed data if available
    if st.session_state.data_processed is not None:
        st.markdown("---")
        st.markdown("### ✨ Processed Data")
        
        processed = st.session_state.data_processed
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Train Samples", f"{len(st.session_state.X_train):,}")
        with col2:
            st.metric("Test Samples", f"{len(st.session_state.X_test):,}")
        with col3:
            st.metric("Features", st.session_state.X_train.shape[1])
        with col4:
            train_churn_rate = st.session_state.y_train.mean() * 100
            st.metric("Train Churn Rate", f"{train_churn_rate:.1f}%")
        
        # Data preview tabs
        tab1, tab2, tab3 = st.tabs(["📊 Processed Data", "📈 Visualizations", "📋 Feature Info"])
        
        with tab1:
            st.dataframe(processed.head(20), width="stretch")
            
            # Download button
            csv = processed.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Processed Data",
                data=csv,
                file_name="processed_churn_data.csv",
                mime="text/csv"
            )
        
        with tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                # Churn distribution
                if 'Churn' in processed.columns:
                    churn_counts = processed['Churn'].value_counts()
                    fig = px.pie(
                        values=churn_counts.values,
                        names=['No Churn', 'Churn'],
                        title='Churn Distribution',
                        color_discrete_sequence=['#10B981', '#EF4444']
                    )
                    st.plotly_chart(fig, width="stretch")
            
            with col2:
                # Tenure distribution
                fig = px.histogram(
                    processed,
                    x='tenure',
                    nbins=30,
                    title='Tenure Distribution',
                    color_discrete_sequence=['#3B82F6']
                )
                st.plotly_chart(fig, width="stretch")
            
            # Monthly charges by tenure group
            if 'tenure_group' in processed.columns:
                fig = px.box(
                    processed,
                    x='tenure_group',
                    y='MonthlyCharges',
                    title='Monthly Charges by Tenure Group',
                    color='tenure_group'
                )
                st.plotly_chart(fig, width="stretch")
        
        with tab3:
            st.markdown("### 📊 Feature Information")
            
            st.markdown("**Engineered Features:**")
            features_info = pd.DataFrame({
                'Feature': ['tenure_group', 'LongTerm', 'AvailingInternetService', 'NumServices', 'AvgCharge', 'LessThan6Months'],
                'Description': [
                    'Categorical tenure grouping (0-12, 13-24, 25-48, 49-60, 61-72 months)',
                    'Binary flag for customers with tenure > 24 months',
                    'Binary flag for customers with internet service',
                    'Total count of services subscribed',
                    'Average charge per service (MonthlyCharges / NumServices)',
                    'Binary flag for customers in first 6 months'
                ],
                'Type': ['Categorical', 'Binary', 'Binary', 'Numeric', 'Numeric', 'Binary']
            })
            st.table(features_info)

else:
    # Show placeholder
    st.info("👆 Please upload a dataset or use the sample dataset to get started.")

# Navigation hint
if st.session_state.data_uploaded:
    st.markdown("---")
    st.markdown("""
    <div style="background-color: #D1FAE5; border-left: 4px solid #10B981; padding: 1rem; border-radius: 5px;">
        <h3 style="margin-top: 0;">✅ Data Ready!</h3>
        <p>Your data is preprocessed and ready for model training. Head to the <strong>Model Training</strong> page to start building models!</p>
    </div>
    """, unsafe_allow_html=True)
