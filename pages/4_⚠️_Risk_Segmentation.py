"""
Risk Segmentation and Strategy Page
"""

import streamlit as st

st.set_page_config(page_title="Risk Segmentation", page_icon="⚠️", layout="wide")

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
sys.path.append('..')

from utils.session_state import init_session_state
from utils.ui import apply_theme
from analysis.risk_segmentation import (
    segment_by_probability,
    summarize_segments,
    recommended_interventions,
)

init_session_state()
apply_theme()

st.markdown('<h1 style="color: #1E3A8A;">⚠️ Customer Risk Segmentation</h1>', unsafe_allow_html=True)

# Check prerequisites
if not st.session_state.get('models_trained', False) or st.session_state.get('best_model') is None:
    st.warning("⚠️ Please train models first!")
    st.stop()

# Generate risk segments (percentile-based)
if st.session_state.risk_segments is None:
    if st.button("🚀 Generate Risk Segments", type="primary"):
        with st.spinner("Segmenting customers..."):
            # Combine train and test
            X_full = pd.concat([st.session_state.X_train, st.session_state.X_test], ignore_index=True)
            y_full = pd.concat([st.session_state.y_train, st.session_state.y_test], ignore_index=True)

            # Predict churn probabilities
            y_pred_proba = st.session_state.best_model.predict_proba(X_full)[:, 1]

            # Create percentile-based segments: 90/70/40 percentiles
            segments = segment_by_probability(y_pred_proba)
            X_full['churn_probability'] = y_pred_proba
            X_full['actual_churn'] = y_full.values
            X_full['risk_segment'] = segments

            # Persist
            st.session_state.risk_segments = X_full
            st.session_state.very_high_risk_customers = X_full[X_full['risk_segment'] == 'Very High Risk']
            st.session_state.high_risk_customers = X_full[X_full['risk_segment'] == 'High Risk']
            st.success("✅ Risk segments created!")
            st.rerun()

# Display segments
if st.session_state.risk_segments is not None:
    X_full = st.session_state.risk_segments
    
    # Summary metrics
    st.markdown("### 📊 Risk Segment Summary")

    seg_order = ['Very High Risk', 'High Risk', 'Medium Risk', 'Low Risk']
    counts = X_full['risk_segment'].value_counts()
    total = len(X_full)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        vhr = counts.get('Very High Risk', 0)
        st.metric("Very High Risk", f"{vhr:,}", f"{(vhr/total*100):.1f}%")
    with c2:
        hr = counts.get('High Risk', 0)
        st.metric("High Risk", f"{hr:,}", f"{(hr/total*100):.1f}%")
    with c3:
        mr = counts.get('Medium Risk', 0)
        st.metric("Medium Risk", f"{mr:,}", f"{(mr/total*100):.1f}%")
    with c4:
        lr = counts.get('Low Risk', 0)
        st.metric("Low Risk", f"{lr:,}", f"{(lr/total*100):.1f}%")
    
    st.markdown("---")
    
    # Visualizations
    tab1, tab2, tab3 = st.tabs(["📊 Distribution", "📈 Churn Rates", "💼 Strategy"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart
            fig = px.pie(
                X_full,
                names='risk_segment',
                title='Customer Distribution by Risk',
                color='risk_segment',
                category_orders={'risk_segment': seg_order},
                color_discrete_map={
                    'Very High Risk': '#991B1B',
                    'High Risk': '#EF4444',
                    'Medium Risk': '#F59E0B',
                    'Low Risk': '#10B981'
                }
            )
            st.plotly_chart(fig, width="stretch")
        
        with col2:
            # Churn rate by segment
            churn_by_segment = (X_full.groupby('risk_segment')['actual_churn'].mean() * 100).reindex(seg_order)
            fig = px.bar(
                x=churn_by_segment.index,
                y=churn_by_segment.values,
                title='Actual Churn Rate by Segment',
                labels={'x': 'Risk Segment', 'y': 'Churn Rate (%)'},
                color=churn_by_segment.index,
                category_orders={'x': seg_order},
                color_discrete_map={
                    'Very High Risk': '#991B1B',
                    'High Risk': '#EF4444',
                    'Medium Risk': '#F59E0B',
                    'Low Risk': '#10B981'
                }
            )
            st.plotly_chart(fig, width="stretch")
    
    with tab2:
        # Detailed metrics table using analyzer helper
        summary = summarize_segments(X_full, prob_col='churn_probability', actual_col='actual_churn')
        summary = summary.set_index('risk_segment').loc[[s for s in seg_order if s in summary.index]]
        st.dataframe(summary, width="stretch")
    
    with tab3:
        st.markdown("### 💼 Retention Strategies")
        
        # Dynamic retention strategies per segment
        seg_counts = X_full['risk_segment'].value_counts()
        for seg, color, bg in [
            ('Very High Risk', '#991B1B', '#FEE2E2'),
            ('High Risk', '#EF4444', '#FECACA'),
            ('Medium Risk', '#92400E', '#FEF3C7'),
            ('Low Risk', '#065F46', '#D1FAE5'),
        ]:
            if seg in seg_counts:
                strat = recommended_interventions(seg)
                st.markdown(f"""
                <div style="background-color: {bg}; border-left: 4px solid {color}; padding: 1rem; border-radius: 5px; margin-top: 1rem;">
                    <h3 style="color: #1F2937;">{seg} ({seg_counts[seg]:,} customers)</h3>
                    <ul style="color: #1F2937;">
                        <li>✓ Priority: <strong>{strat.get('priority','')}</strong></li>
                        <li>✓ Offer: <strong>{strat.get('discount','')}</strong></li>
                        <li>✓ Contact: <strong>{strat.get('contact','')}</strong></li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
    
    # Detailed High-Risk Analysis
    st.markdown("---")
    st.markdown("## 🎯 Detailed High-Risk Analysis")

    # Prefer 'Very High Risk' if available, else fall back to 'High Risk'
    target_segment = 'Very High Risk' if (X_full['risk_segment'] == 'Very High Risk').any() else 'High Risk'
    high_risk_df = X_full[X_full['risk_segment'] == target_segment]
    
    # High-risk metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_churn_prob = high_risk_df['churn_probability'].mean()
        st.metric("Avg Churn Probability", f"{avg_churn_prob:.3f}")
    
    with col2:
        actual_churn_rate = high_risk_df['actual_churn'].mean()
        churned_count = high_risk_df['actual_churn'].sum()
        st.metric("Actual Churn Rate", f"{actual_churn_rate:.1%}", f"{int(churned_count)} customers")
    
    with col3:
        avg_tenure = high_risk_df['tenure'].mean()
        st.metric("Avg Tenure", f"{avg_tenure:.1f} months")
    
    with col4:
        pct_less_12m = (high_risk_df['tenure'] <= 12).sum() / len(high_risk_df) * 100
        st.metric("< 12 Months", f"{pct_less_12m:.1f}%")
    
    # Visualizations
    st.markdown("### 📊 Risk Distribution & Patterns")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Tenure distribution by risk
        fig = go.Figure()
        for segment, color in [
            ('Low Risk', '#10B981'),
            ('Medium Risk', '#F59E0B'),
            ('High Risk', '#EF4444'),
            ('Very High Risk', '#991B1B'),
        ]:
            segment_data = X_full[X_full['risk_segment'] == segment]['tenure']
            fig.add_trace(go.Box(
                y=segment_data,
                name=segment,
                marker_color=color,
                boxmean='sd'
            ))
        fig.update_layout(
            title='Customer Tenure Distribution by Risk Segment',
            yaxis_title='Tenure (months)',
            showlegend=True,
            height=400
        )
        st.plotly_chart(fig, width="stretch")
    
    with col2:
        # Monthly charges by risk
        fig = go.Figure()
        for segment, color in [
            ('Low Risk', '#10B981'),
            ('Medium Risk', '#F59E0B'),
            ('High Risk', '#EF4444'),
            ('Very High Risk', '#991B1B'),
        ]:
            segment_data = X_full[X_full['risk_segment'] == segment]['MonthlyCharges']
            fig.add_trace(go.Violin(
                y=segment_data,
                name=segment,
                marker_color=color,
                box_visible=True,
                meanline_visible=True
            ))
        fig.update_layout(
            title='Monthly Charges Distribution by Risk Segment',
            yaxis_title='Monthly Charges ($)',
            showlegend=True,
            height=400
        )
        st.plotly_chart(fig, width="stretch")
    
    # High-risk characteristics
    st.markdown("### 🔍 Top 3 High-Risk Characteristics")
    
    char_col1, char_col2, char_col3 = st.columns(3)
    
    with char_col1:
        new_customers = (high_risk_df['tenure'] <= 12).sum()
        new_pct = (new_customers / len(high_risk_df)) * 100
        st.markdown(f"""
        <div style="background-color: #FEE2E2; padding: 1rem; border-radius: 10px; border: 2px solid #EF4444;">
            <h4 style="color: #991B1B; margin-top: 0;">🕐 New Customers</h4>
            <p style="color: #1F2937; font-size: 24px; font-weight: bold; margin: 0.5rem 0;">{new_pct:.1f}%</p>
            <p style="color: #1F2937; margin: 0;">Tenure < 12 months<br>({new_customers:,} customers)</p>
            <p style="color: #DC2626; font-weight: bold; margin-top: 0.5rem;">CRITICAL RISK</p>
        </div>
        """, unsafe_allow_html=True)
    
    with char_col2:
        high_charges = (high_risk_df['MonthlyCharges'] > 75).sum()
        high_charges_pct = (high_charges / len(high_risk_df)) * 100
        st.markdown(f"""
        <div style="background-color: #FEF3C7; padding: 1rem; border-radius: 10px; border: 2px solid #F59E0B;">
            <h4 style="color: #92400E; margin-top: 0;">💰 High Monthly Charges</h4>
            <p style="color: #1F2937; font-size: 24px; font-weight: bold; margin: 0.5rem 0;">{high_charges_pct:.1f}%</p>
            <p style="color: #1F2937; margin: 0;">> $75/month<br>({high_charges:,} customers)</p>
            <p style="color: #D97706; font-weight: bold; margin-top: 0.5rem;">HIGH RISK</p>
        </div>
        """, unsafe_allow_html=True)
    
    with char_col3:
        low_services = (high_risk_df['NumServices'] <= 2).sum()
        low_services_pct = (low_services / len(high_risk_df)) * 100
        st.markdown(f"""
        <div style="background-color: #FEF3C7; padding: 1rem; border-radius: 10px; border: 2px solid #F59E0B;">
            <h4 style="color: #92400E; margin-top: 0;">📦 Low Service Adoption</h4>
            <p style="color: #1F2937; font-size: 24px; font-weight: bold; margin: 0.5rem 0;">{low_services_pct:.1f}%</p>
            <p style="color: #1F2937; margin: 0;">≤ 2 services<br>({low_services:,} customers)</p>
            <p style="color: #D97706; font-weight: bold; margin-top: 0.5rem;">HIGH RISK</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Summary table
    st.markdown("### 📋 Risk Segment Summary Table")
    
    risk_summary_data = []
    for segment in ['Very High Risk', 'High Risk', 'Medium Risk', 'Low Risk']:
        seg_df = X_full[X_full['risk_segment'] == segment]
        risk_summary_data.append({
            'Risk Segment': segment,
            'Customer Count': f"{len(seg_df):,}",
            '% of Total': f"{(len(seg_df)/len(X_full)*100):.1f}%",
            'Avg Risk Score': f"{seg_df['churn_probability'].mean():.3f}",
            'Actual Churn Rate': f"{seg_df['actual_churn'].mean():.1%}",
            'Actual Churned': f"{int(seg_df['actual_churn'].sum()):,}"
        })
    
    summary_df = pd.DataFrame(risk_summary_data)
    
    # Style the dataframe
    def highlight_risk(row):
        if row['Risk Segment'] == 'High Risk':
            return ['background-color: #FEE2E2'] * len(row)
        elif row['Risk Segment'] == 'Medium Risk':
            return ['background-color: #FEF3C7'] * len(row)
        else:
            return ['background-color: #D1FAE5'] * len(row)
    
    styled_summary = summary_df.style.apply(highlight_risk, axis=1)
    st.dataframe(styled_summary, width="stretch", hide_index=True)
    
    # Key statistics
    total_customers = len(X_full)
    high_risk_count = len(high_risk_df)
    actual_churn_count = X_full['actual_churn'].sum()
    high_risk_actual_churn = X_full[(X_full['risk_segment'] == target_segment) & (X_full['actual_churn'] == 1)].shape[0]
    
    st.info(f"""
    📊 **Key Statistics:**  
    Total Customers: **{total_customers:,}** | 
    High Risk: **{high_risk_count:,}** ({high_risk_count/total_customers*100:.1f}%) | 
    Actual Churned: **{actual_churn_count:,}** ({actual_churn_count/total_customers*100:.1f}%) | 
    High Risk Churned: **{high_risk_actual_churn:,}**
    """)
    
    # Export
    st.markdown("---")
    st.markdown("### 📥 Export Customer Lists")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Very High Risk list (if any), else High Risk
        vhr_df = X_full[X_full['risk_segment'] == 'Very High Risk']
        if len(vhr_df) > 0:
            csv = vhr_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Very High Risk List", csv, "very_high_risk_customers.csv", "text/csv")
        else:
            csv = high_risk_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 High Risk List", csv, "high_risk_customers.csv", "text/csv")
    
    with col2:
        medium_df = X_full[X_full['risk_segment'] == 'Medium Risk']
        csv = medium_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Medium Risk List", csv, "medium_risk_customers.csv", "text/csv")
    
    with col3:
        low_df = X_full[X_full['risk_segment'] == 'Low Risk']
        csv = low_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Low Risk List", csv, "low_risk_customers.csv", "text/csv")

    with col4:
        hr_df = X_full[X_full['risk_segment'] == 'High Risk']
        csv = hr_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 High Risk List", csv, "high_risk_customers.csv", "text/csv")
