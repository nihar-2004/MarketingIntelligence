import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import timedelta
import streamlit_antd_components as sac

# --- Page Configuration ---
st.set_page_config(
    page_title="Marketing Intelligence Dashboard",
    page_icon="📊",
    layout="wide"
)

st.markdown(f'''
    <style>
    .stApp .main .block-container{{
        padding:30px 50px
    }}
    .stApp [data-testid='stSidebar']>div:nth-child(1)>div:nth-child(2){{
        padding-top:50px
    }}
    iframe{{
        display:block;
    }}
    .stRadio div[role='radiogroup']>label{{
        margin-right:5px
    }}
    </style>
    ''', unsafe_allow_html=True)

# --- Data Loading ---
@st.cache_data
def load_data():
    """Loads and prepares the data."""
    # This assumes 'final_marketing_data.csv' is the result of your data_processor script
    df = pd.read_csv('final_marketing_data.csv')
    return df

df = load_data()

# --- Helper Function for Calculating KPIs ---
def calculate_kpis(data):
    """Calculates key metrics from a given dataframe."""
    if data.empty:
        return {'revenue': 0, 'spend': 0, 'new_customers': 0, 'roas': 0, 'cac': 0, 'ctr': 0}
    
    revenue = data.groupby('date')['total_revenue'].first().sum()
    spend = data['spend'].sum()
    new_customers = data.groupby('date')['new_customers'].first().sum()
    
    roas = (data['attributed revenue'].sum() / spend) if spend > 0 else 0
    cac = (spend / new_customers) if new_customers > 0 else 0
    
    total_clicks = data['clicks'].sum()
    total_impressions = data['impression'].sum()
    ctr = (total_clicks / total_impressions) * 100 if total_impressions > 0 else 0
    
    return {'revenue': revenue, 'spend': spend, 'new_customers': new_customers, 'roas': roas, 'cac': cac, 'ctr': ctr}


# --- Main App ---
if df is None:
    st.error("Data could not be loaded. Please check 'final_marketing_data.csv'.")
else:
    # Convert 'date' column to datetime objects
    df['date'] = pd.to_datetime(df['date'])

    st.title("📊 Marketing Intelligence Dashboard")
    st.markdown("Analyze marketing spend and its connection to business outcomes.")

    c = st.columns(4)
    min_date = df['date'].min().date()
    max_date = df['date'].max().date()
    default_start_date = max_date - timedelta(days=29)
    if default_start_date < min_date:
        default_start_date = min_date
    
    with c[0].expander('Date Range', True):
        start_date, end_date = st.slider(
            "Select Date Range",
            min_value=min_date,
            max_value=max_date,
            value=(default_start_date, max_date),
            format="YYYY-MM-DD",
            label_visibility="collapsed"
        )

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    with c[1].expander('Platform(s)', True):
        options = df['platform'].unique().tolist()
        selected_platforms = sac.checkbox(items=options, label='Select Platform(s)', index=list(range(len(options))), align='center', check_all='Select all')

    with c[2].expander('Tactic(s)', True):
        options = df['tactic'].unique().tolist()
        selected_tactics = sac.checkbox(items=options, label="Select Tactic(s)", index=list(range(len(options))), align='center', check_all='Select all')
        
    with c[3].expander('State(s)', True):
        options = df['state'].unique().tolist()
        selected_states = sac.checkbox(items=options, label="Select State(s)", index=list(range(len(options))), align='center', check_all='Select all')
        
    st.divider()
    
    # --- Filter for Current and Previous Periods ---
    mask_current = (
        (df['date'] >= start_date) & (df['date'] <= end_date) &
        (df['platform'].isin(selected_platforms)) &
        (df['tactic'].isin(selected_tactics)) &
        (df['state'].isin(selected_states))
    )
    df_filtered = df[mask_current]

    period_duration = (end_date - start_date).days
    prev_start_date = start_date - timedelta(days=period_duration + 1)
    prev_end_date = start_date - timedelta(days=1)
    mask_previous = (
        (df['date'] >= prev_start_date) & (df['date'] <= prev_end_date) &
        (df['platform'].isin(selected_platforms)) &
        (df['tactic'].isin(selected_tactics)) &
        (df['state'].isin(selected_states))
    )
    df_previous = df[mask_previous]

    if df_filtered.empty:
        st.warning("No data available for the selected filters. Please adjust your selections.")
    else:
        # --- 1. Executive Summary (KPIs with Deltas) ---
        st.header("Executive Summary")
        st.markdown(f"Comparing **{start_date.strftime('%d %b, %Y')} - {end_date.strftime('%d %b, %Y')}** with preceding period.")

        kpis_current = calculate_kpis(df_filtered)
        kpis_previous = calculate_kpis(df_previous)

        def get_delta(current, previous):
            if previous == 0: return None
            return ((current - previous) / previous) * 100

        delta_revenue = get_delta(kpis_current['revenue'], kpis_previous['revenue'])
        delta_spend = get_delta(kpis_current['spend'], kpis_previous['spend'])
        delta_roas = get_delta(kpis_current['roas'], kpis_previous['roas'])
        delta_new_customers = get_delta(kpis_current['new_customers'], kpis_previous['new_customers'])
        delta_cac = get_delta(kpis_current['cac'], kpis_previous['cac'])
        delta_ctr = get_delta(kpis_current['ctr'], kpis_previous['ctr'])

        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Total Revenue", f"${kpis_current['revenue']:,.2f}", f"{delta_revenue:.2f}%" if delta_revenue is not None else "N/A")
        col2.metric("Total Ad Spend", f"${kpis_current['spend']:,.2f}", f"{delta_spend:.2f}%" if delta_spend is not None else "N/A", delta_color="inverse")
        col3.metric("Overall ROAS", f"{kpis_current['roas']:.2f}x", f"{delta_roas:.2f}%" if delta_roas is not None else "N/A")
        col4.metric("New Customers", f"{int(kpis_current['new_customers']):,}", f"{delta_new_customers:.2f}%" if delta_new_customers is not None else "N/A")
        col5.metric("Average CAC", f"${kpis_current['cac']:,.2f}", f"{delta_cac:.2f}%" if delta_cac is not None else "N/A", delta_color="inverse")
        col6.metric("Average CTR", f"{kpis_current['ctr']:.2f}%", f"{delta_ctr:.2f}%" if delta_ctr is not None else "N/A")

        st.markdown("---")


        # --- 2. Performance Trend Analysis ---
        st.header("Performance Trend Analysis")

        daily_agg = df_filtered.groupby('date').agg({
            'total_revenue': 'first',
            'spend': 'sum',
            'clicks': 'sum',
            'impression': 'sum'
        }).reset_index()
        daily_agg['ctr'] = (daily_agg['clicks'] / daily_agg['impression']) * 100

        trend_col1, trend_col2 = st.columns(2)
        with trend_col1:
            fig_rev_spend = go.Figure()
            fig_rev_spend.add_trace(go.Scatter(x=daily_agg['date'], y=daily_agg['total_revenue'], name='Total Revenue', mode='lines', line=dict(color='#1f77b4')))
            fig_rev_spend.add_trace(go.Scatter(x=daily_agg['date'], y=daily_agg['spend'], name='Ad Spend', mode='lines', line=dict(color='#ff7f0e')))
            fig_rev_spend.update_layout(
            title_text="<b>Daily Revenue vs. Ad Spend</b>",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            yaxis_title="<b>Amount ($)</b>"
            )
            st.plotly_chart(fig_rev_spend, use_container_width=True)
        
        with trend_col2:
            fig_ctr = px.line(
            daily_agg,
            x='date',
            y='ctr',
            title='<b>Daily Click-Through Rate (CTR)</b>'
            )
            fig_ctr.update_traces(line_color='#2ca02c')
            fig_ctr.update_layout(yaxis_title="<b>CTR (%)</b>")
            st.plotly_chart(fig_ctr, use_container_width=True)

        # --- 3. Marketing Performance Deep Dive ---
        st.header("Marketing Performance Deep Dive")

        selected_metric = st.radio(
            "Select Metric for Breakdown Analysis:",
            options=['ROAS', 'CTR'],
            horizontal=True,
        )
        
        col_plat, col_tac = st.columns(2)

        if selected_metric == 'ROAS':
            y_value = 'roas'
            y_title = 'Return on Ad Spend (ROAS)'
            text_format = '.2f'
        else: # CTR
            y_value = 'ctr'
            y_title = 'Click-Through Rate (CTR %)'
            text_format = '.2f'

        with col_plat:
            platform_agg = df_filtered.groupby('platform').agg({
                'spend': 'sum',
                'attributed revenue': 'sum',
                'clicks': 'sum',
                'impression': 'sum'
            }).reset_index()
            platform_agg['roas'] = (platform_agg['attributed revenue'] / platform_agg['spend']).fillna(0)
            platform_agg['ctr'] = (platform_agg['clicks'] / platform_agg['impression'] * 100).fillna(0)
            
            fig_platform = px.bar(
                platform_agg.sort_values(y_value, ascending=False), 
                x='platform', y=y_value,
                title=f"<b>{selected_metric} by Platform</b>",
                labels={'platform': 'Platform', y_value: y_title},
                color='platform',
                text_auto=text_format
            )
            st.plotly_chart(fig_platform, use_container_width=True)

        with col_tac:
            tactic_agg = df_filtered.groupby('tactic').agg({
                'spend': 'sum',
                'attributed revenue': 'sum',
                'clicks': 'sum',
                'impression': 'sum'
            }).reset_index()
            tactic_agg['roas'] = (tactic_agg['attributed revenue'] / tactic_agg['spend']).fillna(0)
            tactic_agg['ctr'] = (tactic_agg['clicks'] / tactic_agg['impression'] * 100).fillna(0)

            fig_tactic = px.bar(
                tactic_agg.sort_values(y_value, ascending=False), 
                x='tactic', y=y_value,
                title=f"<b>{selected_metric} by Tactic</b>",
                labels={'tactic': 'Tactic', y_value: y_title},
                color='tactic',
                text_auto=text_format
            )
            st.plotly_chart(fig_tactic, use_container_width=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        groupbystate = st.radio("Group Campaign Scatter by:", options=['Platform', 'State'], horizontal=True)
        
        criteria = 'platform' if groupbystate == 'Platform' else 'state'

        campaign_agg = df_filtered.groupby(['campaign', criteria]).agg({
            'spend': 'sum',
            'attributed revenue': 'sum'
        }).reset_index()
        campaign_agg['roas'] = (campaign_agg['attributed revenue'] / campaign_agg['spend']).fillna(0)
        
        # Exaggerate the size for better visual distinction. A power of 1.5 provides good emphasis.
        campaign_agg['size_exaggerated'] = campaign_agg['attributed revenue'] ** 1.5
        
        fig_campaign_scatter = px.scatter(
            campaign_agg,
            x='spend',
            y='roas',
            size='size_exaggerated', # Use the new exaggerated size column
            color=criteria,
            hover_name='campaign',
            title='<b>Campaign Efficiency (Spend vs. ROAS)</b>',
            labels={'spend': 'Total Spend ($)', 'roas': 'ROAS', 'attributed revenue': 'Attributed Revenue'},
            size_max=60, # Keep max size reasonable
            log_x=True
        )
        st.plotly_chart(fig_campaign_scatter, use_container_width=True)
        
        st.markdown("---")

        # --- 4. Raw Data View ---
        st.header("Explore Raw Data")
        if st.checkbox("Show Filtered Raw Data"):
            st.dataframe(df_filtered)

