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
    """Loads and prepares the data using the imported function."""
    business_csv = 'business.csv'
    facebook_csv = 'Facebook.csv'
    google_csv = 'Google.csv'
    tiktok_csv = 'TikTok.csv'
    df = pd.read_csv('final_marketing_data.csv')
    return df

df = load_data()

def calculate_kpis(data):
    """Calculates key metrics from a given dataframe."""
    if data.empty:
        return {'revenue': 0, 'spend': 0, 'new_customers': 0, 'roas': 0, 'cac': 0}
    
    revenue = data.groupby('date')['total_revenue'].first().sum()
    spend = data['spend'].sum()
    new_customers = data.groupby('date')['new_customers'].first().sum()
    
    roas = (data['attributed revenue'].sum() / spend) if spend > 0 else 0
    cac = (spend / new_customers) if new_customers > 0 else 0
    
    return {'revenue': revenue, 'spend': spend, 'new_customers': new_customers, 'roas': roas, 'cac': cac}


# --- Main App ---
if df is None:
    st.error("Data could not be loaded. Please check the CSV file paths and content.")
else:
    # --- FIX: Explicitly convert 'date' column to datetime objects ---
    # This ensures all subsequent date operations will work correctly.
    df['date'] = pd.to_datetime(df['date'])

    st.title("📊 Marketing Intelligence Dashboard")
    st.markdown("Analyze marketing spend and its connection to business outcomes.")

    c = st.columns(4)
    # Date Range Filter
    min_date = df['date'].min().date()
    max_date = df['date'].max().date()

    # Set default start date to be 30 days before the max_date
    default_start_date = max_date - timedelta(days=30)
    if default_start_date < min_date:
        default_start_date = min_date
    with c[0].expander('Date Range', True):
        start_date, end_date = st.slider(
            "Select Date Range",
            min_value=min_date,
            max_value=max_date,
            value=(default_start_date, max_date),
            format="YYYY-MM-DD"
        )


    # Convert slider dates to datetime for filtering
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    

    with c[1].expander('Platform(s)', True):
        selected_platforms = sac.checkbox(
            items=df['platform'].unique().tolist(),
            label='Select Platform(s)', index=[0, 1, 2], align='center', check_all='Select all',
        )

    with c[2].expander('Tactic(s)', True):
        selected_tactics = sac.checkbox(
            items=df['tactic'].unique().tolist(),
            label="Select Tactic(s)", index=[0, 1, 2, 3, 4, 5], align='center', check_all='Select all'
        )
    with c[3].expander('State(s)', True):
        selected_states = sac.checkbox(
            items=df['state'].unique().tolist(),
            label="Select State(s)", index=[0, 1], align='center', check_all='Select all'
        )
    st.divider()
    # --- Filter DataFrame based on selections ---
    mask = (
        (df['date'] >= start_date) & (df['date'] <= end_date) &
        (df['platform'].isin(selected_platforms)) &
        (df['tactic'].isin(selected_tactics)) &
        (df['state'].isin(selected_states))
    )
    df_filtered = df[mask]
    mask_current = (
        (df['date'] >= start_date) & (df['date'] <= end_date) &
        (df['platform'].isin(selected_platforms)) &
        (df['tactic'].isin(selected_tactics)) &
        (df['state'].isin(selected_states))
    )
    df_filtered = df[mask_current]

    # Previous Period Calculation
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
        # --- 1. Executive Summary (KPIs) ---
        st.header("Executive Summary")
        st.markdown(f"Comparing **{start_date.strftime('%d %b, %Y')} - {end_date.strftime('%d %b, %Y')}** with preceding period.")

        # Calculate KPIs for both periods
        kpis_current = calculate_kpis(df_filtered)
        kpis_previous = calculate_kpis(df_previous)

        # Calculate deltas safely
        def get_delta(current, previous):
            if previous == 0: return None # Avoid division by zero
            return ((current - previous) / previous) * 100

        delta_revenue = get_delta(kpis_current['revenue'], kpis_previous['revenue'])
        delta_spend = get_delta(kpis_current['spend'], kpis_previous['spend'])
        delta_roas = get_delta(kpis_current['roas'], kpis_previous['roas'])
        delta_new_customers = get_delta(kpis_current['new_customers'], kpis_previous['new_customers'])
        delta_cac = get_delta(kpis_current['cac'], kpis_previous['cac'])

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Revenue", f"${kpis_current['revenue']:,.2f}", f"{delta_revenue:.2f}%" if delta_revenue is not None else "N/A")
        col2.metric("Total Ad Spend", f"${kpis_current['spend']:,.2f}", f"{delta_spend:.2f}%" if delta_spend is not None else "N/A", delta_color="inverse")
        col3.metric("Overall ROAS", f"{kpis_current['roas']:.2f}x", f"{delta_roas:.2f}%" if delta_roas is not None else "N/A")
        col4.metric("New Customers", f"{int(kpis_current['new_customers']):,}", f"{delta_new_customers:.2f}%" if delta_new_customers is not None else "N/A")
        col5.metric("Average CAC", f"${kpis_current['cac']:,.2f}", f"{delta_cac:.2f}%" if delta_cac is not None else "N/A", delta_color="inverse")

        st.markdown("---")


        # --- 2. Performance Trend Analysis ---
        st.header("Performance Trend Analysis")

        # Aggregate data daily for trend charts
        daily_agg = df_filtered.groupby('date').agg({
            'total_revenue': 'first',
            'spend': 'sum',
            'roas': 'mean',
            'cac': 'first'
        }).reset_index()

        # Chart 1: Revenue vs. Spend (Dual-Axis)
        fig_rev_spend = make_subplots(specs=[[{"secondary_y": True}]])
        fig_rev_spend.add_trace(go.Scatter(x=daily_agg['date'], y=daily_agg['total_revenue'], name='Total Revenue', mode='lines', line=dict(color='#1f77b4')), secondary_y=False)
        fig_rev_spend.add_trace(go.Scatter(x=daily_agg['date'], y=daily_agg['spend'], name='Ad Spend', mode='lines', line=dict(color='#ff7f0e')), secondary_y=True)
        fig_rev_spend.update_layout(title_text="<b>Daily Revenue vs. Ad Spend</b>", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        fig_rev_spend.update_yaxes(title_text="<b>Total Revenue ($)</b>", secondary_y=False)
        fig_rev_spend.update_yaxes(title_text="<b>Ad Spend ($)</b>", secondary_y=True)
        st.plotly_chart(fig_rev_spend, use_container_width=True)
        

        # --- 3. Marketing Performance Deep Dive ---
        st.header("Marketing Performance Deep Dive")
        
        col_plat, col_tac = st.columns(2)

        with col_plat:
            # Chart 3: Performance by Platform
            platform_agg = df_filtered.groupby('platform').agg({
                'spend': 'sum',
                'attributed revenue': 'sum'
            }).reset_index()
            platform_agg['roas'] = (platform_agg['attributed revenue'] / platform_agg['spend']).fillna(0)
            
            fig_platform = px.bar(
                platform_agg.sort_values('roas', ascending=False), 
                x='platform', y='roas',
                title="<b>ROAS by Platform</b>",
                labels={'roas': 'Return on Ad Spend (ROAS)', 'platform': 'Platform'},
                color='platform',
                text_auto='.2f'
            )
            st.plotly_chart(fig_platform, use_container_width=True)

        with col_tac:
            # Chart 4: Performance by Tactic
            tactic_agg = df_filtered.groupby('tactic').agg({
                'spend': 'sum',
                'attributed revenue': 'sum'
            }).reset_index()
            tactic_agg['roas'] = (tactic_agg['attributed revenue'] / tactic_agg['spend']).fillna(0)

            fig_tactic = px.bar(
                tactic_agg.sort_values('roas', ascending=False), 
                x='tactic', y='roas',
                title="<b>ROAS by Tactic</b>",
                labels={'roas': 'Return on Ad Spend (ROAS)', 'tactic': 'Tactic'},
                color='tactic',
                text_auto='.2f'
            )
            st.plotly_chart(fig_tactic, use_container_width=True)

        # Chart 5: Spend vs. ROAS by Campaign (Scatter Plot)
        groupbystate = st.pills("Group by",options=['State','Platform'],default='State',selection_mode='single')
        
        if groupbystate=='State':
            criteria = 'state'
        else:
            criteria = 'platform'
        campaign_agg = df_filtered.groupby(['campaign', criteria]).agg({
            'spend': 'sum',
            'attributed revenue': 'sum'
        }).reset_index()
        campaign_agg['roas'] = (campaign_agg['attributed revenue'] / campaign_agg['spend']).fillna(0)
        campaign_agg['area'] = campaign_agg['attributed revenue']**1.6
        fig_campaign_scatter = px.scatter(
            campaign_agg,
            x='spend',
            y='roas',
            size='area',
            color=criteria,
            hover_name='campaign',
            title='<b>Campaign Efficiency (Spend vs. ROAS)</b>',
            labels={
            'spend': 'Total Spend ($)',
            'roas': 'ROAS',
            'attributed revenue': 'Attributed Revenue',
            'platform': 'Platform'
            # 'area' is intentionally omitted so it won't show in the legend or tooltips
            },
            size_max=40,
            log_x=True
        )
        # Remove 'area' from hover data
        fig_campaign_scatter.update_traces(hovertemplate=None, hoverinfo='skip')
        # Or, more precisely, set custom hover data excluding 'area'
        fig_campaign_scatter.for_each_trace(
            lambda t: t.update(
            hovertemplate='<br>'.join([
                'Campaign=%{hovertext}',
                'Spend=%{x}',
                'ROAS=%{y}'

            ])
            )
        )
        st.plotly_chart(fig_campaign_scatter, use_container_width=True)
        
        st.markdown("---")

        # --- 4. Raw Data View ---
        st.header("Explore Raw Data")
        if st.checkbox("Show Filtered Raw Data"):
            st.dataframe(df_filtered)

