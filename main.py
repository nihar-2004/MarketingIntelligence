import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# --- Page Configuration ---
st.set_page_config(
    page_title="Marketing Intelligence Dashboard",
    page_icon="📊",
    layout="wide"
)

# --- Data Loading ---
# Use st.cache_data to prevent reloading data on every interaction
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

# --- Main App ---
if df is None:
    st.error("Data could not be loaded. Please check the CSV file paths and content.")
else:
    st.title("📊 Marketing Intelligence Dashboard")
    st.markdown("Analyze marketing spend and its connection to business outcomes.")

    # --- Sidebar Filters ---
    st.sidebar.header("Dashboard Filters")

    # Date Range Filter
    min_date = df['date'].min().date()
    max_date = df['date'].max().date()
    start_date, end_date = st.sidebar.slider(
        "Select Date Range",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        format="YYYY-MM-DD"
    )

    # Convert slider dates to datetime for filtering
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Other Multiselect Filters
    selected_platforms = st.sidebar.multiselect(
        "Select Platform(s)",
        options=df['platform'].unique(),
        default=df['platform'].unique()
    )

    selected_tactics = st.sidebar.multiselect(
        "Select Tactic(s)",
        options=df['tactic'].unique(),
        default=df['tactic'].unique()
    )

    selected_states = st.sidebar.multiselect(
        "Select State(s)",
        options=df['state'].unique(),
        default=df['state'].unique()
    )

    # --- Filter DataFrame based on selections ---
    mask = (
        (df['date'] >= start_date) & (df['date'] <= end_date) &
        (df['platform'].isin(selected_platforms)) &
        (df['tactic'].isin(selected_tactics)) &
        (df['state'].isin(selected_states))
    )
    df_filtered = df[mask]

    if df_filtered.empty:
        st.warning("No data available for the selected filters. Please adjust your selections.")
    else:
        # --- 1. Executive Summary (KPIs) ---
        st.header("Executive Summary")

        # Calculate KPIs from the filtered dataframe
        total_revenue = df_filtered['total_revenue'].unique().sum()
        total_spend = df_filtered['spend'].sum()
        total_new_customers = df_filtered['new_customers'].unique().sum()
        
        # Calculate overall ROAS and CAC safely
        overall_roas = (df_filtered['attributed revenue'].sum() / total_spend) if total_spend > 0 else 0
        overall_cac = (total_spend / total_new_customers) if total_new_customers > 0 else 0

        # Display KPIs in columns
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Revenue", f"${total_revenue:,.2f}")
        col2.metric("Total Ad Spend", f"${total_spend:,.2f}")
        col3.metric("Overall ROAS", f"{overall_roas:.2f}x")
        col4.metric("Total New Customers", f"{total_new_customers:,}")
        col5.metric("Average CAC", f"${overall_cac:,.2f}")

        st.markdown("---")


        # --- 2. Performance Trend Analysis ---
        st.header("Performance Trend Analysis")

        # Aggregate data daily for trend charts
        daily_agg = df_filtered.groupby('date').agg({
            'total_revenue': 'first', # Since it's daily business data
            'spend': 'sum',
            'roas': 'mean',
            'cac': 'first' # Since CAC is calculated daily
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
                platform_agg, x='platform', y='roas',
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
                tactic_agg, x='tactic', y='roas',
                title="<b>ROAS by Tactic</b>",
                labels={'roas': 'Return on Ad Spend (ROAS)', 'tactic': 'Tactic'},
                color='tactic',
                text_auto='.2f'
            )
            st.plotly_chart(fig_tactic, use_container_width=True)

        # Chart 5: Spend vs. ROAS by Campaign (Scatter Plot)
        campaign_agg = df_filtered.groupby(['campaign', 'platform']).agg({
            'spend': 'sum',
            'attributed revenue': 'sum'
        }).reset_index()
        campaign_agg['roas'] = (campaign_agg['attributed revenue'] / campaign_agg['spend']).fillna(0)
        
        fig_campaign_scatter = px.scatter(
            campaign_agg,
            x='spend',
            y='roas',
            size='attributed revenue',
            color='platform',
            hover_name='campaign',
            title='<b>Campaign Efficiency (Spend vs. ROAS)</b>',
            labels={'spend': 'Total Spend ($)', 'roas': 'ROAS', 'attributed revenue': 'Attributed Revenue'},
            size_max=60,
            log_x=True # Use log scale for spend if values vary widely
        )
        st.plotly_chart(fig_campaign_scatter, use_container_width=True)
        
        st.markdown("---")

        # --- 4. Raw Data View ---
        st.header("Explore Raw Data")
        if st.checkbox("Show Filtered Raw Data"):
            st.dataframe(df_filtered)
