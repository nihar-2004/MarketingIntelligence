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
        return {'revenue': 0, 'spend': 0, 'new_customers': 0, 'roas': 0, 'cac': 0, 'ctr': 0, 'aov': 0}
    
    revenue = data.groupby('date')['total_revenue'].first().sum()
    spend = data['spend'].sum()
    new_customers = data.groupby('date')['new_customers'].first().sum()
    
    roas = (data['attributed revenue'].sum() / spend) if spend > 0 else 0
    cac = (spend / new_customers) if new_customers > 0 else 0
    
    total_clicks = data['clicks'].sum()
    total_impressions = data['impression'].sum()
    ctr = (total_clicks / total_impressions) * 100 if total_impressions > 0 else 0
    aov = (revenue / data.groupby('date')['orders'].first().sum()) if data['orders'].sum() > 0 else 0
    cpc = (spend / total_clicks) if total_clicks > 0 else 0
    return {'revenue': revenue, 'spend': spend, 'new_customers': new_customers, 'roas': roas, 'cac': cac, 'ctr': ctr, 'aov': aov, 'cpc': cpc}


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
    default_start_date = max_date - timedelta(days=30)
    if default_start_date < min_date:
        default_start_date = min_date
    
    with st.sidebar.expander('Date Range', True):
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

    with st.sidebar.expander('Platform(s)', True):
        options = df['platform'].unique().tolist()
        selected_platforms = sac.checkbox(items=options, label='Select Platform(s)', index=list(range(len(options))), align='center', check_all='Select all')

    with st.sidebar.expander('Tactic(s)', True):
        options = df['tactic'].unique().tolist()
        selected_tactics = sac.checkbox(items=options, label="Select Tactic(s)", index=list(range(len(options))), align='center', check_all='Select all')

    with st.sidebar.expander('State(s)', True):
        options = df['state'].unique().tolist()
        selected_states = sac.checkbox(items=options, label="Select State(s)", index=list(range(len(options))), align='center', check_all='Select all')
    
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
        

        # --- NEW: Quick Insights Section ---
        selected_stats = sac.tabs([
            sac.TabsItem(label='Overview', ),
            sac.TabsItem(label='Platforms'),
            sac.TabsItem(label='Tactics'),
            sac.TabsItem(label='Performance'),
        ], align='center')
        
        
        if selected_stats == 'Overview':
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
            delta_aov = get_delta(kpis_current['aov'], kpis_previous['aov'])
            delta_cpc = get_delta(kpis_current['cpc'], kpis_previous['cpc'])

            col1, col2, col3, col4= st.columns(4)
            col1.metric("Total Revenue", f"${kpis_current['revenue']:,.2f}", f"{delta_revenue:.2f}%" if delta_revenue is not None else "N/A")
            col2.metric("Total Ad Spend", f"${kpis_current['spend']:,.2f}", f"{delta_spend:.2f}%" if delta_spend is not None else "N/A", delta_color="inverse")
            col3.metric("Overall ROAS", f"{kpis_current['roas']:.2f}x", f"{delta_roas:.2f}%" if delta_roas is not None else "N/A")
            col4.metric("New Customers", f"{int(kpis_current['new_customers']):,}", f"{delta_new_customers:.2f}%" if delta_new_customers is not None else "N/A")
            col5, col6, col7, col8 = st.columns(4)
        
            col5.metric("Average CAC", f"${kpis_current['cac']:,.2f}", f"{delta_cac:.2f}%" if delta_cac is not None else "N/A", delta_color="inverse")
            col6.metric("Average CTR", f"{kpis_current['ctr']:.2f}%", f"{delta_ctr:.2f}%" if delta_ctr is not None else "N/A")
            col7.metric("Average AOV", f"${kpis_current['aov']:,.2f}", f"{delta_aov:.2f}%" if delta_aov is not None else "N/A")
            col8.metric("Average CPC", f"${kpis_current['cpc']:,.2f}", f"{delta_cpc:.2f}%" if delta_cpc is not None else "N/A")
            st.markdown("---")
            # Calculate aggregations for insights
            platform_agg_insights = df_filtered.groupby('platform').agg(spend=('spend', 'sum'), attributed_revenue=('attributed revenue', 'sum')).reset_index()
            platform_agg_insights['roas'] = (platform_agg_insights['attributed_revenue'] / platform_agg_insights['spend']).fillna(0)
            
            tactic_agg_insights = df_filtered.groupby('tactic').agg(spend=('spend', 'sum'), attributed_revenue=('attributed revenue', 'sum')).reset_index()
            tactic_agg_insights['roas'] = (tactic_agg_insights['attributed_revenue'] / tactic_agg_insights['spend']).fillna(0)

            campaign_agg_insights = df_filtered.groupby('campaign').agg(attributed_revenue=('attributed revenue', 'sum')).reset_index()

            # Find best/worst performers
            best_platform = platform_agg_insights.loc[platform_agg_insights['roas'].idxmax()]
            best_tactic = tactic_agg_insights.loc[tactic_agg_insights['roas'].idxmax()]
            top_campaign = campaign_agg_insights.loc[campaign_agg_insights['attributed_revenue'].idxmax()]
            st.header("Quick Insights")
            insight_col1, insight_col2, insight_col3 = st.columns(3)
            with insight_col1:
                st.info(f"**Best Platform by ROAS**\n\n🚀 **{best_platform['platform']}** delivered the highest return, with a ROAS of **{best_platform['roas']:.2f}x**.", icon="🥇")
            with insight_col2:
                st.info(f"**Most Efficient Tactic**\n\n🎯 **{best_tactic['tactic']}** was the most efficient strategy, achieving a ROAS of **{best_tactic['roas']:.2f}x**.", icon="🎯")
            with insight_col3:
                st.info(f"**Top Revenue Campaign**\n\n🏆 **{top_campaign['campaign']}** was the top performer, generating **${top_campaign['attributed_revenue']:,.0f}** in attributed revenue.", icon="🏆")
            
            st.markdown("---")
            daily_agg = df_filtered.groupby('date').agg(
            total_revenue=('total_revenue', 'first'),
            spend=('spend', 'sum'),
            clicks=('clicks', 'sum'),
            impression=('impression', 'sum')
            ).reset_index()
            daily_agg['ctr'] = (daily_agg['clicks'] / daily_agg['impression']) * 100

            tactic_col1, tactic_col2 = st.columns(2)
            with tactic_col1:
                fig_rev_spend = make_subplots(specs=[[{"secondary_y": True}]])
                fig_rev_spend.add_trace(go.Scatter(x=daily_agg['date'], y=daily_agg['total_revenue'], name='Total Revenue', mode='lines', line=dict(color='#1f77b4')), secondary_y=False)
                fig_rev_spend.add_trace(go.Scatter(x=daily_agg['date'], y=daily_agg['spend'], name='Ad Spend', mode='lines', line=dict(color='#ff7f0e')), secondary_y=True)
                fig_rev_spend.update_layout(title_text="<b>Daily Revenue vs. Ad Spend</b>", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                fig_rev_spend.update_yaxes(title_text="<b>Total Revenue ($)</b>", secondary_y=False)
                fig_rev_spend.update_yaxes(title_text="<b>Ad Spend ($)</b>", secondary_y=True)
                st.plotly_chart(fig_rev_spend, use_container_width=True)

            with tactic_col2:
                st.header("Marketing Funnel")
                st.metric("Total Impressions", f"{daily_agg['impression'].sum():,}")
                st.metric("Total Clicks", f"{daily_agg['clicks'].sum():,}")
                st.metric("Conversion Rate (CTR)", f"{daily_agg['ctr'].mean():.2f}%")
                st.metric("Attributed Revenue", f"${df_filtered['attributed revenue'].sum():,.2f}")

        elif selected_stats == 'Platforms':
            platcol1, platcol2 = st.columns(2)
            with platcol1:
                impression_agg = df_filtered.groupby(['date', 'platform'])['impression'].sum().reset_index()
                fig_impressions = px.line(
                    impression_agg, x='date', y='impression', color='platform',
                    title='<b>Daily Impressions by Platform</b>',
                    labels={'impression': 'Total Impressions', 'date': 'Date', 'platform': 'Platform'}
                )
                st.plotly_chart(fig_impressions, use_container_width=True)
            with platcol2:
                platform_agg_spend_rev = df_filtered.groupby('platform').agg({
                    'spend': 'sum',
                    'attributed revenue': 'sum'
                }).reset_index()
                
                platform_agg_melted = platform_agg_spend_rev.melt(
                    id_vars='platform', 
                    value_vars=['spend', 'attributed revenue'],
                    var_name='Metric', 
                    value_name='Amount'
                )
                
                fig_spend_rev = px.bar(
                    platform_agg_melted,
                    x='platform',
                    y='Amount',
                    color='Metric',
                    barmode='group',
                    title='<b>Spend vs. Attributed Revenue by Platform</b>',
                    labels={'Amount': 'Amount ($)', 'platform': 'Platform'},
                    text_auto=True
                )
                fig_spend_rev.update_traces(texttemplate='%{y:,.0f}', textposition='outside')
                st.plotly_chart(fig_spend_rev, use_container_width=True)
            platcol3, platcol4 = st.columns(2)
            with platcol3:
                platform_agg = df_filtered.groupby('platform').agg(
                    spend=('spend', 'sum'),
                    attributed_revenue=('attributed revenue', 'sum'),
                    clicks=('clicks', 'sum'),
                    impression=('impression', 'sum')
                ).reset_index()
                platform_agg['roas'] = (platform_agg['attributed_revenue'] / platform_agg['spend']).fillna(0)
                
                fig_platform = px.bar(
                    platform_agg.sort_values('roas', ascending=False), 
                    x='platform', y='roas',
                    title=f"<b>ROAS by Platform</b>",
                    labels={'platform': 'Platform', 'roas': 'Return On Ad Spend (ROAS)'}, color='platform', text_auto='.2f'
                )
                st.plotly_chart(fig_platform, use_container_width=True)
            with platcol4:
                platform_agg = df_filtered.groupby('platform').agg(
                    spend=('spend', 'sum'),
                    attributed_revenue=('attributed revenue', 'sum'),
                    clicks=('clicks', 'sum'),
                    impression=('impression', 'sum')
                ).reset_index()
                platform_agg['ctr'] = (platform_agg['clicks'] / platform_agg['impression'] * 100).fillna(0)
                
                fig_platform = px.bar(
                    platform_agg.sort_values('ctr', ascending=False), 
                    x='platform', y='ctr',
                    title=f"<b>CTR by Platform</b>",
                    labels={'platform': 'Platform', 'ctr': 'Click-Through Rate (CTR %)'}, color='platform', text_auto='.2f'
                )
                st.plotly_chart(fig_platform, use_container_width=True)
        elif selected_stats == 'Tactics':
            taccol1, taccol2 = st.columns(2)
            with taccol1:
                tactic_agg = df_filtered.groupby('tactic')['impression'].sum().reset_index()
                fig_pie = px.pie(
                    tactic_agg, 
                    names='tactic', 
                    values='impression',
                    title='<b>Share of Impressions by Tactic</b>',
                    hole=0.4
                )
                fig_pie.update_traces(textinfo='percent+label')
                st.plotly_chart(fig_pie, use_container_width=True)
            with taccol2:
                tactic_agg_spend_rev = df_filtered.groupby('tactic').agg({
                    'spend': 'sum',
                    'attributed revenue': 'sum'
                }).reset_index()
                
                tactic_agg_melted = tactic_agg_spend_rev.melt(
                    id_vars='tactic', 
                    value_vars=['spend', 'attributed revenue'],
                    var_name='Metric', 
                    value_name='Amount'
                )
                
                fig_spend_rev = px.bar(
                    tactic_agg_melted,
                    x='tactic',
                    y='Amount',
                    color='Metric',
                    barmode='group',
                    title='<b>Spend vs. Attributed Revenue by Tactic</b>',
                    labels={'Amount': 'Amount ($)', 'tactic': 'Tactic'},
                    text_auto=True
                )
                fig_spend_rev.update_traces(texttemplate='%{y:,.0f}', textposition='outside')
                st.plotly_chart(fig_spend_rev, use_container_width=True)
            taccol3, taccol4 = st.columns(2)
            with taccol3:
                tactic_agg = df_filtered.groupby('tactic').agg(
                    spend=('spend', 'sum'),
                    attributed_revenue=('attributed revenue', 'sum'),
                    clicks=('clicks', 'sum'),
                    impression=('impression', 'sum')
                ).reset_index()
                tactic_agg['roas'] = (tactic_agg['attributed_revenue'] / tactic_agg['spend']).fillna(0)
                tactic_agg['ctr'] = (tactic_agg['clicks'] / tactic_agg['impression'] * 100).fillna(0)

                fig_tactic = px.bar(
                    tactic_agg.sort_values('roas', ascending=False), 
                    x='tactic', y='roas',
                    title=f"<b>ROAS by Tactic</b>",
                    labels={'tactic': 'Tactic', 'roas': 'Return On Ad Spend (ROAS)'}, color='tactic', text_auto='.2f'
                )
                st.plotly_chart(fig_tactic, use_container_width=True)
            with taccol4:
                tactic_agg = df_filtered.groupby('tactic').agg(
                    spend=('spend', 'sum'),
                    attributed_revenue=('attributed revenue', 'sum'),
                    clicks=('clicks', 'sum'),
                    impression=('impression', 'sum')
                ).reset_index()
                tactic_agg['ctr'] = (tactic_agg['clicks'] / tactic_agg['impression'] * 100).fillna(0)

                fig_tactic = px.bar(
                    tactic_agg.sort_values('ctr', ascending=False), 
                    x='tactic', y='ctr',
                    title=f"<b>CTR by Tactic</b>",
                    labels={'tactic': 'Tactic', 'ctr': 'Click-Through Rate (CTR %)'}, color='tactic', text_auto='.2f'
                )
                st.plotly_chart(fig_tactic, use_container_width=True)
        
        
        
        


        

        


        st.markdown("---")
        st.header("Campaign Level Analysis")
        groupbystate = st.radio("Group Campaign Scatter by:", options=['Platform', 'State'], horizontal=True)
        
        criteria = 'platform' if groupbystate == 'Platform' else 'state'
        campaign_agg = df_filtered.groupby(['campaign', criteria]).agg(
            spend=('spend', 'sum'),
            attributed_revenue=('attributed revenue', 'sum')
        ).reset_index()
        campaign_agg['roas'] = (campaign_agg['attributed_revenue'] / campaign_agg['spend']).fillna(0)
        campaign_agg['size_exaggerated'] = campaign_agg['attributed_revenue'] ** 1.5
        
        fig_campaign_scatter = px.scatter(
            campaign_agg, 
            x='spend', 
            y='roas',
            size='size_exaggerated', 
            color=criteria, 
            hover_name='campaign',
            hover_data={'size_exaggerated': False, 'attributed_revenue': True},
            title='<b>Campaign Efficiency (Spend vs. ROAS)</b>',
            labels={'platform': 'Platform', 'spend': 'Total Spend ($)', 'roas': 'ROAS', 'attributed_revenue': 'Attributed Revenue'},
            size_max=60, 
            log_x=True
        )
        st.plotly_chart(fig_campaign_scatter, use_container_width=True)
        st.markdown("---")

        # --- 4. Raw Data View ---
        st.header("Explore Raw Data")
        if st.checkbox("Show Filtered Raw Data"):
            st.dataframe(df_filtered)

