# COVID-19 Interactive Dashboard with Streamlit
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os
from forecasting_models import run_forecasting_models

# Set page configuration
st.set_page_config(
    page_title="COVID-19 Analysis Dashboard",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.8rem;
        color: #1565C0;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #E3F2FD;
        padding-bottom: 0.5rem;
        font-weight: 600;
    }
    .metric-card {
        background-color: #E3F2FD;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1565C0;
    }
    .metric-label {
        font-size: 1rem;
        color: #424242;
    }
    .footer {
        font-size: 0.9rem;
        color: #757575;
        text-align: center;
        margin-top: 3rem;
        border-top: 1px solid #E0E0E0;
        padding-top: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 4rem;
        white-space: pre-wrap;
        background-color: #E3F2FD;
        border-radius: 4px 4px 0 0;
        gap: 1rem;
        padding-top: 0.5rem;
        padding-bottom: 0.5rem;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E88E5;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and prepare the data for the dashboard"""
    try:
        # Load processed data - adjust path as needed
        df = pd.read_csv("C:/Users/Yuvraj/Desktop/covid_analysis_report/data/processed/processed_covid_data.csv")
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Load forecasts if available
        forecast_file = 'combined_forecasts.csv'
        if os.path.exists(forecast_file):
            forecasts = pd.read_csv(forecast_file)
            forecasts['Date'] = pd.to_datetime(forecasts['Date'])
        else:
            forecasts = None
            
        return df, forecasts
    except Exception as e:
        st.error(f"Error loading data: {e}")
        # Return empty DataFrames if loading fails
        return pd.DataFrame(), pd.DataFrame()

# Load the data
covid_data, forecast_data = load_data()

# Header
st.markdown("<h1 class='main-header'>COVID-19 Analysis Dashboard</h1>", unsafe_allow_html=True)
st.markdown("""
    <div style='text-align: center; color: #424242; margin-bottom: 2rem;'>
        Interactive visualization and analysis of COVID-19 trends, forecasts, and statistics
    </div>
""", unsafe_allow_html=True)

# Sidebar - Controls
with st.sidebar:
    st.markdown("### 📊 Dashboard Controls")
    st.markdown("---")
    
    # Country selection
    if not covid_data.empty:
        countries = sorted(covid_data['Country/Region'].unique())
        default_countries = ['US', 'India', 'Brazil', 'UK', 'France'] if len(countries) >= 5 else countries[:5]
        default_countries = [c for c in default_countries if c in countries]
        
        selected_countries = st.multiselect(
            "🌍 Select Countries",
            options=countries,
            default=default_countries
        )
    else:
        st.warning("No country data available")
        selected_countries = []

    st.markdown("---")
    
    # Metric selection
    metric_options = {
        'Confirmed': 'Total Confirmed Cases',
        'Deaths': 'Total Deaths',
        'Active': 'Active Cases',
        'New_Confirmed': 'New Cases (Daily)',
        'New_Deaths': 'New Deaths (Daily)',
        'MA7_New_Confirmed': '7-Day Average (New Cases)'
    }
    selected_metric = st.radio(
        "📈 Select Metric",
        list(metric_options.keys()),
        format_func=lambda x: metric_options[x],
        index=5
    )

    st.markdown("---")
    
    # Date range selection
    if not covid_data.empty:
        min_date = covid_data['Date'].min().date()
        max_date = covid_data['Date'].max().date()
        default_start = (max_date - timedelta(days=180))
        
        date_range = st.date_input(
            "📅 Select Date Range",
            value=(default_start, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        if len(date_range) == 2:
            start_date, end_date = date_range
        else:
            start_date = date_range[0]
            end_date = date_range[0]
    else:
        today = datetime.now().date()
        start_date = today - timedelta(days=180)
        end_date = today
        st.warning("No date data available")

    st.markdown("---")
    
    # Forecast model selection
    forecast_models = st.multiselect(
        "🔮 Select Forecast Models",
        options=['ARIMA', 'Prophet', 'LSTM'],
        default=['ARIMA']
    )

# Filter data based on selections
@st.cache_data
def filter_data(data, countries, start, end):
    """Filter the data based on selected countries and date range"""
    if data.empty or not countries:
        return pd.DataFrame()
        
    filtered = data[
        (data['Country/Region'].isin(countries)) &
        (data['Date'] >= pd.Timestamp(start)) &
        (data['Date'] <= pd.Timestamp(end))
    ]
    return filtered

# Apply filters to the data
filtered_data = filter_data(covid_data, selected_countries, start_date, end_date)

# Main dashboard tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Trend Analysis",
    "🔮 Forecasting",
    "📊 Statistics",
    "🗺️ Geographic View"
])

# Tab 1: Trend Analysis
with tab1:
    st.markdown("<h2 class='sub-header'>COVID-19 Trends by Country</h2>", unsafe_allow_html=True)
    
    if not filtered_data.empty:
        # Line chart for selected metric
        fig = px.line(
            filtered_data, 
            x='Date', 
            y=selected_metric, 
            color='Country/Region',
            title=f'{metric_options[selected_metric]} Over Time',
            labels={selected_metric: metric_options[selected_metric]},
            template='plotly_white'
        )
        
        # Enhance the figure
        fig.update_layout(
            legend_title_text='Country',
            xaxis_title='Date',
            yaxis_title=metric_options[selected_metric],
            hovermode='x unified',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Comparative Analysis
        st.markdown("<h2 class='sub-header'>Comparative Analysis</h2>", unsafe_allow_html=True)
        
        # Get latest data for selected countries
        latest_data = filtered_data.groupby('Country/Region').last().reset_index()
        
        if not latest_data.empty:
            # Create subplot with multiple metrics
            comp_fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Total Confirmed Cases', 'Case Fatality Rate (%)'),
                specs=[[{"type": "bar"}, {"type": "bar"}]]
            )
            
            # Add total cases bar chart
            comp_fig.add_trace(
                go.Bar(
                    x=latest_data['Country/Region'], 
                    y=latest_data['Confirmed'],
                    name='Confirmed Cases',
                    marker_color='#1E88E5'
                ),
                row=1, col=1
            )
            
            # Calculate and add CFR bar chart
            latest_data['CFR'] = (latest_data['Deaths'] / latest_data['Confirmed']) * 100
            comp_fig.add_trace(
                go.Bar(
                    x=latest_data['Country/Region'], 
                    y=latest_data['CFR'],
                    name='CFR (%)',
                    marker_color='#E53935'
                ),
                row=1, col=2
            )
            
            comp_fig.update_layout(
                height=500,
                title_text=f"Comparative Analysis as of {end_date.strftime('%Y-%m-%d')}",
                template='plotly_white',
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(comp_fig, use_container_width=True)
    else:
        st.warning("No data available for the selected countries and date range.")

# Tab 2: Forecasting
with tab2:
    st.markdown("<h2 class='sub-header'>COVID-19 Forecasts</h2>", unsafe_allow_html=True)
    
    if len(selected_countries) != 1:
        st.info("Please select exactly one country to view forecasts.")
    else:
        selected_country = selected_countries[0]
        
        # Get historical data for selected country
        country_data = covid_data[covid_data['Country/Region'] == selected_country]
        
        if country_data.empty:
            st.warning(f"No historical data found for {selected_country}.")
        else:
            # Run forecasting models if selected
            if forecast_models:
                try:
                    with st.spinner('Generating forecasts... This may take a few minutes.'):
                        forecast_results = run_forecasting_models(country=selected_country, forecast_days=30)
                    
                    # Create figure with historical data
                    fig = go.Figure()
                    
                    # Add historical data
                    fig.add_trace(go.Scatter(
                        x=country_data['Date'],
                        y=country_data['MA7_New_Confirmed'],
                        mode='lines',
                        name='Historical Data',
                        line=dict(color='#424242', width=2)
                    ))
                    
                    # Add forecasts
                    colors = {'ARIMA': '#1E88E5', 'Prophet': '#43A047', 'LSTM': '#E53935'}
                    available_models = []
                    
                    for model in forecast_models:
                        column_name = f'{model}_Forecast'
                        if column_name in forecast_results.columns and not forecast_results[column_name].isna().all():
                            fig.add_trace(go.Scatter(
                                x=forecast_results['Date'],
                                y=forecast_results[column_name],
                                mode='lines',
                                name=f'{model} Forecast',
                                line=dict(color=colors.get(model, '#757575'), dash='dash')
                            ))
                            available_models.append(model)
                    
                    if not available_models:
                        st.warning("No forecast models were able to generate predictions. Please try again later.")
                    else:
                        # Update layout
                        fig.update_layout(
                            title=f'COVID-19 Forecast for {selected_country}',
                            xaxis_title='Date',
                            yaxis_title='Daily New Cases (7-day MA)',
                            template='plotly_white',
                            hovermode='x unified',
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)'
                        )
                        
                        # Add vertical line separating historical data from forecasts
                        last_date = country_data['Date'].max()
                        fig.add_vline(x=last_date, line_width=1, line_dash="dash", line_color="gray")
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Model Comparison
                        if len(available_models) > 1:
                            st.markdown("<h2 class='sub-header'>Forecast Model Comparison</h2>", unsafe_allow_html=True)
                            
                            # Extract error metrics if they exist
                            st.write("Model Performance Metrics:")
                            
                            metrics_cols = st.columns(len(available_models))
                            for i, model in enumerate(available_models):
                                with metrics_cols[i]:
                                    st.metric(
                                        label=f"{model} Model",
                                        value="See Forecast",
                                        delta="MAPE: Not available"
                                    )
                                    
                            # Description of models
                            st.markdown("""
                            **Model Descriptions:**
                            - **ARIMA**: Statistical time-series forecasting model that uses past values and errors
                            - **Prophet**: Facebook's forecasting tool designed for business time series with seasonal patterns
                            - **LSTM**: Deep learning approach for sequence modeling using Long Short-Term Memory networks
                            """)
                except Exception as e:
                    st.error(f"Error running forecasting models: {str(e)}")
            else:
                st.info("Please select at least one forecasting model from the sidebar.")

# Tab 3: Statistics
with tab3:
    st.markdown("<h2 class='sub-header'>Key COVID-19 Statistics</h2>", unsafe_allow_html=True)
    
    if not filtered_data.empty:
        # Get latest data
        latest_global = filtered_data.groupby('Date').sum().reset_index().iloc[-1]
        latest_by_country = filtered_data.groupby(['Country/Region', 'Date']).sum().reset_index()
        latest_by_country = latest_by_country.sort_values('Date').groupby('Country/Region').last()
        latest_by_country['CFR'] = (latest_by_country['Deaths'] / latest_by_country['Confirmed']) * 100
        
        # Display key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
                <div class='metric-card'>
                    <div class='metric-value'>{:,.0f}</div>
                    <div class='metric-label'>Total Confirmed Cases</div>
                </div>
            """.format(latest_global['Confirmed']), unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
                <div class='metric-card'>
                    <div class='metric-value'>{:,.0f}</div>
                    <div class='metric-label'>Total Deaths</div>
                </div>
            """.format(latest_global['Deaths']), unsafe_allow_html=True)
            
        with col3:
            st.markdown("""
                <div class='metric-card'>
                    <div class='metric-value'>{:,.0f}</div>
                    <div class='metric-label'>Active Cases</div>
                </div>
            """.format(latest_global['Active']), unsafe_allow_html=True)
            
        with col4:
            avg_cfr = (latest_global['Deaths'] / latest_global['Confirmed']) * 100
            st.markdown("""
                <div class='metric-card'>
                    <div class='metric-value'>{:.2f}%</div>
                    <div class='metric-label'>Average CFR</div>
                </div>
            """.format(avg_cfr), unsafe_allow_html=True)
            
        # Country Rankings
        st.markdown("<h2 class='sub-header'>Country Rankings</h2>", unsafe_allow_html=True)
        
        rank_metric = st.selectbox(
            "Rank countries by:",
            options=['Confirmed', 'Deaths', 'CFR', 'MA7_New_Confirmed'],
            format_func=lambda x: {'Confirmed': 'Total Cases', 'Deaths': 'Total Deaths', 
                                  'CFR': 'Case Fatality Rate', 'MA7_New_Confirmed': 'New Cases (7-day MA)'}[x]
        )
        
        top_n = st.slider("Number of countries to show:", min_value=5, max_value=20, value=10)
        
        if rank_metric == 'CFR':
            # Calculate CFR for ranking
            ranked_countries = latest_by_country.sort_values('CFR', ascending=False).head(top_n)
            y_values = ranked_countries['CFR']
            y_title = 'Case Fatality Rate (%)'
        else:
            ranked_countries = latest_by_country.sort_values(rank_metric, ascending=False).head(top_n)
            y_values = ranked_countries[rank_metric]
            y_title = metric_options.get(rank_metric, rank_metric)
        
        # Create rank chart
        rank_fig = px.bar(
            x=ranked_countries.index,
            y=y_values,
            title=f"Top {top_n} Countries by {y_title}",
            labels={'x': 'Country', 'y': y_title},
            template='plotly_white'
        )
        
        rank_fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(rank_fig, use_container_width=True)
        
        # Growth Rate Analysis
        st.markdown("<h2 class='sub-header'>Growth Rate Analysis</h2>", unsafe_allow_html=True)
        
        if len(selected_countries) > 0:
            # Calculate weekly growth rate
            growth_data = filtered_data.copy()
            growth_data['Week'] = growth_data['Date'].dt.isocalendar().week
            growth_data['Year'] = growth_data['Date'].dt.isocalendar().year
            
            weekly_data = growth_data.groupby(['Country/Region', 'Year', 'Week']).agg({'Confirmed': 'max'}).reset_index()
            weekly_data['WeekYear'] = weekly_data['Year'].astype(str) + '-W' + weekly_data['Week'].astype(str)
            weekly_data['PrevConfirmed'] = weekly_data.groupby('Country/Region')['Confirmed'].shift(1)
            weekly_data['WeeklyGrowth'] = (weekly_data['Confirmed'] - weekly_data['PrevConfirmed']) / weekly_data['PrevConfirmed'] * 100
            weekly_data = weekly_data.dropna()
            
            # Create growth rate chart
            growth_fig = px.line(
                weekly_data,
                x='WeekYear',
                y='WeeklyGrowth',
                color='Country/Region',
                title='Weekly Growth Rate of Confirmed Cases',
                labels={'WeeklyGrowth': 'Weekly Growth Rate (%)', 'WeekYear': 'Year-Week'},
                template='plotly_white'
            )
            
            growth_fig.update_layout(
                hovermode='x unified',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(growth_fig, use_container_width=True)
        else:
            st.info("Select countries to view growth rate analysis.")
    else:
        st.warning("No data available for the selected date range and countries.")

# Tab 4: Geographic View
with tab4:
    st.markdown("<h2 class='sub-header'>Global Distribution</h2>", unsafe_allow_html=True)
    
    if not covid_data.empty:
        # Get latest data for all countries
        latest_date = covid_data['Date'].max()
        map_data = covid_data[covid_data['Date'] == latest_date]
        
        # Calculate CFR
        map_data['CFR'] = (map_data['Deaths'] / map_data['Confirmed']) * 100
        
        # Select map metric
        map_metric = st.selectbox(
            "Select map metric:",
            options=['Confirmed', 'Deaths', 'CFR', 'MA7_New_Confirmed'],
            format_func=lambda x: {'Confirmed': 'Total Cases', 'Deaths': 'Total Deaths', 
                                  'CFR': 'Case Fatality Rate', 'MA7_New_Confirmed': 'New Cases (7-day MA)'}[x]
        )
        
        # Create choropleth map
        color_scale = 'Reds' if map_metric in ['Confirmed', 'Deaths', 'MA7_New_Confirmed'] else 'RdBu'
        map_fig = px.choropleth(
            map_data,
            locations='Country/Region',
            locationmode='country names',
            color=map_metric,
            hover_name='Country/Region',
            projection='natural earth',
            color_continuous_scale=color_scale,
            title=f'{metric_options.get(map_metric, map_metric)} by Country (as of {latest_date.strftime("%Y-%m-%d")})',
            hover_data=['Confirmed', 'Deaths', 'MA7_New_Confirmed']
        )
        
        map_fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(map_fig, use_container_width=True)
        
        # Add map data table
        st.markdown("<h2 class='sub-header'>Data Table</h2>", unsafe_allow_html=True)
        
        # Select columns to display
        display_cols = ['Country/Region', 'Confirmed', 'Deaths', 'Active', 'CFR', 'MA7_New_Confirmed']
        display_map_data = map_data[display_cols].sort_values('Confirmed', ascending=False)
        
        # Format CFR as percentage
        display_map_data['CFR'] = display_map_data['CFR'].round(2).astype(str) + '%'
        
        # Rename columns for display
        display_map_data = display_map_data.rename(columns={
            'Country/Region': 'Country',
            'Confirmed': 'Total Cases',
            'Deaths': 'Total Deaths',
            'Active': 'Active Cases',
            'CFR': 'Case Fatality Rate',
            'MA7_New_Confirmed': '7-day MA (New Cases)'
        })
        
        st.dataframe(display_map_data, use_container_width=True)
    else:
        st.warning("No geographic data available.")

# Footer
st.markdown(f"""
<div class='footer'>
COVID-19 Data Analysis Dashboard<br>
Data range: {covid_data['Date'].min().strftime('%Y-%m-%d') if not covid_data.empty else 'N/A'} to 
{covid_data['Date'].max().strftime('%Y-%m-%d') if not covid_data.empty else 'N/A'}<br>
Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
</div>
""", unsafe_allow_html=True)
