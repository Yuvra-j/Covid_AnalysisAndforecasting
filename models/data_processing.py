# COVID-19 Trend Analysis Project
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")
sns.set_context("notebook", font_scale=1.2)

print("Loading and preprocessing COVID-19 data...")

df_confirmed = pd.read_csv("C:/Users/Yuvraj/Desktop/covid_analysis_report/data/raw/time_series_covid19_confirmed_global.csv")
df_deaths = pd.read_csv("C:/Users/Yuvraj/Desktop/covid_analysis_report/data/raw/time_series_covid19_deaths_global.csv")
df_recovered = pd.read_csv("C:/Users/Yuvraj/Desktop/covid_analysis_report/data/raw/time_series_covid19_recovered_global.csv")

def preprocess_jhu_data(df, case_type):
    
    id_vars = ['Province/State', 'Country/Region', 'Lat', 'Long']
    df_melted = df.melt(id_vars=id_vars, var_name='Date', value_name=case_type)
    
    df_melted['Date'] = pd.to_datetime(df_melted['Date'])
    
    df_grouped = df_melted.groupby(['Country/Region', 'Date'])[case_type].sum().reset_index()
    
    return df_grouped

# dataset loading 
confirmed_long = preprocess_jhu_data(df_confirmed, 'Confirmed')
deaths_long = preprocess_jhu_data(df_deaths, 'Deaths')
recovered_long = preprocess_jhu_data(df_recovered, 'Recovered')

covid_data = confirmed_long.merge(deaths_long, on=['Country/Region', 'Date'])
covid_data = covid_data.merge(recovered_long, on=['Country/Region', 'Date'])

# active cases
covid_data['Active'] = covid_data['Confirmed'] - covid_data['Deaths'] - covid_data['Recovered']

#  daily new cases, deaths, and recoveries
covid_data['New_Confirmed'] = covid_data.groupby('Country/Region')['Confirmed'].diff().fillna(0)
covid_data['New_Deaths'] = covid_data.groupby('Country/Region')['Deaths'].diff().fillna(0)
covid_data['New_Recovered'] = covid_data.groupby('Country/Region')['Recovered'].diff().fillna(0)

#  7-day moving averages
covid_data['MA7_New_Confirmed'] = covid_data.groupby('Country/Region')['New_Confirmed'].rolling(7).mean().reset_index(level=0, drop=True).fillna(0)
covid_data['MA7_New_Deaths'] = covid_data.groupby('Country/Region')['New_Deaths'].rolling(7).mean().reset_index(level=0, drop=True).fillna(0)

# Calculate growth rates (day-over-day percent change)
covid_data['Growth_Rate_Confirmed'] = covid_data.groupby('Country/Region')['Confirmed'].pct_change().fillna(0) * 100

print(f"Data processed. Dataset contains {len(covid_data['Country/Region'].unique())} countries from {covid_data['Date'].min().strftime('%Y-%m-%d')} to {covid_data['Date'].max().strftime('%Y-%m-%d')}")

print("\nPerforming exploratory data analysis...")

top_countries_by_cases = covid_data.groupby('Country/Region')['Confirmed'].max().sort_values(ascending=False).head(10).index.tolist()
    
top_countries_data = covid_data[covid_data['Country/Region'].isin(top_countries_by_cases)]

# VISUALIZATIONS
print("\nCreating visualizations...")
plt.figure(figsize=(12, 6))
global_cases = covid_data.groupby('Date')[['Confirmed', 'Deaths', 'Recovered', 'Active']].sum()
global_cases.plot(title='Global COVID-19 Cases Over Time', linewidth=2)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlabel('Date')
plt.ylabel('Number of Cases')
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig('global_cumulative_cases.png')

plt.figure(figsize=(12, 6))
daily_global = covid_data.groupby('Date')[['New_Confirmed', 'New_Deaths']].sum()
daily_global.plot(title='Global Daily New COVID-19 Cases and Deaths', linewidth=1)
daily_global['MA7_New_Confirmed'] = daily_global['New_Confirmed'].rolling(7).mean()
daily_global['MA7_New_Deaths'] = daily_global['New_Deaths'].rolling(7).mean()
daily_global[['MA7_New_Confirmed', 'MA7_New_Deaths']].plot(linewidth=2)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlabel('Date')
plt.ylabel('Number of New Cases')
plt.legend(['New Cases', 'New Deaths', '7-Day MA (Cases)', '7-Day MA (Deaths)'])
plt.tight_layout()
plt.savefig('global_daily_new_cases.png')

fig = px.line(top_countries_data, x='Date', y='Confirmed', color='Country/Region', 
              title='COVID-19 Confirmed Cases in Top 10 Countries',
              hover_data=['Deaths', 'Recovered', 'Active'])

fig.update_layout(
    xaxis_title='Date',
    yaxis_title='Confirmed Cases',
    legend_title='Country',
    hovermode='x unified'
)
fig.write_html('top_countries_cases.html')

top_countries_data['CFR'] = (top_countries_data['Deaths'] / top_countries_data['Confirmed']) * 100

plt.figure(figsize=(12, 6))
for country in top_countries_by_cases:
    country_data = top_countries_data[top_countries_data['Country/Region'] == country]
    plt.plot(country_data['Date'], country_data['CFR'], label=country, linewidth=2)

plt.title('Case Fatality Rate (CFR) Over Time for Top 10 Countries')
plt.xlabel('Date')
plt.ylabel('CFR (%)')
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('cfr_over_time.png')

print("\nPerforming time series analysis...")

us_data = covid_data[covid_data['Country/Region'] == 'US'].copy()

us_data['Growth_Rate'] = us_data['Confirmed'].pct_change().fillna(0)
us_data['Doubling_Time'] = np.log(2) / np.log(1 + us_data['Growth_Rate'])
us_data['Doubling_Time'].replace([np.inf, -np.inf], np.nan, inplace=True)
us_data['Doubling_Time_MA7'] = us_data['Doubling_Time'].rolling(7).mean()

plt.figure(figsize=(12, 6))
plt.plot(us_data['Date'], us_data['Doubling_Time_MA7'], linewidth=2)
plt.title('COVID-19 Case Doubling Time in the United States (7-Day Moving Average)')
plt.xlabel('Date')
plt.ylabel('Doubling Time (Days)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.ylim(bottom=0)
plt.tight_layout()
plt.savefig('us_doubling_time.png')

print("\nPerforming advanced analytics...")

latest_date = covid_data['Date'].max()
start_date = latest_date - timedelta(days=30)
recent_data = covid_data[(covid_data['Date'] >= start_date) & (covid_data['Date'] <= latest_date)]

features_data = recent_data.pivot_table(
    index='Country/Region',
    columns='Date',
    values='New_Confirmed'
).fillna(0)

total_cases = covid_data.groupby('Country/Region')['Confirmed'].max()
significant_countries = total_cases[total_cases > 1000].index
features_data = features_data[features_data.index.isin(significant_countries)]

# Standardize the data
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_data)

# Apply KMeans clustering
n_clusters = 4  # Number of clusters to create
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(features_scaled)

# Add cluster information back to the original data
cluster_mapping = pd.DataFrame({'Country/Region': features_data.index, 'Cluster': clusters})
covid_data_with_clusters = covid_data.merge(cluster_mapping, on='Country/Region', how='left')

plt.figure(figsize=(12, 8))
for cluster in range(n_clusters):
    countries_in_cluster = cluster_mapping[cluster_mapping['Cluster'] == cluster]['Country/Region'].tolist()
    cluster_data = covid_data_with_clusters[covid_data_with_clusters['Country/Region'].isin(countries_in_cluster)]
    
    avg_cases = cluster_data.groupby('Date')['MA7_New_Confirmed'].mean()
    plt.plot(avg_cases.index, avg_cases.values, linewidth=2, label=f'Cluster {cluster+1} (n={len(countries_in_cluster)})')

plt.title('Average COVID-19 Trajectories by Country Cluster')
plt.xlabel('Date')
plt.ylabel('Average Daily New Cases (7-Day MA)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('country_clusters.png')

# PART 6: GENERATE FINAL REPORT
print("\nGenerating final report...")
print("Analysis complete! Results saved as images and HTML files.")
print("\nSummary of findings:")
print(f"- Dataset covers {len(covid_data['Country/Region'].unique())} countries from {covid_data['Date'].min().strftime('%Y-%m-%d')} to {covid_data['Date'].max().strftime('%Y-%m-%d')}")
print(f"- Top countries by total confirmed cases: {', '.join(top_countries_by_cases[:5])}")
print(f"- {n_clusters} distinct patterns of infection spread identified through clustering analysis")
print("- Visualizations created: global trends, country comparisons, and pattern analysis")

if __name__ == '__main__':
    covid_data.to_csv("processed_covid_data.csv", index=False)
    print("\nProcessed data saved as 'processed_covid_data.csv'")

