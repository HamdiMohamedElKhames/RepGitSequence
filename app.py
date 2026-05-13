import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Energy Consumption Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better appearance
st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.5rem;
    }
    .main-header p {
        color: rgba(255,255,255,0.9);
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Set seaborn style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Energy_consumption.csv")
    df = df.dropna().drop_duplicates()
    
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # Extract time features
    df['Hour'] = df['Timestamp'].dt.hour
    df['Day'] = df['Timestamp'].dt.day
    df['Month'] = df['Timestamp'].dt.month
    df['Year'] = df['Timestamp'].dt.year
    df['DayOfWeekNum'] = df['Timestamp'].dt.dayofweek
    df['Weekend'] = (df['DayOfWeekNum'] >= 5).astype(int)
    
    # Time of day categories
    df['TimeOfDay'] = pd.cut(df['Hour'], 
                              bins=[0, 6, 12, 18, 24], 
                              labels=['Night', 'Morning', 'Afternoon', 'Evening'],
                              include_lowest=True)
    
    return df

df = load_data()

# -----------------------------
# SIDEBAR FILTERS
# -----------------------------
st.sidebar.markdown("# ⚡ Dashboard Controls")
st.sidebar.markdown("---")

# Date range filter
min_date = df['Timestamp'].min().date()
max_date = df['Timestamp'].max().date()
selected_date_range = st.sidebar.date_input(
    "Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if len(selected_date_range) == 2:
    start_date, end_date = selected_date_range
    df_filtered = df[(df['Timestamp'].dt.date >= start_date) & 
                      (df['Timestamp'].dt.date <= end_date)]
else:
    df_filtered = df.copy()

# Month filter
selected_month = st.sidebar.multiselect(
    "Select Month",
    options=sorted(df["Month"].unique()),
    default=sorted(df["Month"].unique())
)

# Hour range filter
selected_hour = st.sidebar.slider("Hour Range", 0, 23, (0, 23))

# Day filter
selected_day = st.sidebar.multiselect(
    "Select Day of Week",
    options=sorted(df["DayOfWeek"].unique()),
    default=sorted(df["DayOfWeek"].unique())
)

# Holiday filter
holiday_filter = st.sidebar.radio(
    "Holiday Status",
    ["All", "Holiday Only", "Non-Holiday Only"]
)

# Apply filters
df_filtered = df_filtered[
    (df_filtered["Month"].isin(selected_month)) &
    (df_filtered["Hour"].between(selected_hour[0], selected_hour[1])) &
    (df_filtered["DayOfWeek"].isin(selected_day))
]

if holiday_filter == "Holiday Only":
    df_filtered = df_filtered[df_filtered["Holiday"] == "Yes"]
elif holiday_filter == "Non-Holiday Only":
    df_filtered = df_filtered[df_filtered["Holiday"] == "No"]

# -----------------------------
# MAIN HEADER
# -----------------------------
st.markdown("""
    <div class="main-header">
        <h1>⚡ Smart Energy Consumption Dashboard</h1>
        <p>Analyze, Visualize, and Predict Building Energy Usage</p>
    </div>
""", unsafe_allow_html=True)

# Key Metrics
st.markdown("### 📊 Key Performance Indicators")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Total Records", f"{df_filtered.shape[0]:,}", delta=None)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    avg_consumption = df_filtered['EnergyConsumption'].mean()
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Avg Consumption", f"{avg_consumption:.1f} kWh", 
              delta=f"{avg_consumption - df['EnergyConsumption'].mean():.1f}")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    peak_consumption = df_filtered['EnergyConsumption'].max()
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Peak Consumption", f"{peak_consumption:.1f} kWh", delta=None)
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    avg_temp = df_filtered['Temperature'].mean()
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Avg Temperature", f"{avg_temp:.1f}°C", delta=None)
    st.markdown('</div>', unsafe_allow_html=True)

with col5:
    avg_renewable = df_filtered['RenewableEnergy'].mean()
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric("Avg Renewable", f"{avg_renewable:.1f} kWh", delta=None)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# -----------------------------
# NAVIGATION TABS
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📁 Data Explorer", "📊 Visualizations", "🤖 Prediction Model", "📈 Insights"])

# -----------------------------
# TAB 1: DATA EXPLORER
# -----------------------------
with tab1:
    st.subheader("📂 Dataset Overview")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write("**Data Preview**")
        st.dataframe(df_filtered.head(100), use_container_width=True)
    
    with col2:
        st.write("**Data Info**")
        st.write(f"**Shape:** {df_filtered.shape[0]} rows × {df_filtered.shape[1]} columns")
        st.write(f"**Date Range:** {df_filtered['Timestamp'].min()} to {df_filtered['Timestamp'].max()}")
        st.write(f"**Missing Values:** {df_filtered.isnull().sum().sum()}")
        st.write("**Data Types:**")
        st.write(df_filtered.dtypes.value_counts())
    
    st.subheader("📋 Statistical Summary")
    st.dataframe(df_filtered.describe(), use_container_width=True)

# -----------------------------
# TAB 2: VISUALIZATIONS
# -----------------------------
with tab2:
    st.subheader("📊 Interactive Visualizations")
    
    # Visualization type selector
    viz_type = st.selectbox(
        "Select Visualization Type",
        ["Distribution Analysis", "Time Series Analysis", "Correlation Analysis", "Comparative Analysis"]
    )
    
    if viz_type == "Distribution Analysis":
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("#### Energy Consumption Distribution")
            fig, ax = plt.subplots()
            sns.histplot(df_filtered['EnergyConsumption'], kde=True, ax=ax, color='#667eea', bins=50)
            ax.set_title('Distribution of Energy Consumption')
            ax.set_xlabel('Energy Consumption (kWh)')
            ax.set_ylabel('Frequency')
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.write("#### Boxplot by Hour")
            fig, ax = plt.subplots(figsize=(10, 6))
            df_filtered.boxplot(column='EnergyConsumption', by='Hour', ax=ax, color=dict(boxes='#764ba2', whiskers='#764ba2'))
            ax.set_title('Energy Consumption by Hour')
            ax.set_xlabel('Hour')
            ax.set_ylabel('Energy Consumption (kWh)')
            plt.xticks(rotation=45)
            st.pyplot(fig)
            plt.close()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("#### Violin Plot: Weekday vs Weekend")
            fig, ax = plt.subplots()
            data_to_plot = [df_filtered[df_filtered['Weekend'] == 0]['EnergyConsumption'],
                           df_filtered[df_filtered['Weekend'] == 1]['EnergyConsumption']]
            parts = ax.violinplot(data_to_plot, positions=[0, 1], showmeans=True, showmedians=True)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['Weekday', 'Weekend'])
            ax.set_title('Energy Consumption: Weekday vs Weekend')
            ax.set_ylabel('Energy Consumption (kWh)')
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.write("#### Distribution by Time of Day")
            fig, ax = plt.subplots()
            order = ['Night', 'Morning', 'Afternoon', 'Evening']
            sns.boxplot(data=df_filtered, x='TimeOfDay', y='EnergyConsumption', order=order, ax=ax, palette='Set2')
            ax.set_title('Energy Consumption by Time of Day')
            ax.set_xlabel('Time of Day')
            ax.set_ylabel('Energy Consumption (kWh)')
            plt.xticks(rotation=45)
            st.pyplot(fig)
            plt.close()
    
    elif viz_type == "Time Series Analysis":
        # Time series granularity
        granularity = st.selectbox("Select Time Granularity", ["Hour", "Day", "Month"])
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if granularity == "Hour":
            ts_data = df_filtered.groupby('Hour')['EnergyConsumption'].mean()
            ax.plot(ts_data.index, ts_data.values, marker='o', linewidth=2, color='#667eea', markersize=8)
            ax.set_xlabel('Hour')
            ax.set_ylabel('Avg Energy Consumption (kWh)')
            ax.set_title('Average Energy Consumption by Hour')
            ax.grid(True, alpha=0.3)
            
        elif granularity == "Day":
            ts_data = df_filtered.groupby(df_filtered['Timestamp'].dt.date)['EnergyConsumption'].mean()
            ax.plot(range(len(ts_data)), ts_data.values, marker='o', linewidth=2, color='#667eea', markersize=4)
            ax.set_xlabel('Day')
            ax.set_ylabel('Avg Energy Consumption (kWh)')
            ax.set_title('Daily Average Energy Consumption')
            ax.grid(True, alpha=0.3)
            # Reduce x-ticks for better readability
            step = max(1, len(ts_data) // 20)
            ax.set_xticks(range(0, len(ts_data), step))
            
        else:
            ts_data = df_filtered.groupby('Month')['EnergyConsumption'].mean()
            ax.bar(ts_data.index, ts_data.values, color='#667eea', edgecolor='black')
            ax.set_xlabel('Month')
            ax.set_ylabel('Avg Energy Consumption (kWh)')
            ax.set_title('Monthly Average Energy Consumption')
            ax.grid(True, alpha=0.3, axis='y')
        
        st.pyplot(fig)
        plt.close()
        
        # Multi-variable time series
        st.write("#### Multi-Variable Time Series")
        vars_to_plot = st.multiselect(
            "Select variables to plot",
            ["EnergyConsumption", "Temperature", "Humidity", "RenewableEnergy"],
            default=["EnergyConsumption", "Temperature"]
        )
        
        if vars_to_plot:
            fig, ax1 = plt.subplots(figsize=(14, 6))
            
            # Sample data for better performance (take every 10th point)
            sample_data = df_filtered.iloc[::10] if len(df_filtered) > 1000 else df_filtered
            
            color1 = '#667eea'
            color2 = '#764ba2'
            
            for i, var in enumerate(vars_to_plot):
                if var == "EnergyConsumption":
                    ax1.plot(sample_data['Timestamp'], sample_data[var], color=color1, linewidth=2, label=var)
                    ax1.set_ylabel('Energy Consumption (kWh)', color=color1)
                    ax1.tick_params(axis='y', labelcolor=color1)
                else:
                    if i == 1:
                        ax2 = ax1.twinx()
                        ax2.plot(sample_data['Timestamp'], sample_data[var], color=color2, linewidth=2, linestyle='--', label=var)
                        ax2.set_ylabel(var, color=color2)
                        ax2.tick_params(axis='y', labelcolor=color2)
                    else:
                        ax1.plot(sample_data['Timestamp'], sample_data[var], linewidth=2, linestyle='--', label=var)
            
            ax1.set_xlabel('Timestamp')
            ax1.set_title('Time Series Comparison')
            ax1.grid(True, alpha=0.3)
            
            # Combine legends
            lines1, labels1 = ax1.get_legend_handles_labels()
            if 'ax2' in locals():
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            else:
                ax1.legend(loc='upper left')
            
            plt.xticks(rotation=45)
            st.pyplot(fig)
            plt.close()
    
    elif viz_type == "Correlation Analysis":
        # Correlation heatmap
        numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns
        corr_matrix = df_filtered[numeric_cols].corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=0.5, ax=ax, 
                   cbar_kws={"shrink": 0.8})
        ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        st.pyplot(fig)
        plt.close()
        
        # Scatter plots for selected features
        st.write("#### Scatter Plot Analysis")
        selected_features = st.multiselect(
            "Select two features for scatter plot",
            numeric_cols.tolist(),
            default=[numeric_cols[0], numeric_cols[-1]] if len(numeric_cols) > 1 else numeric_cols[:2].tolist()
        )
        
        if len(selected_features) >= 2:
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots()
                scatter = ax.scatter(df_filtered[selected_features[0]], 
                                   df_filtered['EnergyConsumption'],
                                   c=df_filtered['Weekend'], cmap='viridis', alpha=0.6)
                ax.set_xlabel(selected_features[0])
                ax.set_ylabel('Energy Consumption (kWh)')
                ax.set_title(f'{selected_features[0]} vs Energy Consumption')
                ax.grid(True, alpha=0.3)
                # Add legend for weekend
                legend_elements = [Patch(facecolor='#1f77b4', label='Weekday'),
                                 Patch(facecolor='#ff7f0e', label='Weekend')]
                ax.legend(handles=legend_elements)
                st.pyplot(fig)
                plt.close()
            
            with col2:
                fig, ax = plt.subplots()
                scatter = ax.scatter(df_filtered[selected_features[1]], 
                                   df_filtered['EnergyConsumption'],
                                   c=df_filtered['Weekend'], cmap='viridis', alpha=0.6)
                ax.set_xlabel(selected_features[1])
                ax.set_ylabel('Energy Consumption (kWh)')
                ax.set_title(f'{selected_features[1]} vs Energy Consumption')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close()
    
    else:  # Comparative Analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("#### HVAC Usage Impact")
            fig, ax = plt.subplots()
            sns.boxplot(data=df_filtered, x='HVACUsage', y='EnergyConsumption', ax=ax, palette=['#667eea', '#764ba2'])
            ax.set_title('Energy Consumption by HVAC Status')
            ax.set_xlabel('HVAC Usage')
            ax.set_ylabel('Energy Consumption (kWh)')
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.write("#### Lighting Usage Impact")
            fig, ax = plt.subplots()
            sns.boxplot(data=df_filtered, x='LightingUsage', y='EnergyConsumption', ax=ax, palette=['#667eea', '#764ba2'])
            ax.set_title('Energy Consumption by Lighting Status')
            ax.set_xlabel('Lighting Usage')
            ax.set_ylabel('Energy Consumption (kWh)')
            st.pyplot(fig)
            plt.close()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("#### Holiday Impact")
            fig, ax = plt.subplots()
            sns.boxplot(data=df_filtered, x='Holiday', y='EnergyConsumption', ax=ax, palette=['#667eea', '#764ba2'])
            ax.set_title('Energy Consumption: Holiday vs Regular')
            ax.set_xlabel('Holiday')
            ax.set_ylabel('Energy Consumption (kWh)')
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.write("#### Day of Week Impact")
            fig, ax = plt.subplots(figsize=(10, 6))
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            sns.boxplot(data=df_filtered, x='DayOfWeek', y='EnergyConsumption', 
                       order=day_order, ax=ax, palette='Set2')
            ax.set_title('Energy Consumption by Day of Week')
            ax.set_xlabel('Day of Week')
            ax.set_ylabel('Energy Consumption (kWh)')
            plt.xticks(rotation=45)
            st.pyplot(fig)
            plt.close()

# -----------------------------
# TAB 3: PREDICTION MODEL
# -----------------------------
with tab3:
    st.subheader("🤖 Energy Consumption Prediction")
    
    st.markdown("""
    This section uses Machine Learning to predict energy consumption based on various building parameters.
    You can either train a custom model or use the pre-trained model.
    """)
    
    # Model selection
    model_type = st.selectbox(
        "Select Model Type",
        ["Linear Regression", "Random Forest Regressor"]
    )
    
    # Prepare data
    df_model = df.copy()
    df_model = df_model.drop('Timestamp', axis=1)
    
    # Encode categorical variables
    le_dict = {}
    categorical_cols = ['HVACUsage', 'LightingUsage', 'DayOfWeek', 'Holiday', 'TimeOfDay']
    
    for col in categorical_cols:
        if col in df_model.columns:
            le_dict[col] = LabelEncoder()
            df_model[col] = le_dict[col].fit_transform(df_model[col])
    
    X = df_model.drop('EnergyConsumption', axis=1)
    y = df_model['EnergyConsumption']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    with st.spinner("Training model..."):
        if model_type == "Linear Regression":
            model = LinearRegression()
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
    
    # Display metrics
    st.success("✅ Model Training Complete!")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("R² Score", f"{r2:.3f}")
    with col2:
        st.metric("RMSE", f"{rmse:.2f} kWh")
    with col3:
        st.metric("MAE", f"{mae:.2f} kWh")
    with col4:
        st.metric("MSE", f"{mse:.2f}")
    
    # Actual vs Predicted plot
    st.write("#### Model Performance: Actual vs Predicted")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_test, y_pred, alpha=0.5, color='#667eea')
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
    ax.set_xlabel('Actual Consumption (kWh)')
    ax.set_ylabel('Predicted Consumption (kWh)')
    ax.set_title('Actual vs Predicted Energy Consumption')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close()
    
    # Feature importance (for Random Forest)
    if model_type == "Random Forest Regressor":
        st.write("#### Feature Importance")
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=True)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        bars = ax.barh(feature_importance['Feature'], feature_importance['Importance'], color='#667eea')
        ax.set_xlabel('Importance')
        ax.set_ylabel('Feature')
        ax.set_title('Feature Importance Analysis')
        ax.grid(True, alpha=0.3, axis='x')
        st.pyplot(fig)
        plt.close()
    
    # Custom prediction
    st.write("---")
    st.write("#### Make a Custom Prediction")
    st.markdown("Adjust the parameters below to predict energy consumption:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        temperature = st.slider("Temperature (°C)", 15.0, 35.0, 25.0, key="temp_slider")
        humidity = st.slider("Humidity (%)", 20.0, 80.0, 50.0, key="hum_slider")
        square_footage = st.slider("Square Footage (sq ft)", 1000.0, 2000.0, 1500.0, key="sqft_slider")
        occupancy = st.slider("Occupancy (people)", 0, 10, 5, key="occ_slider")
    
    with col2:
        hvac = st.selectbox("HVAC Usage", ["On", "Off"], key="hvac_select")
        lighting = st.selectbox("Lighting Usage", ["On", "Off"], key="light_select")
        renewable = st.slider("Renewable Energy (kWh)", 0.0, 30.0, 15.0, key="ren_slider")
        holiday = st.selectbox("Holiday", ["No", "Yes"], key="holiday_select")
        day_of_week = st.selectbox("Day of Week", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], key="dow_select")
        hour = st.slider("Hour of Day", 0, 23, 12, key="hour_slider")
    
    # Prepare input for prediction
    if st.button("🔮 Predict Energy Consumption", type="primary"):
        input_dict = {
            'Temperature': temperature,
            'Humidity': humidity,
            'SquareFootage': square_footage,
            'Occupancy': occupancy,
            'HVACUsage': 1 if hvac == "On" else 0,
            'LightingUsage': 1 if lighting == "On" else 0,
            'RenewableEnergy': renewable,
            'DayOfWeek': day_of_week,
            'Holiday': 1 if holiday == "Yes" else 0,
            'Hour': hour,
            'Day': 1,
            'Month': 1,
            'Year': 2022,
            'DayOfWeekNum': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'].index(day_of_week),
            'Weekend': 1 if day_of_week in ['Saturday', 'Sunday'] else 0,
            'TimeOfDay': 'Morning' if 6 <= hour < 12 else 'Afternoon' if 12 <= hour < 18 else 'Evening' if 18 <= hour < 24 else 'Night'
        }
        
        # Encode TimeOfDay
        time_map = {'Night': 0, 'Morning': 1, 'Afternoon': 2, 'Evening': 3}
        input_dict['TimeOfDay'] = time_map[input_dict['TimeOfDay']]
        
        # Encode DayOfWeek
        input_dict['DayOfWeek'] = le_dict['DayOfWeek'].transform([input_dict['DayOfWeek']])[0]
        
        input_df = pd.DataFrame([input_dict])
        input_df = input_df.reindex(columns=X.columns, fill_value=0)
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)
        
        st.balloons()
        st.success(f"### ⚡ Predicted Energy Consumption: {prediction[0]:.2f} kWh")

# -----------------------------
# TAB 4: INSIGHTS
# -----------------------------
with tab4:
    st.subheader("📈 Key Insights & Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔍 Top Insights")
        
        # Calculate key insights
        avg_hvac_on = df_filtered[df_filtered['HVACUsage'] == 'On']['EnergyConsumption'].mean()
        avg_hvac_off = df_filtered[df_filtered['HVACUsage'] == 'Off']['EnergyConsumption'].mean()
        
        avg_lighting_on = df_filtered[df_filtered['LightingUsage'] == 'On']['EnergyConsumption'].mean()
        avg_lighting_off = df_filtered[df_filtered['LightingUsage'] == 'Off']['EnergyConsumption'].mean()
        
        peak_hour = df_filtered.groupby('Hour')['EnergyConsumption'].mean().idxmax()
        low_hour = df_filtered.groupby('Hour')['EnergyConsumption'].mean().idxmin()
        
        st.info(f"💡 **HVAC Impact:** Using HVAC increases consumption by {(avg_hvac_on - avg_hvac_off):.1f} kWh on average")
        st.info(f"💡 **Lighting Impact:** Using lighting increases consumption by {(avg_lighting_on - avg_lighting_off):.1f} kWh on average")
        st.info(f"💡 **Peak Hour:** Highest consumption occurs at {peak_hour}:00")
        st.info(f"💡 **Best Hour:** Lowest consumption occurs at {low_hour}:00")
        
        # Correlation with temperature
        temp_corr = df_filtered['Temperature'].corr(df_filtered['EnergyConsumption'])
        st.info(f"💡 **Temperature Correlation:** {temp_corr:.2f} correlation between temperature and energy consumption")
    
    with col2:
        st.markdown("#### 💡 Recommendations")
        
        st.success("""
        **1. Optimize HVAC Usage**
        - Schedule HVAC during occupied hours only
        - Consider programmable thermostats
        - Potential savings: 15-20%
        
        **2. Lighting Efficiency**
        - Implement motion sensors
        - Use LED lighting where possible
        - Maximize natural lighting
        
        **3. Peak Load Management**
        - Shift non-critical operations to off-peak hours
        - Consider energy storage systems
        - Implement demand response strategies
        
        **4. Renewable Energy**
        - Increase renewable energy integration
        - Consider solar panel installation
        - Battery storage for peak shaving
        
        **5. Monitoring & Automation**
        - Implement real-time monitoring
        - Use predictive maintenance
        - Automate based on occupancy
        """)
    
    # Hourly consumption pattern
    st.markdown("#### 📊 Hourly Consumption Pattern")
    hourly_pattern = df_filtered.groupby('Hour')['EnergyConsumption'].agg(['mean', 'std']).reset_index()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(hourly_pattern['Hour'], hourly_pattern['mean'], marker='o', linewidth=2, color='#667eea', markersize=8, label='Average Consumption')
    ax.fill_between(hourly_pattern['Hour'], 
                    hourly_pattern['mean'] - hourly_pattern['std'], 
                    hourly_pattern['mean'] + hourly_pattern['std'], 
                    alpha=0.3, color='#667eea', label='±1 Std Dev')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Energy Consumption (kWh)')
    ax.set_title('Hourly Energy Consumption Pattern with Variability')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close()

st.markdown("---")
st.markdown("### 🎯 Summary")
st.info("This dashboard provides comprehensive analysis of building energy consumption patterns. Use the filters to explore specific scenarios and leverage the prediction model for forecasting energy needs.")