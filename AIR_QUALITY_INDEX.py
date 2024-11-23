import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Set page config
st.set_page_config(
    page_title="Industrial Pollution & AQI Calculator",
    page_icon="🌍",
    layout="wide"
)

# Constants (from previous code)
DENSITIES = {
    'cement': 1440,
    'steel': 7850,
    'asphalt': 2350,
    'diesel': 850,
    'lpg': 510,
    'gasoline': 720,
    'coal': 1400,
    'iron': 7870
}

EMISSION_FACTORS = {
    'cement': {
        'sulfur': 0.02,
        'PM10': 0.3,
        'PM2': 0.15,
        'NO': 0.1,
        'NO2': 0.05,
        'CO': 0.1
    },
    'steel': {
        'sulfur': 0.05,
        'PM10': 0.8,
        'PM2': 0.4,
        'NO': 0.2,
        'NO2': 0.1,
        'CO': 0.15
    },
    'asphalt': {
        'sulfur': 0.05,
        'PM10': 0.5,
        'PM2': 0.25,
        'NO': 0.1,
        'NO2': 0.05,
        'CO': 0.2
    },
    'diesel': {
        'sulfur': 0.03,
        'PM10': 0.8,
        'PM2': 0.4,
        'NO': 0.3,
        'NO2': 0.15,
        'CO': 0.3
    },
    'lpg': {
        'sulfur': 0,
        'PM10': 0.02,
        'PM2': 0.4,
        'NO': 0.08,
        'NO2': 0.04,
        'CO': 0.08
    },
    'gasoline': {
        'sulfur': 0,
        'PM10': 0.08,
        'PM2': 0.04,
        'NO': 0.2,
        'NO2': 0.1,
        'CO': 0.8
    },
    'coal': {
        'sulfur': 0.08,
        'PM10': 1.5,
        'PM2': 0.8,
        'NO': 0.4,
        'NO2': 0.2,
        'CO': 0.2
    },
    'iron': {
        'sulfur': 0.0016,
        'PM10': 0.0063,
        'PM2': 0.0029,
        'NO': 0.0012,
        'NO2': 0.0005,
        'CO': 0.0042
    }
}

@st.cache_data
def load_and_prepare_data():
    """Load and prepare the AQI dataset"""
    df = pd.read_csv("city_day.csv")
    df.drop(['City', 'Date', 'Xylene', 'Toluene', 'Benzene', 'AQI_Bucket', 'NOx', 'O3', 'NH3'], axis=1, inplace=True)
    df_new = df.dropna()
    return df_new

@st.cache_resource
def train_model(df):
    """Train the AQI prediction model"""
    X = df[['PM2.5', 'PM10', 'NO', 'NO2', 'CO', 'SO2']]
    y = df[['AQI']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Calculate R2 score
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    
    return model, r2

def calculate_emissions(material, quantity_kg):
    """Calculate emissions for a given material and quantity."""
    if quantity_kg == 0:
        return 0, {pollutant: 0 for pollutant in EMISSION_FACTORS[material]}
    
    density = DENSITIES[material]  # kg/m³
    volume = quantity_kg / density  # Convert to m³
    
    emissions = {}
    factors = EMISSION_FACTORS[material]
    
    for pollutant, factor in factors.items():
        if pollutant == 'CO':  # CO in mg
            emissions[pollutant] = volume * factor * 1000  # mg
        else:  # Other pollutants in µg
            emissions[pollutant] = volume * factor * 1_000_000  # µg
            
    return volume, emissions

def calculate_concentrations(total_emissions, ambient_volume):
    """Calculate pollutant concentrations based on ambient air volume."""
    if ambient_volume <= 0:
        return None  # Avoid division by zero
    
    concentrations = {
        'SO2': total_emissions['sulfur'] / ambient_volume,  # µg/m³
        'PM10': total_emissions['PM10'] / ambient_volume,   # µg/m³
        'PM2.5': total_emissions['PM2'] / ambient_volume,   # µg/m³
        'NO': total_emissions['NO'] / ambient_volume,       # µg/m³
        'NO2': total_emissions['NO2'] / ambient_volume,     # µg/m³
        'CO': (total_emissions['CO'] / ambient_volume) / 1000  # mg/m³
    }
    return concentrations

def main():
    st.title("🌍 Industrial Pollution & AQI Calculator")
    
    # Load data and train model
    try:
        df = load_and_prepare_data()
        model, r2_score = train_model(df)
        st.sidebar.success(f"Model R² Score: {r2_score:.4f}")
    except FileNotFoundError:
        st.error("Error: city_day.csv file not found. Please ensure the file is in the same directory as the application.")
        return
    
    # Create tabs for different functionalities
    tab1, tab2 = st.tabs(["Material Input", "Results & Predictions"])
    
    # Dictionary to store material quantities
    quantities = {}
    
    with tab1:
        st.markdown("""Enter the quantities of materials used (in kilograms) to calculate resulting pollutant 
        concentrations and predict the Air Quality Index (AQI).""")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Basic Materials")
            quantities['cement'] = st.number_input("Cement (kg)", min_value=0.0, value=0.0, step=100.0)
            quantities['steel'] = st.number_input("Steel (kg)", min_value=0.0, value=0.0, step=100.0)
            quantities['asphalt'] = st.number_input("Asphalt (kg)", min_value=0.0, value=0.0, step=100.0)
            quantities['iron'] = st.number_input("Iron (kg)", min_value=0.0, value=0.0, step=100.0)

        with col2:
            st.subheader("Fuels")
            quantities['diesel'] = st.number_input("Diesel (kg)", min_value=0.0, value=0.0, step=100.0)
            quantities['lpg'] = st.number_input("LPG (kg)", min_value=0.0, value=0.0, step=100.0)
            quantities['gasoline'] = st.number_input("Gasoline (kg)", min_value=0.0, value=0.0, step=100.0)
            quantities['coal'] = st.number_input("Coal (kg)", min_value=0.0, value=0.0, step=100.0)

        # Ambient air volume input
        ambient_volume = st.number_input(
            "Ambient Air Volume (m³):",
            min_value=0.1,
            value=1000.0,
            step=10.0
        )
    
    with tab2:
        if st.button("Calculate Pollutant Concentrations & Predict AQI"):
            total_volume = 0
            total_emissions = {
                'sulfur': 0, 'PM10': 0, 'PM2': 0,
                'NO': 0, 'NO2': 0, 'CO': 0
            }
            
            # Calculate emissions for each material
            for material, quantity in quantities.items():
                volume, emissions = calculate_emissions(material, quantity)
                total_volume += volume
                
                # Accumulate emissions
                for pollutant in emissions:
                    total_emissions[pollutant] += emissions[pollutant]
            
            # Calculate concentrations
            concentrations = calculate_concentrations(total_emissions, ambient_volume)
            
            if concentrations:
                st.subheader("Pollutant Concentrations (µg/m³ or mg/m³ for CO):")
                st.write(pd.DataFrame.from_dict(concentrations, orient='index', columns=['Concentration']))
                
                # Predict AQI
                input_data = np.array([[concentrations['PM2.5'], concentrations['PM10'], concentrations['NO'],
                                        concentrations['NO2'], concentrations['CO'], concentrations['SO2']]])
                predicted_aqi = model.predict(input_data)[0][0]
                st.success(f"Predicted AQI: {predicted_aqi:.2f}")
            else:
                st.error("Invalid ambient air volume. Please enter a positive value.")

if __name__ == "__main__":
    main()
