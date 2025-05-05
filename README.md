🌫️ Air Quality Index Contributor
A Streamlit-based web application to estimate AQI contribution from construction sites based on raw materials used.

📌 Project Description
The Air Quality Index Contributor project is designed to assess and quantify the impact of construction activities on air pollution. By calculating the emissions of seven major pollutants from the raw materials used at a construction site, this application estimates the contribution of these activities to the Air Quality Index (AQI).

The project is built using real-world data from Delhi and integrates machine learning to analyze pollutant concentrations. A trained linear regression model is employed to predict AQI values based on the calculated pollutant concentrations.

Additionally, the project features a calculator that converts the weight or quantity of raw materials used in construction (in kilograms) into pollutant concentrations (in µg/m³). This conversion helps in accurately determining the level of pollutants released into the air, facilitating better environmental management and decision-making during construction activities.
🔍 Key Features
📦 Input Raw Materials used in a construction site.

🌫️ Estimate Emissions for 7 major air pollutants:
PM2.5
PM10
NO₂
SO₂
CO
O₃
NH₃
📊 Trained ML Model using Delhi AQI data (1 month) to predict AQI from pollutant concentrations.
🖥️ Streamlit Integration for a user-friendly interactive web interface.
🧾 Result Output: Final AQI score and its category (e.g., Moderate, Poor, Severe).


🧠 Tech Stack
Python
Pandas & NumPy
Scikit-learn – Linear Regression model
Streamlit – Web application interface
Matplotlib / Seaborn – (Optional) for data visualization
CSV Dataset – Delhi city AQI data

You can check the app by searching this--> https://airqualityindexcontributor.streamlit.app/
