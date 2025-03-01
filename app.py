import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Title
st.title("🌍 Air Quality PM2.5 Level Prediction")

# Sidebar for easy Navigation
st.sidebar.image("image.png", use_column_width=True)
st.sidebar.title("Fasten Your Seatbelts! Let’s Navigate Through the Project")
option = st.sidebar.radio("🧭 Choose Your Path:", ["About 💡", "Let Predict! 📈"])



# Dataset Loading
@st.cache_data
def load_data():
    df = pd.read_csv("air-quality-india.csv")
    return df


df = load_data()
df_pure = pd.read_csv("updated_air_quality.csv")

# function to determine the AQI Level Category
def get_aqi_category(level):
    if level<=50:
        return "Good"
    elif level<=100:
        return "Moderate"
    elif level<=150:
        return "Unhealthy for Sensitive Groups"
    elif level<=200:
        return "Unhealthy"
    elif level<=300:
        return "Very Unhealthy"
    else:
        return "Hazardous"

# About Section
if option == "About 💡":
    st.header("📌 About The Project")
    st.markdown("""
    Leveraging Machine Learning techniques, this project predicts PM2.5 levels based on key environmental factors such as Year, Month, Day, and Hour. The dataset, sourced from air quality monitoring stations across India, ensures a data-driven approach to air pollution analysis.
    To enhance predictive accuracy, advanced methodologies such as feature engineering and feature selection are employed, allowing for optimal data preprocessing and meaningful pattern extraction. PM2.5, a fine particulate matter in the air, poses severe health risks, making its prediction crucial for environmental monitoring and public health initiatives.
    By integrating these techniques, the model provides precise forecasts, aiding in proactive pollution control and strategic decision-making. 
    """)

    st.subheader("📊 What’s Inside the Data Vault?")
    st.markdown("""
    - **Source**: Air quality dataset of India 
    - **Features in Dataset**: Timestamp, Year, Month, Day, Hour, PM2.5  
    - **Features added**: Time of Day, Season, AQI Category
    - **Features Used**: Year, Month, Day, Hour  
    - **Target Variable**: PM2.5 concentration (µg/m³)  
    - **Goal**: Predict PM2.5 levels to help in pollution analysis and control  
    """)



    # Showing the Original(Initial) Dataset
    st.write("Original Dataset")
    st.dataframe(df)

    # Showing the Dataset after doing modifications
    st.write('Modified Dataset')
    st.dataframe(df_pure)

    st.subheader("📈 Data Visualization")

    # PM2.5 Distribution
    st.write("### PM2.5 Level Distribution")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["PM2.5"], bins=30, kde=True, color="blue", ax=ax)
    st.pyplot(fig)

    # Monthly PM2.5 Trend
    st.write("### Average PM2.5 Levels by Month")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(x=df_pure['Month'], y=df_pure['PM2.5'], marker="o", ax=ax,color='green')
    ax.set_xlabel("Month")
    ax.set_ylabel("PM2.5 Level")
    st.pyplot(fig)

    # PM2.5 Levels based on Season
    st.write("### PM2.5 Levels based on Season")
    fig, ax = plt.subplots(figsize=(8,5))
    sns.barplot(data=df_pure,x='Season',y='PM2.5',ax=ax,color='red')
    ax.set_xlabel("Season")
    ax.set_ylabel("PM2.5 Level")
    st.pyplot(fig)

    # Area Chart (Made using Power BI)
    st.write("### Day Wise PM2.5 Average")
    st.image("area_chart.png",use_column_width=True)

    st.info("Move to the **Let's Predict** section from the sidebar to get PM2.5 predictions!")

# Coming to the Prediction Section
elif option == "Let Predict! 📈":
    st.header("🔮 PM2.5 Level Prediction")

    # Loading the Model
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)

    # Input Fields
    st.subheader("📥 Enter Required Data")
    year = st.number_input("Year", min_value=1990, max_value=3000, step=1)
    month = st.number_input("Month", min_value=1, max_value=12, step=1)
    day = st.number_input("Day", min_value=1, max_value=31, step=1)
    hour = st.number_input("Hour", min_value=0, max_value=23, step=1)

    # Predicting
    if st.button("🔍 Predict PM2.5 Level"):
        input_data = np.array([[year, month, day, hour]])

        # Prediction
        prediction = model.predict(input_data)

        # Showing the Result
        st.success(f"Predicted PM2.5 Level: {prediction[0]:.2f} µg/m³")
        cat_res = get_aqi_category(prediction)
        st.success(f"{cat_res} Level")
