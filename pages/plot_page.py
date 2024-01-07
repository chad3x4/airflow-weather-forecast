import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Weather predict page",
    page_icon=":cloud:",
    layout="wide",

)
weather_df = pd.read_csv("airflow/data/weather_hanoi.csv")
weather_df["Date"]=pd.to_datetime(weather_df['timestamp_local'])

selected_start_date = st.date_input("Select start date")
selected_end_date = st.date_input("Select end date")

def visualize_weather_between_days(weather_df, start_day, end_day, features):

    filtered_data = weather_df[(weather_df['Date'].dt.date >= start_day) & (weather_df['Date'].dt.date <= end_day)]

    num_features = len(features)

    plt.figure(figsize=(12, 8))

    for i, feature in enumerate(features, start=1):
        plt.subplot(num_features, 1, i)
        plt.plot(filtered_data['Date'], filtered_data[feature], label=feature)
        plt.xlabel('Date')
        plt.ylabel(feature)
        plt.title('{} biến thiên giữa {} và {}'.format(feature, start_day, end_day))
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)

    plt.tight_layout()
    st.pyplot(plt) 

st.markdown(
    f"""
    ## Change from {selected_start_date} to {selected_end_date}
    """
)

features_to_visualize = ['temp', 'rh', 'precip']  # Bo sung features can thiet


visualize_weather_between_days(weather_df, selected_start_date, selected_end_date, features_to_visualize)