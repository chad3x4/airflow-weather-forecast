import streamlit as st
from datetime import date
import pandas as pd
import math

weather = pd.read_csv("airflow/weather_predict.csv")
weather_image = {
    'Overcast clouds': 'https://www.rochesterfirst.com/wp-content/uploads/sites/66/2021/04/storm-466677_1920.jpg',
    'Broken clouds': 'https://images-wixmp-ed30a86b8c4ca887773594c2.wixmp.com/f/625a747a-061b-477d-958f-a0d6cea9e4cb/dax9bd4-dd0da73d-5b6e-415c-b05e-19471f366e5a.jpg/v1/fill/w_1024,h_768,q_75,strp/broken_clouds_by_kevintheman_dax9bd4-fullview.jpg?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1cm46YXBwOjdlMGQxODg5ODIyNjQzNzNhNWYwZDQxNWVhMGQyNmUwIiwiaXNzIjoidXJuOmFwcDo3ZTBkMTg4OTgyMjY0MzczYTVmMGQ0MTVlYTBkMjZlMCIsIm9iaiI6W1t7ImhlaWdodCI6Ijw9NzY4IiwicGF0aCI6IlwvZlwvNjI1YTc0N2EtMDYxYi00NzdkLTk1OGYtYTBkNmNlYTllNGNiXC9kYXg5YmQ0LWRkMGRhNzNkLTViNmUtNDE1Yy1iMDVlLTE5NDcxZjM2NmU1YS5qcGciLCJ3aWR0aCI6Ijw9MTAyNCJ9XV0sImF1ZCI6WyJ1cm46c2VydmljZTppbWFnZS5vcGVyYXRpb25zIl19.2HBtScMyydNDUe606gk2Jd8RHs6iM-76feSI7Dc3sLw',
    'Scattered clouds': 'https://www.shutterstock.com/shutterstock/photos/1755527540/display_1500/stock-photo-scattered-clouds-on-a-blue-sky-day-1755527540.jpg',
    'Light rain': 'https://s7d2.scene7.com/is/image/TWCNews/0622_n13_light_rain?wid=1250&hei=703&$wide-bg$',
    'Few clouds': 'https://media.istockphoto.com/id/171225633/vi/anh/khung-c%E1%BA%A3nh-m%C3%A0u-xanh-th%E1%BA%B3m-trong-m%E1%BB%99t-ng%C3%A0y-m%C3%A2y-nh%E1%BA%B9.jpg?s=1024x1024&w=is&k=20&c=F2dx-XOSmYWy8cF3rjS-moZkk8RvVIASE6ybGyts0Fo=',
    'Clear sky': 'https://lh3.googleusercontent.com/CnHg3skxcIhFKh5oE_ZV61x-a-tqWKIWC04a4hWkmQymuBRGlp3Kgnr_d3bEj-jgvPZAM1kh4nkpALUr0bDaUJdzPQ=s1280-w1280-h800',
    'Fog': 'https://wpcdn.us-east-1.vip.tn-cloud.net/www.wmdt.com/content/uploads/2021/01/fog-1.jpg',
    'Moderate rain': 'https://myrepublica.nagariknetwork.com/uploads/media/rain_20210802140558.jpg',
    'Thunderstorm with heavy rain': 'https://weatherwatch-assets.s3.ap-southeast-2.amazonaws.com/wp-content/uploads/2020/02/23033607/picture-17030.jpg',
    'Heavy rain': 'https://weatherwatch-assets.s3.ap-southeast-2.amazonaws.com/wp-content/uploads/2020/02/23033607/picture-17030.jpg',
    'Haze': 'https://cdn-assets-eu.frontify.com/s3/frontify-enterprise-files-eu/eyJvYXV0aCI6eyJjbGllbnRfaWQiOiJmcm9udGlmeS1maW5kZXIifSwicGF0aCI6ImloaC1oZWFsdGhjYXJlLWJlcmhhZFwvYWNjb3VudHNcL2MzXC80MDAwNjI0XC9wcm9qZWN0c1wvMjA5XC9hc3NldHNcLzIwXC8zODEwMFwvOWQyMzBiNjE5MDZlZjdmMzFiYTY5NDE3YjY5ZmVmODAtMTY1ODMwMTAzMC5qcGcifQ:ihh-healthcare-berhad:hfyAKfkN0RWZDGy9EulsZ31Qx6CPCnYpXzpI_ZaHDG0?format=webp'
}

st.set_page_config(
    page_title="Weather predict page",
    page_icon=":cloud:",
    layout="wide",

)

st.sidebar.success("Select a demo above.")

st.write("# Welcome to weather predict app! 👋")

st.markdown(
    """
    ## This app will predict the next 24h weather status.  
    """
)
img_path = 'imgs/picture.png'

st.markdown(
    f"""
        ### Weather prediction in {date.today()}
    """
)

columns = st.columns(12)

for i, col in enumerate(columns):
    index = i+7
    col.markdown(
        f"""
        <style>
        .image-column {{
            background-image: url({weather_image[weather.loc[i, 'weather_main']]});
            background-size: cover;
            background-position: center
            width: 400px;
            height: 100px;
            border: 0px solid #ccc;
        }}
        .box {{
            border: 1px solid #ccc;
            border-radius: 5px;
            
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
    col.markdown(
        f"""
        <div>
            {index}h
        </div>
        <div class="box">
            <div class="image-column">
            </div>
            <div style = "margin: 10px;">        
                <div>
                    {math.floor(weather.loc[i, 'app_temp'])}'C
                </div>
                <div>
                    {weather.loc[i, 'weather_main']}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


columns = st.columns(12)

for i, col in enumerate(columns):
    if i+19 >= 24: index = i-5 
    else: index = i+19

    col.markdown(
        f"""
        <style>
        .image-column {{
            background-image: url({weather_image[weather.loc[i+12, 'weather_main']]});
            background-size: cover;
            background-position: center
            width: 400px;
            height: 100px;
            border: 0px solid #ccc;
        }}
        .box {{
            border: 1px solid #ccc;
            border-radius: 5px;
            
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
    col.markdown(
        
        f"""
        <div>
            {index}h
        </div>
        <div class="box">
            <div class="image-column">
            </div>
            <div style = "margin: 10px;">        
                <div>
                    {math.floor(weather.loc[i+12, 'app_temp'])}'C
                </div>
                <div>
                    {weather.loc[i+12, 'weather_main']}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

