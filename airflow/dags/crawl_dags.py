from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
from datetime import datetime, timedelta
import requests
import pandas as pd
import numpy as np

api_key = '8e2e60b235374622baa14585b2d2657d'
lat = 21.0285
lon = 105.8542
start_date = '2023-12-20'
end_date = '2024-1-7'
dt = 1702566000
url = f'https://api.weatherbit.io/v2.0/history/hourly?key={api_key}&lat={lat}&lon={lon}&start_date={start_date}&end_date={end_date}'

def crawl_data():
    print('Crawl Data start!!!')
    df = pd.DataFrame(columns=['lat', 'lon', 'name', 'timestamp_local','timestamp_utc', 'app_temp', 'azimuth', 'clouds', 'dewpt', 'dhi', 'elev_angle', 'ghi', 'pod', 'precip', 'pres', 'revision_status', 'rh', 'slp', 'snow', 'solar_rad','temp','ts','uv','vis', 'weather_code', 'weather_description', 'wind_dir','wind_gust_spd','wind_spd'])
    weather_hanoi = pd.read_csv('/opt/airflow/data/weather_hanoi.csv')

    response = requests.get(url)
    data = response.json()
    list = data['data']

    for i in list:
        row = {'lat': lat,
            'lon': lon,
            'name': data['city_name'],
            'timestamp_local': i['timestamp_local'],
            'timestamp_utc': i['timestamp_utc'],
            'app_temp': i['app_temp'],
            'azimuth': i['azimuth'],
            'dewpt': i['dewpt'],
            'dhi': i['dhi'],
            'elev_angle': i['elev_angle'],
            'ghi': i['ghi'],
            'pod': i['pod'],
            'precip': i['precip'],
            'pres': i['pres'],
            'clouds': i['clouds'],
            'wind_spd': i['wind_spd'],
            'wind_dir': i['wind_dir'],
            'wind_gust_spd': i['wind_gust_spd'],
            'revision_status': i['revision_status'],
            'rh': i['rh'],
            'slp': i['slp'],
            'snow': i['snow'],
            'solar_rad': i['solar_rad'],
            'temp': i['temp'],
            'ts': i['ts'],
            'uv': i['uv'],
            'vis': i['vis'],
            'weather_code': i['weather']['code'],
            'weather_description': i['weather']['description'],
            }
        df.loc[len(df)] = row

    weather_hanoi = pd.concat([weather_hanoi, df], ignore_index=True)
    weather_hanoi = weather_hanoi.sort_values("timestamp_local", ascending=True)
    weather_hanoi.to_csv('/opt/airflow/data/weather_hanoi.csv')

def confirm_finished():
    print("Crawl Successfully!!!")

dag = DAG(
    'crawl_data',
    default_args={'start_date': days_ago(1)},
    schedule_interval=timedelta(days=1),
    catchup=False
)

preprocess_data_task = PythonOperator(
    task_id='crawl_data',
    python_callable=crawl_data,
    dag=dag
)

confirm_task = PythonOperator(
    task_id='confirm_finished',
    python_callable=confirm_finished,
    dag=dag
)

preprocess_data_task >> confirm_task