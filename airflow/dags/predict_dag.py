from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

def preprocess_data():
    print("Process Data!!!")


def predict():
    import tensorflow as tf
    import joblib
    from sklearn.preprocessing import StandardScaler

    df = pd.read_csv('/opt/airflow/data/weather_hanoi.csv')
    df = df.drop(df.columns[0], axis=1)
 
    #lọc các trường k sử dụng
    df_test = df.drop(['timestamp_local', 'ts', 'name', 'pod', 'weather_code', 'revision_status', 'weather_description', 'lat', 'lon', 'snow', 'precip', 'vis'], axis=1)

    #Chuyển về tham số sin_cos cho các đại lượng góc
    azimuth = df_test.pop('azimuth')
    elev_angle = df_test.pop('elev_angle')
    wind_dir = df_test.pop('wind_dir')

    df_test['azimuth_sin'] = np.sin(azimuth * np.pi / 180)
    df_test['azimuth_cos'] = np.cos(azimuth * np.pi / 180)
    df_test['elev_angle_sin'] = np.sin(elev_angle * np.pi / 180)
    df_test['elev_angle_cos'] = np.cos(elev_angle * np.pi / 180)
    df_test['wind_dir_sin'] = np.sin(wind_dir * np.pi / 180)
    df_test['wind_dir_cos'] = np.cos(wind_dir * np.pi / 180)

    

    #chuyển thời gian utc về dạng unix
    date_time = pd.to_datetime(df_test.pop('timestamp_utc'), format='%Y-%m-%dT%H:%M:%S')
    timestamp_s = date_time.map(pd.Timestamp.timestamp)

    #Chuyển thời gian unix về dạng sin_cos tượng trưng cho khoảng thời gian trong 1 ngày
    day = 24*60*60
    df_test['day_sin'] = np.sin(timestamp_s * (2 * np.pi / day))
    df_test['day_cos'] = np.cos(timestamp_s * (2 * np.pi / day))

    train_df = df_test[0:int(len(df_test)*0.8)]
    val_df = df_test[int(len(df_test)*0.8):]

    #Chuẩn hóa z_score cho tập train, áp dụng lên tâp val và test
    train_mean = train_df.mean()
    train_std = train_df.std()
    train_df = (train_df - train_mean) / train_std
    val_df = (val_df - train_mean) / train_std
    num_features = 21

    OUT_STEPS = 24
    multi_lstm_model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.LSTM(64, return_sequences=False),
    tf.keras.layers.Dense(OUT_STEPS * num_features, activation='linear', kernel_initializer='glorot_uniform'),
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
    ])

    dummy_input = tf.zeros((1, OUT_STEPS, num_features))
    _ = multi_lstm_model(dummy_input)

    multi_lstm_model.load_weights('/opt/airflow/models/model_lstm_weights.h5')
    model_rf = joblib.load('/opt/airflow/models/random_forest_model.joblib')

    print(df_test.columns)

    pred = multi_lstm_model.predict(np.expand_dims(df_test[-72:], axis=0))
    pred_unproc = [pred[0,i] * train_std + train_mean for i in range(24)]
    scaler = StandardScaler()
    pred_unproc_normalize = scaler.fit_transform(pred_unproc)
    weather = model_rf.predict(pred_unproc_normalize)

    pred_unproc = pd.DataFrame(pred_unproc)
    pred_unproc['weather_main'] = weather
    pred_unproc.to_csv("weather_predict.csv")


dag = DAG(
    'predict_dag',
    default_args={'start_date': days_ago(1)},
    schedule_interval=timedelta(days=1),  # Run each one day
    catchup=False
)

preprocess_data_task = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data,
    dag=dag
)

predict_task = PythonOperator(
    task_id='predict_data',
    python_callable=predict,
    dag=dag
)

# Set the dependencies between the tasks
preprocess_data_task >> predict_task