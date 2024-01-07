from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
import pandas as pd
import numpy as np



def preprocess_data():
    print("Start trainning!!!")


def train_model():
    import tensorflow as tf
    from sklearn.ensemble import RandomForestClassifier
    import joblib
    
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

    num_features = df_test.shape[1]
    weather_labels = df['weather_description']

    class WindowGenerator():
        def __init__(self, input_width, label_width, shift,
                    train_df=train_df, val_df=val_df,
                    label_columns=None):
            # Store the raw data.
            self.train_df = train_df
            self.val_df = val_df

            # Work out the label column indices.
            self.label_columns = label_columns
            if label_columns is not None:
                self.label_columns_indices = {name: i for i, name in
                                            enumerate(label_columns)}
            self.column_indices = {name: i for i, name in
                                    enumerate(train_df.columns)}

            # Work out the window parameters.
            self.input_width = input_width
            self.label_width = label_width
            self.shift = shift

            self.total_window_size = input_width + shift

            self.input_slice = slice(0, input_width)
            self.input_indices = np.arange(self.total_window_size)[self.input_slice]

            self.label_start = self.total_window_size - self.label_width
            self.labels_slice = slice(self.label_start, None)
            self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

        def __repr__(self):
            return '\n'.join([
                f'Total window size: {self.total_window_size}',
                f'Input indices: {self.input_indices}',
                f'Label indices: {self.label_indices}',
                f'Label column name(s): {self.label_columns}'])

    OUT_STEPS = 24
    wide_window = WindowGenerator(input_width=72, label_width=OUT_STEPS, shift=OUT_STEPS)

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    WindowGenerator.split_window = split_window

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32,)

        ds = ds.map(self.split_window)

        return ds

    WindowGenerator.make_dataset = make_dataset

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result

    WindowGenerator.train = train
    WindowGenerator.val = val
    WindowGenerator.example = example

    MAX_EPOCHS = 30
    class PrintLearningRateCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            current_lr = self.model.optimizer.learning_rate.numpy()
            print(f"Learning rate for epoch {epoch + 1}: {current_lr}")

    def compile_and_fit(model, window):
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=1e-3, decay_steps=65000)

        model.compile(loss=tf.losses.MeanSquaredError(),
                    optimizer=tf.optimizers.Adam(learning_rate=lr_schedule),
                    metrics=[tf.metrics.MeanAbsoluteError()])

        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        'model_lstm.keras',
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode="min",
        save_freq="epoch",
        initial_value_threshold=None
        )
        print_lr_callback = PrintLearningRateCallback()

        history = model.fit(window.train, epochs=MAX_EPOCHS,
                            validation_data=window.val,
                            callbacks=[checkpoint_callback, print_lr_callback])
        return history
    multi_lstm_model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.LSTM(64, return_sequences=False),
        tf.keras.layers.Dense(OUT_STEPS * num_features, activation='linear', kernel_initializer='glorot_uniform'),
        tf.keras.layers.Reshape([OUT_STEPS, num_features])
    ])

    history = compile_and_fit(multi_lstm_model, wide_window)

    #random_forest_classifier
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(train_df, weather_labels[0:int(len(weather_labels)*0.8)])

    joblib.dump(model, 'random_forest_model.joblib')


dag = DAG(
    'train_model',
    default_args={'start_date': days_ago(1)},
    schedule_interval=timedelta(days=7),
    catchup=False
)

preprocess_data_task = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data,
    dag=dag
)

train_model_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag
)

# Set the dependencies between the tasks
preprocess_data_task >> train_model_task