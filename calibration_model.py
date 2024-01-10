import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta

import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

import matplotlib; matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
import seaborn as sns; sns.set_theme(color_codes=True)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from matplotlib.animation import FuncAnimation

import re
import os
import time
import io
import argparse

def check_directory_exists(directory):
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory '{directory}' does not exist.")

def get_date(file_name):
    date_pattern = r'\d{8}'  # Matches 8 digits (YYYYMMDD)
    match = re.search(date_pattern, file_name)
    if match:
        date_str = match.group()
        return date_str
    else:
        return np.nan

def get_files(start_date, end_date):
    filename_list = []

    while start_date <= end_date:
        filename = f'QR_TAKEHOME_{str(start_date.strftime("%Y%m%d"))}.csv.parquet'
        filename_list.append(filename)
        start_date += timedelta(days=1)

    return filename_list

def get_data(start_date, end_date, directory='/Projects/ClearStreet/qr_takehome', period='morning'):
    
    check_directory_exists(directory)
    
    morning_df = pd.DataFrame()
    midday_df = pd.DataFrame()
    afternoon_df = pd.DataFrame()

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    filename_list = get_files(start_date, end_date)

    for filename in filename_list:
        filepath = os.path.join(directory, filename)
        if not os.path.exists(filepath):
            continue

        parquet_table = pq.read_table(filepath)
        df_datewise = parquet_table.to_pandas()
        df_datewise['date'] = pd.to_datetime(get_date(filename), format='%Y%m%d')

        morning_df = pd.concat([morning_df, df_datewise.loc[df_datewise['time'] <= 39600000].sort_values(by='time')], ignore_index=True)
        midday_df = pd.concat([midday_df, df_datewise.loc[(df_datewise['time'] >= 39601000) & (df_datewise['time'] <= 50400000)].sort_values(by='time')], ignore_index=True)
        afternoon_df = pd.concat([afternoon_df, df_datewise.loc[(df_datewise['time'] >= 50401000)].sort_values(by='time')], ignore_index=True)

    df = morning_df

    if period == 'midday':
        df = midday_df
    elif period == 'afternoon':
        df = afternoon_df

    df['time'] = pd.to_timedelta(df['time'], unit='ms')
    df['datetime'] = df['date'] + df['time']
    df['datetime'] = pd.to_datetime(df['datetime'])

    return df

def handle_nan(data, outlier_col='Y'):
    df = data.loc[data['Q1']>=0.99].loc[data['Q2']>=0.99]
    df.replace(999999, np.nan, inplace=True)
    df = df.dropna(thresh=int(0.5 * df.shape[1]))
    df.reset_index(inplace=True, drop=True)

    numeric_columns = df.select_dtypes(include=['number']).columns
    columns_to_interpolate = [col for col in numeric_columns if col not in ['date', 'time', 'datetime']]
    df[columns_to_interpolate] = df[columns_to_interpolate].interpolate(method='linear')

    percentile_low = 0.001
    percentile_high = 0.999

    max_val = df.groupby('date')[outlier_col].transform(lambda x: x.quantile(percentile_high))
    min_val = df.groupby('date')[outlier_col].transform(lambda x: x.quantile(percentile_low))
    df[outlier_col] = np.where((df[outlier_col] > min_val) & (df[outlier_col] < max_val), df[outlier_col], np.nan)
    df = df.dropna().reset_index(drop=True)

    return df

def get_top_features(df):
    scaler = StandardScaler()

    X = scaler.fit_transform(df[[f'X{i}' for i in range(1, 376)]])
    Y = scaler.fit_transform(df['Y'].values.reshape(-1, 1))

    param_grid = {
      'n_estimators': [100, 200],
      'learning_rate': [0.1, 0.05, 0.01],
      'max_depth': [3, 4, 5],
      'max_features': ['sqrt']
    }

    n_splits = 3
    tscv = TimeSeriesSplit(n_splits=n_splits)

    gbm = GradientBoostingRegressor()
    grid_search = GridSearchCV(estimator=gbm, param_grid=param_grid, cv=tscv, n_jobs=-1)
    Y = np.ravel(Y)
    grid_search.fit(X, Y)

    importances = grid_search.best_estimator_.feature_importances_
    importances_df = pd.DataFrame({'Feature': [f'X{i}' for i in range(1, 376)], 'Importances': importances})
    importances_df.sort_values(by='Importances', inplace=True, ascending=False)

    return handle_multicollinearity(df, importances_df)

def handle_multicollinearity(df, importances_df, threshold=0.7):
    imp_columns = importances_df.head(100)['Feature']
    X = df[imp_columns]
    corr_matrix = X.corr()

    high_corr_dict = {}
    for col in corr_matrix.columns:
        high_corr_cols = [c for c in corr_matrix.columns if c != col and abs(corr_matrix.loc[col, c]) > threshold]
        high_corr_dict[col] = high_corr_cols

    imp_columns = importances_df.head(100)['Feature']
    skipped = set()
    new_features = []
    for imp_col in imp_columns:
        if imp_col in skipped:
            continue
        new_features.append(imp_col)
        for coll in high_corr_dict[imp_col]:
            skipped.add(coll)
    return new_features

def modified_timeseries_model(time_series_data):
    independent_cols = [col for col in time_series_data.columns if col not in ['Y', 'datetime']]
    time_series_model = sm.OLS(time_series_data['Y'], sm.add_constant(time_series_data[independent_cols])).fit()
    print('Time series model summary', flush=True)
    print(time_series_model.summary(), flush=True)
    return time_series_model, independent_cols

def get_timeseries_data(df, imp_columns, y='Y1'):
    scaler = StandardScaler()
    time_series_data = df[imp_columns].copy()
    time_series_data[imp_columns] = scaler.fit_transform(time_series_data[imp_columns])
    time_series_data['Y'] = scaler.fit_transform(df['Y'].values.reshape(-1, 1))
    time_series_data['datetime'] = df['datetime']

    if y == 'Y2':
        time_series_data['Y_lag_2'] = time_series_data['Y'].shift(1)

    time_series_data['Y_lag_1'] = time_series_data['Y'].shift(1)
    time_series_data['Y_lag_301'] = time_series_data['Y'].shift(301)
    return time_series_data[301:].reset_index(drop=True)


# Function to update the plot
def update(frame, ax, test, y_pred, lines, max_display_points=500):
    predictions, actual, time_p, time_a = lines

    if frame < len(test):
        new_time = test.iloc[frame]['datetime']
        predictions.append(y_pred[frame])
        time_p.append(new_time)

        if len(predictions) > max_display_points:
            predictions.pop(0)
            time_p.pop(0)
            actual.pop(0)
            time_a.pop(0)

    ax.clear()
    ax.plot(time_p, predictions, color='maroon', alpha=0.7, label='Predictions')
    ax.plot(time_a, actual, color='darkgreen', alpha=0.7, label='Real time record')
    ax.set_title('Real-Time Calibrations')
    ax.set_xlabel('Time')
    ax.set_ylabel('Y')
    ax.legend(loc='upper right')
    
    new_actual = test.iloc[frame]['Y']
    actual.append(new_actual)
    time_a.append(new_time)

def calibration_plot(test, model, independent_cols):
    fig, ax = plt.subplots(figsize=(10, 6))
    predictions, actual, time_p, time_a = [], [], [], []
    y_pred = model.predict(sm.add_constant(test[independent_cols]))

    anim = FuncAnimation(fig, update, frames=len(test), fargs=(ax, test, y_pred, (predictions, actual, time_p, time_a)),
                         interval=10, blit=False, repeat=False)

    plt.show()

def train_test_split(data_to_split, split_date):
    split_date = pd.to_datetime(split_date)
    train, test = data_to_split.loc[data_to_split['datetime'] <= split_date], data_to_split.loc[data_to_split['datetime'] > split_date]
    return train, test

def get_main_args():
    print('Parsing Arguments config file', flush=True)
    parser = argparse.ArgumentParser(description="This is the calibration module, please ensure your settings for teh script have been specified in the config file")
    parser.add_argument("--config", type=str, default='config.json', help="Config file path")

    args = parser.parse_args()

    # Load configuration settings from the JSON file
    with open(args.config, 'r') as config_file:
        config = json.load(config_file)
    start_date = config.get('start_date', '2022-01-01')
    end_date = config.get('end_date', '2022-03-31')
    split_date = config.get('split_date', '2022-03-12')
    period = config.get('period', 'morning')
    directory = config.get('directory', '/Projects/ClearStreet/qr_takehome')
    hardcode_imp_features = config.get('hardcode_imp_features', True)
    Y = config.get('Y', 'Y1')
    
    return start_date, end_date, split_date, period, directory, hardcode_imp_features, Y

def main():
    start_date, end_date, split_date, period, directory, hardcode_imp_features, Y = get_main_args()
    
    print('Procuring Data...', flush=True)
    data = get_data(start_date, end_date, directory, period)

    if Y == 'Y1':
        data.rename(columns={'Y1': 'Y'}, inplace=True)
        data.drop(columns=['Y2'], inplace=True)
    elif Y == 'Y2':
        data.rename(columns={'Y2': 'Y'}, inplace=True)
        data.drop(columns=['Y1'], inplace=True)

    print('Handling NaN...', flush=True)
    data = handle_nan(data)
    train, test = train_test_split(data, split_date)
    
    print('Performing feature selection... ')
    imp_columns = []
    if hardcode_imp_features:
        if Y == 'Y1':
            imp_columns = ['X121', 'X51', 'X49', 'X205', 'X120', 'X230', 'X253', 'X53', 'X41', 'X96'] #hardcoded
        elif Y == 'Y2':
            imp_columns = ['X51', 'X233', 'X253', 'X52', 'X118', 'X202', 'X49', 'X232', 'X41', 'X372'] #Already found
    else:
        top_features = get_top_features(train)
        imp_columns = top_features[:10]
    print(f'Important features: {imp_columns}', flush=True)

    print('Training time-series AR model', flush=True)
    time_series_data = get_timeseries_data(data, imp_columns, Y)
    train, test = train_test_split(time_series_data, split_date)

    model, independent_cols = modified_timeseries_model(train)
    
    print('Plotting calibrations', flush=True)
    calibration_plot(test.reset_index(drop=True), model, independent_cols)

if __name__ == "__main__":
    main()