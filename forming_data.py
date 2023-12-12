import datetime
import pandas as pd
import numpy as np
import time
import requests
from sqlalchemy import create_engine
import traceback
from sklearn.preprocessing import MinMaxScaler

# configs
eng = create_engine('postgresql://postgres:achtung@192.168.251.133:5432/fortum_wind')


def openmeteo_request(latitude = 48.117605, longitude = 39.954884, start_date='2022-09-12', end_date='2022-09-12'):
    """
    Функция берет данные погоды (скорости вретра на 10м) из openmeteo по широте и долготе а так же диапазону дат
    """
    
    try:
        r = requests.get("http://api.open-meteo.com/v1/forecast", params = {'latitude': latitude, 
                                                                            'longitude': longitude, 
                                                                            'hourly': ['wind_speed_10m'],
                                                                            'start_date': start_date,
                                                                            'end_date': end_date,
                                                                            'timezone': 'UTC'})
        r.json()['hourly']
    except:
        r = requests.get("http://archive-api.open-meteo.com/v1/archive", params = {'latitude': latitude, 
                                                                            'longitude': longitude, 
                                                                            'hourly': ['wind_speed_10m'],
                                                                            'start_date': start_date,
                                                                            'end_date': end_date,
                                                                            'timezone': 'UTC'})
        r.json()['hourly']
    try:
        result = r.json()
    except:
        print('ree')
        traceback.print_exc()
    return result['hourly']

def first_prep_data(loc_points, 
              eng, 
              name_gtp = 'GUK_3', 
              start_date=str(datetime.datetime.now()-datetime.timedelta(days=365*3))[:10], 
              end_date=str(datetime.datetime.now()-datetime.timedelta(days=14))[:10]):
    
    """
    Функция формирует датафрейм из данных по выработке по каждому из ветряков и из заданных точек (loc_points) на карте (широта, долгота) выбирает и добавляет в датафрейм точку с наибольшей корреляцией (сравниваются параметры скорости ветра на конкретном ветряке с данными каждой точки (loc_points) из openmeteo и выбирается лучшая) 
    """
    # Данные из БД
    df_res = pd.DataFrame()
    engine = create_engine('postgresql://postgres:achtung@192.168.251.133:5432/fortum_wind')
    df_req = pd.read_sql_query(f"""select tstamp_initial_utc, var_name, value from wind_ini_data as wi 
                where wi.tstamp_initial_utc >= '{start_date}' 
                and wi.tstamp_initial_utc <= '{end_date}' 
                and wi.gtp_name = '{name_gtp}'""", engine)
    
    for name in df_req.var_name.unique():
        df_req_test = df_req[df_req.var_name == name]
        test = df_req_test.tstamp_initial_utc.duplicated()
        df_req = df_req.drop(test[test == True].index)

    df_req = df_req.pivot(index = 'tstamp_initial_utc', columns='var_name',values='value').dropna()
    df_req.index = pd.to_datetime(df_req.index, utc=True).tz_convert('UTC').tz_localize(None)
    
    set_nums = set(map(lambda x: x if 'N_' in x else None, df_req.columns))
    set_nums.remove(None)
    set_nums.remove('N_GTP')
    nums = []
    start_date = pd.to_datetime(df_req.dropna().index[0]).date()
    end_date   = pd.to_datetime(df_req.dropna().index[-1]).date()
    
    # Данные из openmeteo
    all_weather_points = []
    for coord in loc_points:
        latitude = coord[0]
        longitude = coord[1]
        weather = openmeteo_request(latitude=latitude,longitude=longitude, start_date=start_date, end_date=end_date)
        time.sleep(1)
        weather_df = pd.DataFrame(np.array([weather[f'wind_speed_10m']]).T, index=weather['time'], columns=[f'wind_speed_10m'])
        weather_df.index = pd.to_datetime(weather_df.index.astype(str), format='%Y-%m-%dT%H:%M') 
        all_weather_points.append(weather_df)
        
    for num in set_nums:
        nums.append(num[2:])
        
    for num in nums:
        tmp_df = df_req[[f'N_{num}', f'WS_{num}']].copy()
        compare_dict = {}
        compare_dict['spearman'] = dict()
        compare_dict['pearson'] = dict()
        i = 0

        # ищем наиболее подходящие координаты для построения прогноза из прогноза погоды. Строим корреляцию с реальной скоростью в месте установки ветряка
        for weather_df in all_weather_points:
            tmp_df1 = pd.concat([tmp_df[start_date:end_date][[f'WS_{num}']], #tmp_df[start_date:end_date][['WS_06']]
                                weather_df[start_date:end_date][f'wind_speed_10m']/2], axis=1)
            compare_dict['pearson'][i] = tmp_df1.corr(method='pearson').iloc[0,1]
            compare_dict['spearman'][i] = tmp_df1.corr(method='spearman').iloc[0,1]
            i+=1
            
        # сравниваем:
        compare_df = pd.DataFrame(np.array([list(compare_dict['pearson'].values()),
                                            list(compare_dict['spearman'].values())]).T, 
                                  index=compare_dict['spearman'].keys(),
                                  columns=['pearson','spearman'])
        compare_df['mean'] = (compare_df['pearson'] + compare_df['spearman'])/2
        ind_of_location = compare_df.sort_values('mean', na_position='first').index[-1]
        best_latitude_from_openmeteo = loc_points[ind_of_location][0]
        best_longitude_from_openmeteo = loc_points[ind_of_location][1]
        print('№ ветряка: ', num, 'с координатами: ', loc_points[ind_of_location])
        weather = openmeteo_request(latitude=best_latitude_from_openmeteo,
                                    longitude=best_longitude_from_openmeteo, 
                                    start_date=start_date, 
                                    end_date=end_date)
        weather_df = pd.DataFrame(np.array([weather[f'wind_speed_10m']]).T, index=weather['time'], columns=[f'wind_speed_{num}_10m'])
        weather_df.index = pd.to_datetime(weather_df.index.astype(str), format='%Y-%m-%dT%H:%M')    
        df_res = pd.concat([df_res, tmp_df[[f'N_{num}', f'WS_{num}']], weather_df], axis=1)
    eng.dispose()
# 	                    N_06	WS_06	wind_speed_06_10m	N_19	WS_19	wind_speed_19_10m
# 2020-12-01 00:00:00	NaN	     NaN	    26.3	        NaN 	NaN     	26.3
# 2020-12-01 01:00:00	NaN	     NaN	    26.1	        NaN 	NaN     	26.1
# 2020-12-01 02:00:00	NaN	     NaN	    25.9	        NaN 	NaN     	25.9
# 2020-12-01 03:00:00	NaN	     NaN	    25.8	        NaN 	NaN     	25.8
    return df_res, nums

def do_scaling(df):
    df2 = df.copy()
    set_nums = set(map(lambda x: x if 'N_' in x else None, df.columns))
    set_nums.remove(None)
    nums = []
    
    for num in set_nums:
        nums.append(num[2:])
    N_nums = [f'N_{num}' for num in nums]
    wind_nums = [f'wind_speed_{num}_10m' for num in nums]
    scally = MinMaxScaler().fit(np.array(df2[N_nums]).reshape(-1,1))
    scallx = MinMaxScaler().fit(np.array(df2[wind_nums]).reshape(-1,1))
    for col in N_nums:
        df2[col] = scally.transform(df2[[col]])
    for col in wind_nums:
        df2[col] = scallx.transform(df2[[col]])
    # df2 = pd.concat([pd.DataFrame(scally.transform(df2.iloc[:,[0]]), index=df2.index, columns=[f'N_{num}']),
    #                  pd.DataFrame(scallx.transform(df2.iloc[:,1:]), index=df2.index, columns=[f'wind_speed_{num}_10m'])], axis=1)
    return df2, scallx, scally

def form_batch_trees(df, nums):
    'Функция формирует словарь из batches для каждого ветряка (на основе scallx, scally???)'
    dict_df_num = {}
    tmp_df = None
    df2 = df.copy()
    df2 = df2.reindex(pd.date_range(df2.index[0], df2.index[-1], freq='1h'))
    for num in nums:
        tmp_df = df2[[f'N_{num}',f'wind_speed_{num}_10m']]
        for shift in range(24):
            tmp_df[f'N_{num}_shift_{48-shift}'] = tmp_df[f'N_{num}'].shift(48-shift)
            tmp_df[f'weather_{num}_shift_{48-shift}'] = tmp_df[f'wind_speed_{num}_10m'].shift(48-shift)
            tmp_df[f'diff_N_wind_{shift}'] = tmp_df[f'N_{num}_shift_{48-shift}'] - tmp_df[f'weather_{num}_shift_{48-shift}']
        for shift in [-1,1,2]:
            tmp_df[f'wind_shift_{shift}'] = tmp_df[f'wind_speed_{num}_10m'].shift(shift)
        tmp_df = tmp_df.dropna()
        dict_df_num[num] = tmp_df
    return dict_df_num

def form_batch_nn(df, nums):
    'Функция формирует словарь из batches для каждого ветряка (на основе scallx, scally???)'
    dict_df_num = {}
    tmp_df = None
    df2 = df.copy()
    df2 = df2.reindex(pd.date_range(df2.index[0], df2.index[-1], freq='1h'))
    for num in nums:
        tmp_df = df2[[f'N_{num}',f'wind_speed_{num}_10m']]
        for shift in range(24):
            tmp_df[f'N_{num}_shift_{48-shift}'] = tmp_df[f'N_{num}'].shift(48-shift)
            tmp_df[f'weather_{num}_shift_{-shift}'] = tmp_df[f'wind_speed_{num}_10m'].shift(-shift)
            tmp_df[f'N_targ_{shift}'] = tmp_df[f'N_{num}'].shift(-shift)
        tmp_df = tmp_df.dropna()
        dict_df_num[num] = tmp_df
    return dict_df_num


if __name__ == '__main__':
    loc_points = [[48.117803, 39.977637],
                    [48.118372, 39.961951],
                    [48.102330, 39.972937],
                    [48.098727, 39.940489],
                    [48.096223, 39.948417],
                    [48.101592, 39.972652],
                    [48.095786, 39.976725],
                    [48.087902, 39.950087],
                    [48.090580, 39.941083],
                    [48.082495, 39.965106],
                    [48.083546, 39.981992],
                    [48.075435, 39.970784],
                    [48.068148, 39.971316],
                    [48.071882, 39.997197],
                    [48.078555, 39.999711],
                    [48.116879, 39.977684],
                    [48.118266, 39.963142]]
    df, nums = first_prep_data(loc_points = loc_points, eng = eng, name_gtp = 'GUK_3')
    df2, scallx, scally = do_scaling(df)
    dict_tree = form_batch_trees(df2, nums)
    print(df2, nums)
    print('----------------------------------------')
    for num in nums:
        print(dict_tree[num])