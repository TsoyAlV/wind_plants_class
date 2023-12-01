import pandas as pd
import dill
import numpy as np
import models
import sklearn
import time

# Удалить позже
with open ('scally','rb') as f:
    scally = dill.load(f)
Ncap = 22.8

params = {
    'catboost':
        {'learning_rate': 0.024648325316324382,
        'l2_leaf_reg': 8.14993506614837,
        'depth': 8},
    'lgbm':
        {'num_leaves': 99,
        'learning_rate': 0.01040976680049962,
        'linear_lambda': 9.09258101406036,
        'linear_tree': True}, 
    'fc_nn': 
        {'activate1': 'selu',
        'activate2': 'relu',
        'activate3': 'relu',
        'activate4': 'relu',
        'activate5': 'relu',
        'units0': 89,
        'units1': 21,
        'units2': 18,
        'units3': 9,
        'add_layer': True},
    'lstm':None
        }


class Model:
    """
    Необходимо выбрать модель среди ['catboost','lgbm','fc_nn','lstm'] а так же оптионально свои параметры настройки и метрику
    """
    def __init__(self, model_name='catboost', metric='mae', debug=False):
        self.model = None
        self.model_name = model_name
        self.model_params = params[model_name]
        self.metric=metric
        self.debug=debug
        assert self.model_name in ['catboost','lgbm','fc_nn','lstm'], "Модель не нахоится в списке доступных: ['catboost','lgbm','fc_nn','lstm']"

    
    def prep_data(self):
        # пока не буду прорабатывать данную функцию полностью
        num = '26'
        # -----------
        if self.model_name in ['fc_nn','lstm']:
            with open('tmp_df_nn.dat','rb') as f:
                df_nn = dill.load(f)
            y_cols = list(map(lambda x: f'N_targ_{x}', range(24)))
            self.x_fcnn = df_nn.drop(columns=y_cols).drop(columns=[f'N_{num}',f'wind_speed_{num}_10m'])
            y_cols = list(map(lambda x: f'N_targ_{x}', range(24)))
            self.x_fcnn = df_nn.drop(columns=y_cols).drop(columns=[f'N_{num}',f'wind_speed_{num}_10m'])
            self.y_fcnn = df_nn[y_cols]
            self.x_lstm = np.array(self.x_fcnn).reshape(-1,24,2)
            self.y_lstm = self.y_fcnn.copy()
            self.df_err = None
            self.model = None
            self.history = None
        
        if self.model_name in ['catboost','lgbm']:
            with open('tmp_df_trees.dat','rb') as f:
                df_trees = dill.load(f)        
            self.x_trees = df_trees.drop(columns=[f'N_{num}'])
            self.y_trees = df_trees[[f'N_{num}']]


    def fit_predict(self, optuna = False, test_size=0.175):
        t_initial = time.time()
        if self.model_name == 'catboost':
            if self.debug:
                epoches = 25
                verbose = 10
            else:
                epoches = 1000
                verbose = 0
            self.df_err, self.model, self.history = models.solve_model_catboost(self.x_trees, self.y_trees, '26', params['catboost'], epoches, scally, Ncap, verbose, test_size=0.175)
        elif self.model_name == 'lgbm':
            if self.debug:
                epoches = 25
                verbose = 10
            else:
                epoches = 1000
                verbose = 0
            self.df_err, self.model, self.history = models.solve_model_lgbm(self.x_trees, self.y_trees, '26', params['lgbm'], epoches, scally, Ncap, verbose)
        elif self.model_name == 'fc_nn':
            if self.debug:
                epoches = 25
                verbose = 1
            else:
                epoches = 1500
                verbose = 1
            self.df_err, self.model, self.history = models.solve_model_fc_nn(self.x_fcnn, self.y_fcnn, '26', params['fc_nn'], epoches, scally, Ncap, 1,verbose)
        elif self.model_name == 'lstm':
            if self.debug:
                epoches = 25
                verbose = 1
            else:
                epoches = 1500
                verbose = 1
            self.df_err, self.model, self.history = models.solve_model_lstm(self.x_lstm, self.y_lstm, '26', epoches, scally, Ncap,  random_seed = 42, verbose_=verbose)
        
        learning_time = round(float(f'{time.time() - t_initial:3.3f}'), 2)
        if self.debug:
            if self.model_name in ['lgbm', 'catboost']:
                print(f'Время выполнения обучения модели (25 эпох): {learning_time} c.', f'Максимальное время расчета модели {learning_time/25*1000//60} мин {round(learning_time/25*1000, 0)%60} с')    
            if self.model_name in ['fc_nn', 'lstm']:
                print(f'Время выполнения обучения модели (25 эпох): {learning_time} c.', f'Максимальное время расчета модели {learning_time/25*1500//60} мин {round(learning_time/25*1500, 0)%60} с')    
        else:
            print(f'Время обучения модели составило: {learning_time//60} мин {learning_time%60} с')    


class Pipeline_wind_forecast:
    def __init__(self, models=['catboost','lgbm','fc_nn','lstm'], model_params = {}, metric='mae'):
        self.models = models
        self.model_params = model_params
        # self.df_nn = None
        # self.df_trees = None
        self.params = params
        self.x_fcnn = None
        self.y_fcnn = None
        self.x_lstm = None
        self.y_lstm = None
        self.x_trees = None
        self.y_trees = None
        self.optuna = None
        self.metric = None


    def prep_data(self):
        # пока не буду прорабатывать данную функцию полностью
        num = '26'
        # -----------
        
        with open('tmp_df_nn.dat','rb') as f:
            df_nn = dill.load(f)
        with open('tmp_df_trees.dat','rb') as f:
            df_trees = dill.load(f)        

        y_cols = list(map(lambda x: f'N_targ_{x}', range(24)))

        self.x_fcnn = df_nn.drop(columns=y_cols).drop(columns=[f'N_{num}',f'wind_speed_{num}_10m'])
        self.y_fcnn = self.df_nn[y_cols]
        self.x_lstm = np.array(self.x_fcnn).reshape(-1,24,2)
        self.y_lstm = np.array(self.y_fcnn.copy())
        self.x_trees = df_trees.drop(columns=[f'N_{num}'])
        self.y_trees = df_trees[[f'N_{num}']]
    

    def fit(self, start_date_test, optuna = False, test_size=0.175):
        pass