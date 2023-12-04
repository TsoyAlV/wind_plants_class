import pandas as pd
import dill
import numpy as np
import models
import time

from forming_data import *

# Удалить позже
with open ('scally','rb') as f:
    scally = dill.load(f)
Ncap = 22.8

# Параметры моделей (дефолтные)
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
        'add_layer': True,
        'loss_func': 'mae'
        },
    'lstm':None}

# Параметры для подбора гиперпараметров (число обучающих итераций и время в сек.)
tune_params = {'n_trials': 3,'timeout_secunds': 60*3} # Удалить и раскоментировать после отладки (пока нужно для проверки работы подбора гиперпараметров)
# tune_params = {'n_trials': 50,'timeout_secunds': 3600 * 3}

class Model:
    """
    Необходимо выбрать модель среди ['catboost','lgbm','fc_nn','lstm'] а так же оптионально свои параметры настройки и метрику
    """
    def __init__(self, model_name='catboost', metric='mae'):
        self.model = None
        self.best_params = None
        self.df_err = None
        self.history = None
        self.model_name = model_name
        assert self.model_name in ['catboost','lgbm','fc_nn','lstm'], f"Модель '{model_name}' не нахоится в списке доступных: ['catboost','lgbm','fc_nn','lstm']"
        self.model_params = params[model_name]
        self.metric=metric

    def prep_all_data(self, loc_points, eng, name_gtp='GUK_3'):
        # пока не буду прорабатывать данную функцию полностью
        self.df_all, self.nums = first_prep_data(loc_points, 
              eng=eng, 
              name_gtp = name_gtp, 
              start_date=str(datetime.datetime.now()-datetime.timedelta(days=365*3))[:10], 
              end_date=str(datetime.datetime.now()-datetime.timedelta(days=14))[:10])
    
    def prep_data(self, num):
        df, self.scallx, self.scally = do_scaling(self.df_all)
        df = df.dropna()
        #trees
        df_trees = form_batch_trees(df, self.nums)[num]
        self.x_trees = df_trees.drop(columns=[f'N_{num}'])
        self.y_trees = df_trees[[f'N_{num}']]

        # nns
        df_nn = form_batch_nn(df, self.nums)[num]
        y_cols = list(map(lambda x: f'N_targ_{x}', range(24)))
        self.x_fcnn = df_nn.drop(columns=y_cols).drop(columns=[f'N_{num}',f'wind_speed_{num}_10m'])
        self.y_fcnn = df_nn[y_cols]
        self.x_lstm = np.array(self.x_fcnn).reshape(-1,24,2)
        self.y_lstm = self.y_fcnn.copy()

    def fit_predict(self, num, test_size=0.175, purpose='fit_by_default_params'):
        """
        purpose: Optional -> ['test', 'fit_by_default_params', 'tune_params']\n
            test - означает, что производится легкое обучение (25 итераций)\n
            fit_by_default_params - означает, что модель обучается с заданными по дефолту параметрами\n
            tune_params - означает, что будут подбираться гиперпараметры и создастся отдельная переменная класса self.best_params. На обучение уходит значительно больше времени!!!
        """
        assert purpose in ['test', 'fit_by_default_params', 'tune_params'], f'Аргумента функции fit_predict purpose="{purpose}" нет в списке доступных ["test", "fit_by_default_params", "tune_params"]'
        self.purpose=purpose
        t_initial = time.time()
        # catboost
        if self.model_name == 'catboost':
            if self.purpose == 'test':
                epoches = 25
                verbose = 10
                self.df_err, self.model, self.history, self.best_params = models.solve_model_catboost(self.x_trees, self.y_trees, num, params['catboost'], epoches, scally, Ncap, False, 
                                                                                    None, verbose, test_size=test_size)
            elif self.purpose == 'fit_by_default_params':
                epoches = 1000
                verbose = 0
                self.df_err, self.model, self.history, self.best_params = models.solve_model_catboost(self.x_trees, self.y_trees, num, params['catboost'], epoches, scally, Ncap, False, 
                                                                                    None, verbose, test_size=test_size)
            elif self.purpose == 'tune_params':
                epoches = 1000
                verbose = 0
                self.df_err, self.model, self.history, self.best_params = models.solve_model_catboost(self.x_trees, self.y_trees, num, params['catboost'], epoches, scally, Ncap, True, 
                                                                                    tune_params, verbose, test_size=test_size)

        # lgbm
        elif self.model_name == 'lgbm':
            if self.purpose == 'test':
                epoches = 25
                verbose = 10
                self.df_err, self.model, self.history, self.best_params = models.solve_model_lgbm(self.x_trees, self.y_trees, num, params['lgbm'], epoches, scally, Ncap, False, 
                                                                                None, verbose, test_size=test_size)
            elif self.purpose == 'fit_by_default_params':
                epoches = 1000
                verbose = 0
                self.df_err, self.model, self.history, self.best_params = models.solve_model_lgbm(self.x_trees, self.y_trees, num, params['lgbm'], epoches, scally, Ncap, False, 
                                                                                None, verbose, test_size=test_size)

            elif self.purpose == 'tune_params':
                epoches = 1000
                verbose = 0
                self.df_err, self.model, self.history, self.best_params = models.solve_model_lgbm(self.x_trees, self.y_trees, num, params['lgbm'], epoches, scally, Ncap, True, 
                                                                                tune_params, verbose, test_size=test_size)


        # fc_nn
        elif self.model_name == 'fc_nn':
            if self.purpose == 'test':
                epoches = 25
                verbose = 1
                self.df_err, self.model, self.history, self.best_params = models.solve_model_fc_nn(self.x_fcnn, self.y_fcnn, num, params['fc_nn'], epoches, scally, Ncap, False, 
                                                                                                None, 1,verbose, test_size=test_size)
            elif self.purpose == 'fit_by_default_params':
                epoches = 1500
                verbose = 0
                self.df_err, self.model, self.history, self.best_params = models.solve_model_fc_nn(self.x_fcnn, self.y_fcnn, num, params['fc_nn'], epoches, scally, Ncap, False, 
                                                                                                None, 1,verbose, test_size=test_size)
            elif self.purpose == 'tune_params':
                epoches = 1500
                verbose = 0
                self.df_err, self.model, self.history, self.best_params = models.solve_model_fc_nn(self.x_fcnn, self.y_fcnn, num, params['fc_nn'], epoches, scally, Ncap, True, 
                                                                                                tune_params = tune_params, random_seed=1, verbose_=verbose, test_size=test_size)
                

        # LSTM
        elif self.model_name == 'lstm':
            if self.purpose == 'test':
                epoches = 25
                verbose = 1
                self.df_err, self.model, self.history, self.best_params = models.solve_model_lstm(self.x_lstm, self.y_lstm, num, epoches, scally, Ncap, False, 
                                                                                None,  random_seed = 42, verbose_=verbose, test_size=test_size)
            elif self.purpose == 'fit_by_default_params':
                epoches = 1500
                verbose = 0
                self.df_err, self.model, self.history, self.best_params = models.solve_model_lstm(self.x_lstm, self.y_lstm, num, epoches, scally, Ncap, False, 
                                                                                None,  random_seed = 42, verbose_=verbose, test_size=test_size)
            elif self.purpose == 'tune_params':
                epoches = 1500
                verbose = 0
                self.df_err, self.model, self.history, self.best_params = models.solve_model_lstm(self.x_lstm, self.y_lstm, num, epoches, scally, Ncap, True, 
                                                                                tune_params,  random_seed = 42, verbose_=verbose, test_size=test_size)
        
        # Считаем время выполнения расчета расчета
        learning_time = round(float(f'{time.time() - t_initial:3.3f}'), 2)
        if self.purpose=='test':
            if self.model_name in ['lgbm', 'catboost']:
                print(f'Время выполнения обучения модели (25 эпох): {learning_time} c.', f'Максимальное время расчета модели cо стандартными гиперпараметрами: {learning_time/25*1000//60} мин {round(learning_time/25*1000, 0)%60} с')    
            if self.model_name in ['fc_nn', 'lstm']:
                print(f'Время выполнения обучения модели (25 эпох): {learning_time} c.', f'Максимальное время расчета модели cо стандартными гиперпараметрами: {learning_time/25*1500//60} мин {round(learning_time/25*1500, 0)%60} с')    
        elif self.purpose=='fit_by_default_params':
            print(f'Время обучения модели "{self.model_name}" составило: {learning_time//60} мин {learning_time%60} с') 
        elif self.purpose=='tune_params':
            print(f'Время обучения модели "{self.model_name}" с учетом подбора гиперпараметров составило: {learning_time//60} мин {learning_time%60} с') 


class Pipeline_wind_forecast:
    """
    Необходимо выбрать модель среди ['catboost','lgbm','fc_nn','lstm'] а так же оптионально свои параметры настройки и метрику
    """
    def __init__(self, model_name='catboost', metric='mae', debug=False):
        self.model = None
        self.df_err = None
        self.history = None
        self.all_data_weather_points_with_wind_openmeteo = None
        self.model_name = model_name
        assert self.model_name in ['catboost','lgbm','fc_nn','lstm'], f"Модель '{model_name}' не нахоится в списке доступных: ['catboost','lgbm','fc_nn','lstm']"
        self.model_params = params[model_name]
        self.metric=metric
        self.debug=debug

    
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
            self.df_err, self.model, self.history = models.solve_model_lgbm(self.x_trees, self.y_trees, '26', params['lgbm'], epoches, scally, Ncap, verbose, test_size=0.175)
        elif self.model_name == 'fc_nn':
            if self.debug:
                epoches = 25
                verbose = 1
            else:
                epoches = 1500
                verbose = 0
            self.df_err, self.model, self.history = models.solve_model_fc_nn(self.x_fcnn, self.y_fcnn, '26', params['fc_nn'], epoches, scally, Ncap, 1, verbose, test_size=0.175)
        elif self.model_name == 'lstm':
            if self.debug:
                epoches = 25
                verbose = 1
            else:
                epoches = 1500
                verbose = 0
            self.df_err, self.model, self.history = models.solve_model_lstm(self.x_lstm, self.y_lstm, '26', epoches, scally, Ncap,  random_seed = 42, verbose_=verbose, test_size=0.175)
        
        # Считаем время выполнения расчета расчета
        learning_time = round(float(f'{time.time() - t_initial:3.3f}'), 2)
        if self.debug:
            if self.model_name in ['lgbm', 'catboost']:
                print(f'Время выполнения обучения модели (25 эпох): {learning_time} c.', f'Максимальное время расчета модели {learning_time/25*1000//60} мин {round(learning_time/25*1000, 0)%60} с')    
            if self.model_name in ['fc_nn', 'lstm']:
                print(f'Время выполнения обучения модели (25 эпох): {learning_time} c.', f'Максимальное время расчета модели {learning_time/25*1500//60} мин {round(learning_time/25*1500, 0)%60} с')    
        else:
            print(f'Время обучения модели составило: {learning_time//60} мин {learning_time%60} с') 