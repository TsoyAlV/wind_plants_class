import pandas as pd
import dill
import numpy as np
import models
import time

from forming_data import *


# Параметры моделей (дефолтные)
params = {
    'catboost':
        {'learning_rate': 0.024648325316324382,
        'l2_leaf_reg': 8.14993506614837,
        'depth': 8,
        'loss_func':'MAPE'},

    'lgbm':
        {'num_leaves': 99,
        'learning_rate': 0.01040976680049962,
        'lambda_l1':0,
        'lambda_l2':0,
        'linear_lambda': 9.09258101406036,
        'linear_tree': True,
        'loss_function':'mae'
        }, 

    'fc_nn': 
        {'activate1': 'selu',
        'activate2': 'relu',
        'activate3': 'relu',
        'activate4': 'relu',
        'activate5': 'relu',
        'units0': 89,
        'units1': 21,
        'units2': 18,
        'add_units3': 9,
        'add_layer': True,
        'loss_func': 'mae'},

    'lstm':
        {'units0': 40,
        'activate1': 'sigmoid',
        'activate3': 'linear',
        'learning_rate': 0.015,
        'loss_func': 'mae'}
        }
# Параметры для подбора гиперпараметров (число обучающих итераций и время в сек.)
# tune_params = {'n_trials': 2,'timeout_secunds': 60*2} # Удалить и раскоментировать после отладки (пока нужно для проверки работы подбора гиперпараметров)
tune_params = {'n_trials': 50,'timeout_secunds': 3600 * 2}

class Model:
    """
    Необходимо выбрать модель среди ['catboost','lgbm','fc_nn','lstm'] а так же оптионально свои параметры настройки и метрику.
    Класс содержит:
        - методы по формированию данных для обучения и прогноза,
        - метод для предсказания.
    Является родительским классом для основного класса Pipeline_wind_forecast
    """
    def __init__(self, model_name='catboost'):
        self.model = None
        self.best_params = None
        self.df_err = None
        self.history = None
        self.model_name = model_name
        assert self.model_name in ['catboost','lgbm','fc_nn','lstm'], f"Модель '{model_name}' не нахоится в списке доступных: ['catboost','lgbm','fc_nn','lstm']"
        self.model_params = params[model_name]
    
    def get_available_GTPs(self, eng):
        """
        Метод для получения списка доступных ГТП
        На входе - engine (подключение к БД)
        На выходе - список доступных ГТП
        """
        res = pd.read_sql_query(f"""select gtp_name from wind_gtp """, eng)
        eng.dispose()
        print(list(res.values.astype(str).T[0]))
        return res.values.astype(str).T[0]

    def prep_all_data(self, loc_points, eng, name_gtp='GUK_3'):
        """
        Метод готовит данные для всех ветряков заданной ГТП
        На входе:
            - loc_points - локации для проверки на наилучшую корреляцию # [[45.22, 43.5513],
                                                                           [42.35, 43.5512],
                                                                           ...............]]
            - eng - подключение к БД
            - name_gtp - наименование ГТП для формирования данных по всем ветрякам данной ГТП
        На выходе:
            - сохраняет внутри класса self.Ncap - коэффициент для оценки точности (обычно максимальная выработка по всем ветрякам ГТП)
            - сохраняет внутри класса self.gtp_id 
        """
        # пока не буду прорабатывать данную функцию полностью
        self.df_all, self.nums = first_prep_data(loc_points, 
                                                eng=eng, 
                                                name_gtp = name_gtp, 
                                                start_date=str(datetime.datetime.now()-datetime.timedelta(days=365*3))[:10], 
                                                end_date=str(datetime.datetime.now()-datetime.timedelta(days=1))[:10])
        self.nums.sort()
        res = pd.read_sql_query(f"""select gtp_power, gtp_id from wind_gtp where gtp_name = '{name_gtp}'""", eng)
        assert res.shape == (1, 2), f'Не удалось найти конкретную выработку Ncap для дальнейшего расчета погрешности. Должно быть res.shape == (1, 2) а получилось {res.shape}'
        self.Ncap = res.values[0][0]
        self.gtp_id = int(res.values[0][1])
        eng.dispose()    

    def prep_data(self, df_all, num):
        """
        Метод готовит данные для конкретного ветряка по его номеру
        На входе:
            - df_all - датафрейм с данными по всем ветрякам
            - num - номер ветряка
        На выходе:
            - self.x_trees, self.y_trees, self.x_fc_nn, self.y_fc_nn, self.x_lstm, self.y_lstm, self.scallx, self.scally - датафреймы для обучения и предсказания и MinMaxscaller's 
        """
        df, self.scallx, self.scally = do_scaling(df_all)
        df = df.dropna()
        #trees
        df_trees = form_batch_trees(df, self.nums)[num]
        self.x_trees = df_trees.drop(columns=[f'N_{num}'])
        self.y_trees = df_trees[[f'N_{num}']]

        # nns
        df_nn = form_batch_nn(df, self.nums)[num]
        y_cols = list(map(lambda x: f'N_targ_{x}', range(24)))
        self.x_fc_nn = df_nn.drop(columns=y_cols).drop(columns=[f'N_{num}',f'wind_speed_{num}_10m'])
        self.y_fc_nn = df_nn[y_cols]
        self.x_lstm = np.array(self.x_fc_nn).reshape(-1,24,2)
        self.y_lstm = self.y_fc_nn.copy()
        return self.x_trees, self.y_trees, self.x_fc_nn, self.y_fc_nn, self.x_lstm, self.y_lstm, self.scallx, self.scally

    def fit_predict(self, x, y, num, params, model_name, epoches=None, early_stopping_rounds=None, test_size=None, start_test_date=None, purpose='fit_by_setted_params'):
        """
        Метод производит прогнозирование по заданным параметрам:
        Входные данные:
            - x - датафрейм с признаками для прогнозирования
            - y - датафрейм с таргетом
            - num - номер ветряка
            - model_name - модель предсказания, выбирается из списка ['catboost','lgbm','fc_nn','lstm']
            - epoches - число эпох предсказания
            - early_stopping_rounds - число эпох, в случае, если нет улучшения за этот период, обучение останавливается
            - test_size - разделение на выборки тренировочную и тестовую (задается доля тестовой выборки к общему датафрейму от 0 до 1) по дефолту None (задание тестовой выборки выбирается либо test_size либо start_test_date)
            - start_test_date - разделение на выборки тренировочную и тестовую путем задания времени старта тестовых данных по дефолту None (задание тестовой выборки выбирается либо test_size либо start_test_date)
            - purpose: Optional -> ['test', 'fit_by_setted_params', 'tune_params']
               * test - означает, что производится легкое обучение (25 итераций)
               * fit_by_setted_params - означает, что модель обучается с заданными по дефолту параметрами
               * tune_params - означает, что будут подбираться гиперпараметры и создастся отдельная переменная класса self.best_params. На обучение уходит значительно больше времени!!!
        Выходные данные:
            - объекты класса self.df_err, self.model, self.history, self.best_params - результаты (df, обученная модель, history для отображения процесса обучения и best_params - подбор гиперпараметров)
        """
        if num is None:
            num = self.data['nums'][0]
            print(f'Первое обучение производится по ветряку {num}')
        assert purpose in ['test', 'fit_by_setted_params', 'tune_params'], f'Аргумента функции fit_predict purpose="{purpose}" нет в списке доступных ["test", "fit_by_setted_params", "tune_params"]'
        self.purpose=purpose
        t_initial = time.time()
        # catboost
        if model_name == 'catboost':
            print(f"Начинаю расчет модели {model_name}")
            if self.purpose == 'test':
                epoches = 25
                early_stopping_rounds = epoches
                verbose = 10
                self.df_err, self.model, self.history, self.best_params = models.solve_model_catboost(x, y, num, params['catboost'], epoches, early_stopping_rounds, self.scally, self.Ncap, False, None, verbose, test_size=test_size, start_test_date=start_test_date)
            elif self.purpose == 'fit_by_setted_params':
                if epoches is None:
                    epoches = 1000
                else:
                    pass
                if early_stopping_rounds is None:
                    early_stopping_rounds = 50
                else:
                    pass
                verbose = 0
                self.df_err, self.model, self.history, self.best_params = models.solve_model_catboost(x, y, num, params['catboost'], epoches, early_stopping_rounds, self.scally, self.Ncap, False, None, verbose, test_size=test_size, start_test_date=start_test_date)
            elif self.purpose == 'tune_params':
                if epoches is None:
                    epoches = 1000
                else:
                    pass
                if early_stopping_rounds is None:
                    early_stopping_rounds = 50
                else:
                    pass
                verbose = 0
                self.df_err, self.model, self.history, self.best_params = models.solve_model_catboost(x, y, num, params['catboost'], epoches, early_stopping_rounds, self.scally, self.Ncap, True, tune_params, verbose, test_size=test_size, start_test_date=start_test_date)

        # lgbm
        elif model_name == 'lgbm':
            print(f"Начинаю расчет модели {model_name}")
            if self.purpose == 'test':
                epoches = 25
                early_stopping_rounds = epoches
                verbose = 10
                self.df_err, self.model, self.history, self.best_params = models.solve_model_lgbm(x, y, num, params['lgbm'], epoches, early_stopping_rounds, self.scally, self.Ncap, False, 
                                                                                None, verbose, test_size=test_size, start_test_date=start_test_date)
            elif self.purpose == 'fit_by_setted_params':
                if epoches is None:
                    epoches = 1000
                else:
                    pass
                if early_stopping_rounds is None:
                    early_stopping_rounds = 50
                else:
                    pass
                verbose = 0
                self.df_err, self.model, self.history, self.best_params = models.solve_model_lgbm(x, y, num, params['lgbm'], epoches, early_stopping_rounds, self.scally, self.Ncap, False, 
                                                                                None, verbose, test_size=test_size, start_test_date=start_test_date)

            elif self.purpose == 'tune_params':
                if epoches is None:
                    epoches = 1000
                else:
                    pass
                if early_stopping_rounds is None:
                    early_stopping_rounds = 50
                else:
                    pass
                verbose = 0
                self.df_err, self.model, self.history, self.best_params = models.solve_model_lgbm(x, y, num, params['lgbm'], epoches, early_stopping_rounds, self.scally, self.Ncap, True, 
                                                                                tune_params, verbose, test_size=test_size, start_test_date=start_test_date)


        # fc_nn
        elif model_name == 'fc_nn':
            print(f"Начинаю расчет модели {model_name}")
            if self.purpose == 'test':
                epoches = 25
                early_stopping_rounds = epoches
                verbose = 1
                self.df_err, self.model, self.history, self.best_params = models.solve_model_fc_nn(x, y, num, params['fc_nn'], epoches, early_stopping_rounds, self.scally, self.Ncap, False, None, 1,verbose, test_size=test_size, start_test_date=start_test_date)
            elif self.purpose == 'fit_by_setted_params':
                if epoches is None:
                    epoches = 1500
                else:
                    pass
                if early_stopping_rounds is None:
                    early_stopping_rounds = 50
                else:
                    pass
                verbose = 0
                self.df_err, self.model, self.history, self.best_params = models.solve_model_fc_nn(x, y, num, params['fc_nn'], epoches, early_stopping_rounds, self.scally, self.Ncap, False, None, 1,verbose, test_size=test_size, start_test_date=start_test_date)
            elif self.purpose == 'tune_params':
                if epoches is None:
                    epoches = 1500
                else:
                    pass
                if early_stopping_rounds is None:
                    early_stopping_rounds = 50
                else:
                    pass
                verbose = 0
                self.df_err, self.model, self.history, self.best_params = models.solve_model_fc_nn(x, y, num, params['fc_nn'], epoches, early_stopping_rounds, self.scally, self.Ncap, True, tune_params = tune_params, random_seed=1, verbose_=verbose, test_size=test_size, start_test_date=start_test_date)
                

        # LSTM
        elif model_name == 'lstm':
            print(f"Начинаю расчет модели {model_name}")
            if self.purpose == 'test':
                epoches = 25
                early_stopping_rounds=epoches
                verbose = 1
                self.df_err, self.model, self.history, self.best_params = models.solve_model_lstm(x, y, num, params['lstm'], epoches, early_stopping_rounds, self.scally, self.Ncap, False, None,  random_seed = 42, verbose_=verbose, test_size=test_size, start_test_date=start_test_date)
            elif self.purpose == 'fit_by_setted_params':
                if epoches is None:
                    epoches = 1500
                else:
                    pass
                if early_stopping_rounds is None:
                    early_stopping_rounds = 50
                else:
                    pass
                verbose = 0
                self.df_err, self.model, self.history, self.best_params = models.solve_model_lstm(x, y, num, params['lstm'], epoches, early_stopping_rounds, self.scally, self.Ncap, False, None,  random_seed = 42, verbose_=verbose, test_size=test_size, start_test_date=start_test_date)
            elif self.purpose == 'tune_params':
                if epoches is None:
                    epoches = 1500
                else:
                    pass
                if early_stopping_rounds is None:
                    early_stopping_rounds = 50
                else:
                    pass
                verbose = 0
                self.df_err, self.model, self.history, self.best_params = models.solve_model_lstm(x, y, num, params['lstm'], epoches, early_stopping_rounds, self.scally, self.Ncap, True, tune_params,  random_seed = 42, verbose_=verbose, test_size=test_size, start_test_date=start_test_date)
        
        # Считаем время выполнения расчета и отображение его в логах
        learning_time = round(float(f'{time.time() - t_initial:3.3f}'), 2)
        if self.purpose=='test':
            if model_name in ['lgbm', 'catboost']:
                print(f'Время выполнения обучения модели "{model_name}" (25 эпох): {learning_time} c.', f'Максимальное время расчета модели cо стандартными гиперпараметрами: {learning_time/25*1000//60} мин {round(learning_time/25*1000, 0)%60} с')    
            if model_name in ['fc_nn', 'lstm']:
                print(f'Время выполнения обучения модели "{model_name}" (25 эпох): {learning_time} c.', f'Максимальное время расчета модели cо стандартными гиперпараметрами: {learning_time/25*1500//60} мин {round(learning_time/25*1500, 0)%60} с')    
        elif self.purpose=='fit_by_setted_params':
            print(f'Время обучения модели "{model_name}" составило: {learning_time//60} мин {learning_time%60} с') 
        elif self.purpose=='tune_params':
            print(f'Время обучения модели "{model_name}" с учетом подбора гиперпараметров составило: {learning_time//60} мин {learning_time%60} с') 
        return self.df_err, self.model, self.history, self.best_params


class Pipeline_wind_forecast(Model):
    """
    Класс Pipeline_wind_forecast, который объединяет в себе методы по формированию обучаемого датафрейма, его обучению и сохранению полученных результатов. Он наследуется от родительского класса Model.
    Методы:
        - prep_all_data - готовит данные по всем ветрякам выбранного ГТП
        - form_dict_prep_data - формирует словарь из ветряков с заготовленными данными для обучения
        - relearn_model - метод переобучает модель с заданием числа эпох и early_stopping_range
        - do_ansamble - производит простейшее ансамблирование (стеккинг) по лучшим моделям из выбранных
        - plot_lerning_process - метод для построения графика процесса обучения (на основании его можно сделать вывод о переобучении модели для конкретного ветряка)
        - get_description - дает описание результатов прогноза
        - upseart_best_params_to_db - загрузка в БД best_params (гиперпараметров), в случае, если модель обучалась с их подбором 
        - save_models - сохранение моделей в виде архива с моделями
        - save_data - сохранение остальных данных модели
    """
    def __init__(self):
        self.default_params_per_model = params
        self.models = dict()
        self.models['models'] = None
        self.models['trained_model'] = dict()
        self.models['best_params'] = dict()
        self.models['df_err'] = dict()
        self.models['history'] = dict()
        self.data = dict()
        self.data['Ncap'] = None
        self.data['all_data'] = None
        self.data['nums'] = dict()
        self.data['scallx'] = dict()
        self.data['scally'] = dict()
        self.data['x_catboost'] = dict()
        self.data['y_catboost'] = dict()
        self.data['x_lgbm'] = dict()
        self.data['y_lgbm'] = dict()
        self.data['x_fc_nn'] = dict()
        self.data['y_fc_nn'] = dict()
        self.data['x_lstm'] = dict()
        self.data['y_lstm'] = dict()
        self.data['results'] = dict()
        self.data['results']['df_err'] = dict()
        self.data['results']['trained_model'] = dict()
        self.data['results']['history'] = dict()
        # self.data['results']['best_params'] = dict()

    def prep_all_data(self, loc_points, eng, name_gtp='GUK_3'):
        """
        Метод для подготовки данных по всем ветрякам данной ГТП 
        """
        super().prep_all_data(loc_points, eng, name_gtp)
        self.eng = eng
        self.data['Ncap'] = self.Ncap
        self.data['all_data'] = self.df_all
        self.data['nums'] = self.nums
        self.name_gtp = name_gtp

    def form_dict_prep_data(self):
        """
        Метод по формированию данных для прогнозирования для всех ветряков выбранногй ГТП по всем моделям
        На выходе:
            self.x_trees, self.y_trees, self.x_fc_nn, self.y_fc_nn, self.x_lstm, self.y_lstm - объекты класса (датафреймы) x_... и y_... - признаки и таргеты для построения моделей предсказания
        """
        for num in self.nums:
            super().prep_data(self.data['all_data'], num)
            self.data['x_catboost'][num] = self.x_trees 
            self.data['y_catboost'][num] = self.y_trees 
            self.data['x_lgbm'][num] = self.x_trees 
            self.data['y_lgbm'][num] = self.y_trees 
            self.data['x_fc_nn'][num] = self.x_fc_nn 
            self.data['y_fc_nn'][num] = self.y_fc_nn 
            self.data['x_lstm'][num] = self.x_lstm 
            self.data['y_lstm'][num] = self.y_lstm 

        self.data['scallx'] = self.scallx 
        self.data['scally'] = self.scally 
        # Удаляем последние (после выполнения функции эти временные параметры будут удлены)
        del self.x_trees, self.y_trees, self.x_fc_nn, self.y_fc_nn, self.x_lstm, self.y_lstm

    def form_dict_fit_predict(self, num_to_fit=None, models=['catboost','lgbm','fc_nn','lstm'], test_size=None, start_test_date=None, purpose='fit_by_setted_params'):
        """
        Метод для прогнозирования выработки ЭЭ всей ГТП
        На вход:
            - num_to_fit - номер ветряка, по которому выполнять подбор гиперпараметров
            - models - модели для прогнозирования 
            - test_size - разделение на выборки тренировочную и тестовую (задается доля тестовой выборки к общему датафрейму от 0 до 1) по дефолту None (задание тестовой выборки выбирается либо test_size либо start_test_date)
            - start_test_date - разделение на выборки тренировочную и тестовую путем задания времени старта тестовых данных по дефолту None (задание тестовой выборки выбирается либо test_size либо start_test_date)
            - purpose: Optional -> ['test', 'fit_by_setted_params', 'tune_params']
                * test - означает, что производится легкое обучение (25 итераций)
                * fit_by_setted_params - означает, что модель обучается с заданными по дефолту параметрами
                * tune_params - означает, что будут подбираться гиперпараметры и создастся отдельная переменная класса self.best_params. На обучение уходит значительно больше времени!!!
        """
        self.settings = dict()
        self.settings['test_size'] = test_size
        self.settings['start_test_date'] = start_test_date
        self.settings['is_ansamble'] = False
        if num_to_fit is None:
            num_to_fit = self.data['nums'][0]
            print(f'Первое обучение производится по ветряку {num_to_fit}')
        self.time_tune = datetime.datetime.now()
        # Проверка ввода
        self.models['models'] = models
        available_models = ['catboost','lgbm','fc_nn','lstm']
        tmp_compare = available_models.copy()
        tmp_compare.extend(models)
        assert len(set(tmp_compare)) == len(available_models), f"В списке models пристутствуют модели, не входящие в список допустимых ['catboost','lgbm','fc_nn','lstm']"
        assert purpose in ['test', 'fit_by_setted_params', 'tune_params'], f'Аргумента функции fit_predict purpose="{purpose}" нет в списке доступных ["test", "fit_by_setted_params", "tune_params"]'

        # Обучаю один ветряк на каждый тип модели
        for model_name in self.models['models']:
            x = self.data[f'x_{model_name}'][num_to_fit]
            y = self.data[f'y_{model_name}'][num_to_fit]
            super().fit_predict(x, y, num_to_fit, params=self.default_params_per_model, model_name=model_name, test_size=test_size, start_test_date=self.settings['start_test_date'], purpose=purpose)
            self.models['df_err'][model_name] = self.df_err
            self.models['trained_model'][model_name] = self.model
            self.models['history'][model_name] = self.history
            self.models['best_params'][model_name] = self.best_params

        # Обучаем всее модели для каждого ветряка
        for num in self.nums:
            self.data['results']['df_err'][num]        = dict()
            self.data['results']['trained_model'][num] = dict()
            self.data['results']['history'][num]       = dict()
            # self.data['results']['best_params'][num]   = dict()

            for model_name in self.models['models']:
                print('------------------------------------------------')
                print()
                print()
                x = self.data[f'x_{model_name}'][num]
                y = self.data[f'y_{model_name}'][num]
                if purpose == 'tune_params':
                    params_to_forecast = self.models['best_params']
                else:
                    params_to_forecast = self.default_params_per_model
                print(f'{num} params of model "{model_name}": ', params_to_forecast[model_name])

                if self.purpose == 'test':
                    super().fit_predict(x, y, num, params_to_forecast, model_name, test_size=test_size, start_test_date=self.settings['start_test_date'], purpose='test')
                else:
                    super().fit_predict(x, y, num, params_to_forecast, model_name, test_size=test_size, start_test_date=self.settings['start_test_date'], purpose='fit_by_setted_params')
                self.data['results']['df_err'][num][model_name]        = self.df_err
                self.data['results']['trained_model'][num][model_name] = self.model
                self.data['results']['history'][num][model_name]       = self.history

        # Удаляем последние (после выполнения функции эти временные параметры будут удлены)
        del self.df_err, self.model, self.history, self.best_params

        # Результаты по всем ветрякам в сумме
        self.data['results']['df_err_windfarm'] = dict()
        dict_sum_errs = {}
        dict_errs = {}
        for model in self.models['models']:
            dict_errs[model] = {}
            dict_sum_errs[model] = pd.DataFrame()
            for num in self.nums:
                dict_errs[model][num] = self.__dict__['data']['results']['df_err'][num][model][[f'N_{num}',f'N_pred_{num}']]
                dict_sum_errs[model] = pd.concat([dict_sum_errs[model], dict_errs[model][num]], axis=1)
            dict_sum_errs[model]['N_sum'] = dict_sum_errs[model][[x for x in dict_sum_errs[model].columns if 'N_pred' not in x]].sum(axis=1)
            dict_sum_errs[model]['N_pred_sum'] = dict_sum_errs[model][[x for x in dict_sum_errs[model].columns if 'N_pred' in x]].sum(axis=1)
            dict_sum_errs[model] = dict_sum_errs[model].reindex(pd.date_range(dict_sum_errs[model].index[0], dict_sum_errs[model].index[-1], freq='1h'))
            dict_sum_errs[model]['N_naiv_sum'] = dict_sum_errs[model]['N_sum'].shift(48)
            dict_sum_errs[model]['err'] = (dict_sum_errs[model]['N_sum'] - dict_sum_errs[model]['N_pred_sum']).abs()*100/self.Ncap
            dict_sum_errs[model]['err_naiv'] = (dict_sum_errs[model]['N_sum'] - dict_sum_errs[model]['N_naiv_sum']).abs()*100/self.Ncap
            dict_sum_errs[model] = dict_sum_errs[model][['N_sum', 'N_pred_sum', 'N_naiv_sum', 'err', 'err_naiv']].dropna()
            self.data['results']['df_err_windfarm'][model] = dict_sum_errs[model]

    def relearn_model(self, model_name, num, new_epoches=2000, new_early_stopping_range=200):
        """
        Метод для переобучения модели
        На входе:
            - model_name - наименование модели
            - num - номер ветряка
            - new_epoches - число эпох
            - new_early_stopping_range - число эпох, в случае, если нет улучшения за этот период, обучение останавливается
        На выходе:
            - данный метод ничего не возвращает, но заменяет данные модели на новые (переобученные)
        """
        available_models = ['catboost','lgbm','fc_nn','lstm']
        assert model_name in available_models, f"Модели нет в список допустимых ['catboost','lgbm','fc_nn','lstm']"
        assert num in self.nums, f"Номера нет в списке допустимых {self.nums}"

        # Обучаю один ветряк на каждый тип модели
        x = self.data[f'x_{model_name}'][num]
        y = self.data[f'y_{model_name}'][num]
        try:
            print("Расчет модели с учетом подбора гиперпараметров")
            super().fit_predict(x, y, num, params=self.models['best_params'], model_name=model_name, epoches=new_epoches, early_stopping_rounds=new_early_stopping_range, test_size=self.settings['test_size'], start_test_date=self.settings['start_test_date'], purpose='fit_by_setted_params')
        except:
            print("Расчет модели без учета подбора гиперпараметров")
            super().fit_predict(x, y, num, params=self.default_params_per_model, model_name=model_name, epoches=new_epoches, early_stopping_rounds=new_early_stopping_range, test_size=self.settings['test_size'], start_test_date=self.settings['start_test_date'], purpose='fit_by_setted_params')
        self.data['results']['df_err'][num][model_name]        = self.df_err
        self.data['results']['trained_model'][num][model_name] = self.model
        self.data['results']['history'][num][model_name]       = self.history
        del self.df_err, self.model, self.history, self.best_params

        # Результаты по всем ветрякам в сумме
        self.data['results']['df_err_windfarm'] = dict()
        dict_sum_errs = {}
        dict_errs = {}
        for model in self.models['models']:
            dict_errs[model] = {}
            dict_sum_errs[model] = pd.DataFrame()
            for num in self.nums:
                dict_errs[model][num] = self.data['results']['df_err'][num][model][[f'N_{num}',f'N_pred_{num}']]
                dict_sum_errs[model] = pd.concat([dict_sum_errs[model], dict_errs[model][num]], axis=1)
            dict_sum_errs[model]['N_sum'] = dict_sum_errs[model][[x for x in dict_sum_errs[model].columns if 'N_pred' not in x]].sum(axis=1)
            dict_sum_errs[model]['N_pred_sum'] = dict_sum_errs[model][[x for x in dict_sum_errs[model].columns if 'N_pred' in x]].sum(axis=1)
            dict_sum_errs[model] = dict_sum_errs[model].reindex(pd.date_range(dict_sum_errs[model].index[0], dict_sum_errs[model].index[-1], freq='1h'))
            dict_sum_errs[model]['N_naiv_sum'] = dict_sum_errs[model]['N_sum'].shift(48)
            dict_sum_errs[model]['err'] = (dict_sum_errs[model]['N_sum'] - dict_sum_errs[model]['N_pred_sum']).abs()*100/self.Ncap
            dict_sum_errs[model]['err_naiv'] = (dict_sum_errs[model]['N_sum'] - dict_sum_errs[model]['N_naiv_sum']).abs()*100/self.Ncap
            dict_sum_errs[model] = dict_sum_errs[model][['N_sum', 'N_pred_sum', 'N_naiv_sum', 'err', 'err_naiv']].dropna()
            self.data['results']['df_err_windfarm'][model] = dict_sum_errs[model]
        if self.settings['is_ansamble']:
            self.do_ansamble()

    # Добавляю ансамбль моделей
    def do_ansamble(self, models=None):
        """
        Введите модели, для их ансамблирования из списка ['catboost','lgbm','fc_nn','lstm']. 
        По сравнению результатов расчетов моделей автоматически выбираются модели, из которых производится ансамблирование.
        В случае принудительного занесения аргумента "models" в функцию, данная опция отключается и ансамблирование производится по заданным пользователем моделям. 
        Далее будут выбираться для ансамблирования модели 
        """
        self.settings['is_ansamble'] = True
        # Если на задавать модели ансамблирования, то функция будет рассматривать используемые модели внутри класса
        if models is not None:
            # models = models
            pass
        else:
            dict_all_errs_mean = pd.DataFrame()
            dict_all_errs = {}
            for num in self.nums:
                dict_errs = {}
                for model in self.models['models']:
                    dict_errs[model] =  round(self.data['results']['df_err'][num][model].describe()[['abs_err']]['mean':'mean'].values[0][0], 3)
                df_abs_errs = pd.DataFrame(dict_errs.values(), index=dict_errs.keys(), columns=['abs_err']).sort_values('abs_err')
                dict_all_errs[num] = df_abs_errs
            for num in self.nums:
                dict_all_errs_mean = pd.concat([dict_all_errs_mean, dict_all_errs[num]], axis=1)
            dict_all_errs_mean['mean_abs_err'] = dict_all_errs_mean.sum(axis=1)
            dict_all_errs_mean = dict_all_errs_mean.sort_values('mean_abs_err')
            min_err = dict_all_errs_mean['mean_abs_err'].min()
            cols_to_ansamble = dict_all_errs_mean[dict_all_errs_mean['mean_abs_err'] <= min_err*1.2].index # Index(['catboost', 'lgbm', 'fc_nn'], dtype='object')
            models = cols_to_ansamble
        print(f'Произвожу ансамблирование по моделям {models}')
        df_err_all = pd.DataFrame()
        for num in self.nums:
            df_err_num = pd.DataFrame()
            for model in models:
                tmp_df = self.data['results']['df_err'][num][model][[f'N_{num}',f'N_pred_{num}']]
                tmp_df.columns = [f'N_{num}_{model}',f'N_pred_{num}_{model}']
                df_err_num = pd.concat([df_err_num, tmp_df], axis=1)
            df_err_num[f'N_{num}'] = df_err_num[[x for x in df_err_num.columns if f'N_pred' not in x]].mean(axis=1)
            df_err_num[f'N_pred_{num}'] = df_err_num[[x for x in df_err_num.columns if f'N_pred' in x]].mean(axis=1)
            df_err_all = pd.concat([df_err_all, df_err_num[[f'N_{num}',f'N_pred_{num}']]], axis=1) 
        df_err_all['N_sum'] = df_err_all[[x for x in df_err_all.columns if 'N_pred' not in x]].sum(axis=1)
        df_err_all['N_pred_sum'] = df_err_all[[x for x in df_err_all.columns if 'N_pred' in x]].sum(axis=1)
        df_err_all = df_err_all.reindex(pd.date_range(df_err_all.index[0], df_err_all.index[-1], freq='1h'))
        df_err_all['N_naiv_sum'] = df_err_all['N_sum'].shift(48)
        df_err_all['err'] = (df_err_all['N_sum'] - df_err_all['N_pred_sum']).abs()*100/self.Ncap
        df_err_all['err_naiv'] = (df_err_all['N_sum'] - df_err_all['N_naiv_sum']).abs()*100/self.Ncap
        df_err_all = df_err_all[['N_sum', 'N_pred_sum', 'N_naiv_sum', 'err', 'err_naiv']].dropna()
        self.data['results']['df_err_windfarm']['ansamble'] = df_err_all

    def plot_lerning_process(self):
        """
        отображение процесса обучения
        На входе:
            - num - номер ветряка
            - from_iter - отобажение с N-й итерации
        На выходе:
            - график с процессом обучения  
        """
        import plotly.express as px
        models = self.models['models']
        dict_of_learning_results = dict()
        for i in range(len(models)):
            model = models[i]
            if model=='catboost':
                tmp_df = pd.DataFrame()
                for num in self.nums:
                    history_train = list(self.data['results']['history'][num][model]['train'].values())[0][20:]
                    history_test = list(self.data['results']['history'][num][model]['test'].values())[0][20:]
                    tmp_df_2 = pd.DataFrame(history_test, columns=[f'{model}_{num}'])
                    tmp_df = pd.concat([tmp_df, tmp_df_2], axis=1)
            elif model=='lgbm':
                tmp_df = pd.DataFrame()
                for num in self.nums:
                    history_test = list(self.data['results']['history'][num][model]['valid_0'].values())[0][20:]
                    tmp_df_2 = pd.DataFrame(history_test, columns=[f'{model}_{num}'])
                    tmp_df = pd.concat([tmp_df, tmp_df_2], axis=1)
            elif model in ['fc_nn','lstm']:
                tmp_df = pd.DataFrame()
                for num in self.nums:
                    history_train = self.data['results']['history'][num][model]['loss'][20:]
                    history_test = self.data['results']['history'][num][model]['val_loss'][20:]
                    tmp_df_2 = pd.DataFrame(history_test, columns=[f'{model}_{num}'])
                    tmp_df = pd.concat([tmp_df, tmp_df_2], axis=1)
            tmp_df.index.name = 'epoches'
            dict_of_learning_results[model] = tmp_df
        for model in dict_of_learning_results.keys():
            fig = px.line(dict_of_learning_results[model], title=model)
            fig.show()
        return dict_of_learning_results

    def get_description(self):
        """
        Метод, формирующий данные для отображения результатов
        На выходе:
            - df_all - описание датайрейма (.describe()) 
            - N - график фактических данных выработки станции и прогнозные значения
            - err - ошибка, осредненная за 24ч 
        """
        import plotly.express as px
        df_all = pd.DataFrame()
        for model in self.data['results']['df_err_windfarm'].keys():
            tmp_df = self.data['results']['df_err_windfarm'][model][['N_sum','N_pred_sum']]
            tmp_df.columns = ['N_sum', f'N_pred_{model}']
            df_all = pd.concat([df_all, tmp_df], axis=1)
        tmp_df = pd.DataFrame(df_all[['N_sum']].mean(axis=1), columns=['N_sum'])
        df_all = df_all.drop(columns=['N_sum'])
        df_all = pd.concat([tmp_df, df_all], axis=1)
        
        cols_to_describe = list(self.data['results']['df_err_windfarm'].keys())
        cols_to_describe.append('naiv')
        df_all['N_pred_naiv'] = df_all['N_sum'].shift(48)
        for model in cols_to_describe:
            df_all[f'err_{model}'] = (df_all['N_sum']-df_all[f'N_pred_{model}']).abs()*100/self.Ncap
        df_all = df_all.dropna()
        N = px.line(df_all[[x for x in df_all.columns if 'N_' in x]], height=500, width=1250, title='N, МВт')
        err = px.line(df_all[[x for x in df_all.columns if 'err_' in x]].rolling(24).mean(), height=500, width=1250, title='err, %')
        return df_all.describe(), N, err

    def upseart_best_params_to_db(self):
        """
        Метод, который сохраняет данные, которые дали лучший результат по модели при подборе гиперпараметров в модели
        """
        df_to_insert = pd.DataFrame([[pd.to_datetime(self.time_tune), self.gtp_id, 48, str(self.models['best_params'][model]).replace('\'', '"'), model, self.data['results']['df_err_windfarm'][model].describe()[['err']].iloc[[1],[0]].values[0][0], len(self.data['results']['df_err_windfarm'][model])] for model in self.models['models']], columns = ['init_date', 'gtp_id', 'hour_distance', 'optuna_data','model_name', 'error', 'count_rows'])
        values =  ','.join([str(i) for i in list(df_to_insert.to_records(index=False))])#.replace('"', "'")
        query_string1 = f"""
        INSERT INTO wind_best_params(init_date, gtp_id, hour_distance, optuna_data, model_name, error, count_rows)
        VALUES {values}
        ON CONFLICT (init_date, gtp_id, hour_distance, model_name)
        DO UPDATE SET optuna_data = excluded.optuna_data
        """
        with self.eng.connect() as connection:
            result = connection.execute(query_string1)
            result.close()
            print(f'upseart_best_params_to_db result.rowcount={result.rowcount}, len(df_to_insert) = {len(df_to_insert)}')
        return result.rowcount == len(df_to_insert)

    def save_models(self):
        """
        Метод, который сохраняет модели на локальный диск в директорию ./results как архив моделей "models_YYYY-MM-DD hh.mm.ss.zip"
        Далее приведены примеры для получения моделей внутри языка python:
            - catboost_model = catboost.CatBoostRegressor() 
            - catboost_model.load_model('03_catboost.h5')
            - lgb_model = lgb.Booster(model_file='03_lgbm.h5')
            - fc_nn_model = tf.keras.models.load_model('03_fc_nn.h5')
            - lstm_model = tf.keras.models.load_model('03_lstm.h5')
        """
        folder = 'results'
        # сохраняю данные в файл zip
        from zipfile import ZipFile
        import os
        import catboost
        import lightgbm as lgb
        import tensorflow as tf

        lst_models = []
        for num in self.data['results']['trained_model'].keys():
            for model in self.models['models']:
                lst_models.append(f'{model}_{num}.h5')
                if model == 'catboost':
                    self.data['results']['trained_model'][num][model].save_model(f'{model}_{num}.h5')
                elif model == 'lgbm':
                    self.data['results']['trained_model'][num][model].save_model(f'{model}_{num}.h5')
                elif model in ['fc_nn', 'lstm']:
                    self.data['results']['trained_model'][num][model].save(f'{model}_{num}.h5')

        name_zip_file = f"models_{str(self.time_tune)[:19].replace(':','.').replace(' ','_')}.zip"
        with ZipFile(name_zip_file, "w") as myzip:
            pass
        with ZipFile(name_zip_file, "a") as myzip:
            for file in lst_models:
                myzip.write(file)
        for file in lst_models:
            os.remove(file)
        path_to_file = f'./{folder}/'+name_zip_file
        os.replace(name_zip_file, path_to_file)
        print(f'Архив с моделями сохранен в директории "{path_to_file}"')
        return path_to_file

    def save_data(self):
        """
        Метод, который сохраняет данные на локальный диск в директорию ./results как файл "data_YYYY-MM-DD hh.mm.ss.dat"
        Далее приведены примеры для получения данных внутри языка python:
        with open('data.dat', 'rb') as f:
            data_to_save = dill.load(f)
        """
        import os
        folder = 'results'
        name_data = f"data_{str(self.__dict__['time_tune'])[:19].replace(':','.').replace(' ','_')}.dat"
        data_to_save = dict()
        for key in self.data.keys():
            if key == 'results':
                data_to_save[key] = dict()
                for key_res in self.data['results'].keys():
                    if key_res == 'trained_model':
                        pass
                    else:
                        data_to_save[key][key_res] = self.data['results'][key_res]
            else:
                data_to_save[key] = self.data[key]
            # Добавляю сюда results
        with open(name_data, 'wb') as f:
            dill.dump(data_to_save, f)
        path_to_file = f'./{folder}/'+name_data
        os.replace(name_data, path_to_file)
        print(f'Файл с данными сохранен в директории "{path_to_file}"')
        return path_to_file

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
    eng = create_engine('postgresql://postgres:achtung@192.168.251.133:5432/fortum_wind')
    pipeline = Pipeline_wind_forecast()
    pipeline.prep_all_data(loc_points, eng, name_gtp='GUK_3')
    pipeline.form_dict_prep_data()
    pipeline.form_dict_fit_predict(pipeline.nums[0], test_size=0.175, purpose='test')
    print(pipeline.__dict__.keys())
    pipeline.plot_lerning_process()