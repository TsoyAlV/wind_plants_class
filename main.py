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
# fitting
# tune_params = {'n_trials': 2,'timeout_secunds': 60*2} # Удалить и раскоментировать после отладки (пока нужно для проверки работы подбора гиперпараметров)
tune_params = {'n_trials': 50,'timeout_secunds': 3600 * 2}

class Model:
    """
    Необходимо выбрать модель среди ['catboost','lgbm','fc_nn','lstm'] а так же оптионально свои параметры настройки и метрику
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
        res = pd.read_sql_query(f"""select gtp_name from wind_gtp """, eng)
        eng.dispose()
        print(list(res.values.astype(str).T[0]))

    def prep_all_data(self, loc_points, eng, name_gtp='GUK_3'):
        # пока не буду прорабатывать данную функцию полностью
        self.df_all, self.nums = first_prep_data(loc_points, 
                                                eng=eng, 
                                                name_gtp = name_gtp, 
                                                start_date=str(datetime.datetime.now()-datetime.timedelta(days=365*3))[:10], 
                                                end_date=str(datetime.datetime.now()-datetime.timedelta(days=1))[:10])
        res = pd.read_sql_query(f"""select gtp_power, gtp_id from wind_gtp where gtp_name = '{name_gtp}'""", eng)
        assert res.shape == (1, 2), f'Не удалось найти конкретную выработку Ncap для дальнейшего расчета погрешности. Должно быть res.shape == (1, 2) а получилось {res.shape}'
        self.Ncap = res.values[0][0]
        self.gtp_id = int(res.values[0][1])
        eng.dispose()    

    def prep_data(self, df_all, num):
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

    def fit_predict(self, x, y, num, params, model_name, epoches=1500, early_stopping_rounds=50, test_size=0.175, start_test_date=None, purpose='fit_by_setted_params'):
        """
        purpose: Optional -> ['test', 'fit_by_setted_params', 'tune_params']\n
            test - означает, что производится легкое обучение (25 итераций)\n
            fit_by_setted_params - означает, что модель обучается с заданными по дефолту параметрами\n
            tune_params - означает, что будут подбираться гиперпараметры и создастся отдельная переменная класса self.best_params. На обучение уходит значительно больше времени!!!
        """
        if num is None:
            num = self.data['nums'][0]
            print(f'Первое обучение производится по ветряку {num}')
        assert purpose in ['test', 'fit_by_setted_params', 'tune_params'], f'Аргумента функции fit_predict purpose="{purpose}" нет в списке доступных ["test", "fit_by_setted_params", "tune_params"]'
        self.purpose=purpose
        t_initial = time.time()
        self.time_tune = datetime.datetime.now()
        # catboost
        if model_name == 'catboost':
            print(f"Начинаю расчет модели {model_name}")
            if self.purpose == 'test':
                epoches = 250
                verbose = 10
                self.df_err, self.model, self.history, self.best_params = models.solve_model_catboost(x, y, num, params['catboost'], epoches, early_stopping_rounds, self.scally, self.Ncap, False, None, verbose, test_size=test_size, start_test_date=start_test_date)
            elif self.purpose == 'fit_by_setted_params':
                epoches = 1000
                verbose = 0
                self.df_err, self.model, self.history, self.best_params = models.solve_model_catboost(x, y, num, params['catboost'], epoches, early_stopping_rounds, self.scally, self.Ncap, False, None, verbose, test_size=test_size, start_test_date=start_test_date)
            elif self.purpose == 'tune_params':
                epoches = 1000
                verbose = 0
                self.df_err, self.model, self.history, self.best_params = models.solve_model_catboost(x, y, num, params['catboost'], epoches, early_stopping_rounds, self.scally, self.Ncap, True, tune_params, verbose, test_size=test_size, start_test_date=start_test_date)

        # lgbm
        elif model_name == 'lgbm':
            print(f"Начинаю расчет модели {model_name}")
            if self.purpose == 'test':
                epoches = 250
                verbose = 10
                self.df_err, self.model, self.history, self.best_params = models.solve_model_lgbm(x, y, num, params['lgbm'], epoches, early_stopping_rounds, self.scally, self.Ncap, False, 
                                                                                None, verbose, test_size=test_size, start_test_date=start_test_date)
            elif self.purpose == 'fit_by_setted_params':
                epoches = 1000
                verbose = 0
                self.df_err, self.model, self.history, self.best_params = models.solve_model_lgbm(x, y, num, params['lgbm'], epoches, early_stopping_rounds, self.scally, self.Ncap, False, 
                                                                                None, verbose, test_size=test_size, start_test_date=start_test_date)

            elif self.purpose == 'tune_params':
                epoches = 1000
                verbose = 0
                self.df_err, self.model, self.history, self.best_params = models.solve_model_lgbm(x, y, num, params['lgbm'], epoches, early_stopping_rounds, self.scally, self.Ncap, True, 
                                                                                tune_params, verbose, test_size=test_size, start_test_date=start_test_date)


        # fc_nn
        elif model_name == 'fc_nn':
            print(f"Начинаю расчет модели {model_name}")
            if self.purpose == 'test':
                epoches = 250
                verbose = 1
                self.df_err, self.model, self.history, self.best_params = models.solve_model_fc_nn(x, y, num, params['fc_nn'], epoches, early_stopping_rounds, self.scally, self.Ncap, False, None, 1,verbose, test_size=test_size, start_test_date=start_test_date)
            elif self.purpose == 'fit_by_setted_params':
                epoches = 1500
                verbose = 0
                self.df_err, self.model, self.history, self.best_params = models.solve_model_fc_nn(x, y, num, params['fc_nn'], epoches, early_stopping_rounds, self.scally, self.Ncap, False, None, 1,verbose, test_size=test_size, start_test_date=start_test_date)
            elif self.purpose == 'tune_params':
                epoches = 1500
                verbose = 0
                self.df_err, self.model, self.history, self.best_params = models.solve_model_fc_nn(x, y, num, params['fc_nn'], epoches, early_stopping_rounds, self.scally, self.Ncap, True, tune_params = tune_params, random_seed=1, verbose_=verbose, test_size=test_size, start_test_date=start_test_date)
                

        # LSTM
        elif model_name == 'lstm':
            print(f"Начинаю расчет модели {model_name}")
            if self.purpose == 'test':
                epoches = 250
                verbose = 1
                self.df_err, self.model, self.history, self.best_params = models.solve_model_lstm(x, y, num, params['lstm'], epoches, early_stopping_rounds, self.scally, self.Ncap, False, None,  random_seed = 42, verbose_=verbose, test_size=test_size, start_test_date=start_test_date)
            elif self.purpose == 'fit_by_setted_params':
                epoches = 1500
                verbose = 0
                self.df_err, self.model, self.history, self.best_params = models.solve_model_lstm(x, y, num, params['lstm'], epoches, early_stopping_rounds, self.scally, self.Ncap, False, None,  random_seed = 42, verbose_=verbose, test_size=test_size, start_test_date=start_test_date)
            elif self.purpose == 'tune_params':
                epoches = 1500
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
    Необходимо выбрать модель среди ['catboost','lgbm','fc_nn','lstm'] а так же оптионально свои параметры настройки и метрику
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
        super().prep_all_data(loc_points, eng, name_gtp)
        self.eng = eng
        self.data['Ncap'] = self.Ncap
        self.data['all_data'] = self.df_all
        self.data['nums'] = self.nums
        self.name_gtp = name_gtp

    def form_dict_prep_data(self):
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
        purpose: Optional -> ['test', 'fit_by_setted_params', 'tune_params']\n
            test - означает, что производится легкое обучение (25 итераций)\n
            fit_by_setted_params - означает, что модель обучается с заданными по дефолту параметрами\n
            tune_params - означает, что будут подбираться гиперпараметры и создастся отдельная переменная класса self.best_params. На обучение уходит значительно больше времени!!!
        """
        self.settings = dict()
        self.settings['test_size'] = test_size
        self.settings['start_test_date'] = start_test_date
        self.settings['is_ansamble'] = False
        if num_to_fit is None:
            num_to_fit = self.data['nums'][0]
            print(f'Первое обучение производится по ветряку {num_to_fit}')
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
                # if purpose in ['test', 'fit_by_setted_params']:
                #     self.data['results']['best_params'][num][model_name] = None
                # else:
                #     self.data['results']['best_params'][num]   = params_to_forecast

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
        available_models = ['catboost','lgbm','fc_nn','lstm']
        assert model_name in available_models, f"Модели нет в список допустимых ['catboost','lgbm','fc_nn','lstm']"
        assert num in self.nums, f"Номера нет в списке допустимых {self.nums}"

        # Обучаю один ветряк на каждый тип модели
        x = self.data[f'x_{model_name}'][num]
        y = self.data[f'y_{model_name}'][num]
        if self.purpose == 'tune_params':
            super().fit_predict(x, y, num, params=self.models['best_params'], model_name=model_name, epoches=new_epoches, early_stopping_rounds=new_early_stopping_range, test_size=self.settings['test_size'], start_test_date=self.settings['start_test_date'], purpose='fit_by_setted_params')
        else:
            super().fit_predict(x, y, num, params=self.default_params_per_model, model_name=model_name, epoches=new_epoches, early_stopping_rounds=new_early_stopping_range, test_size=self.settings['test_size'], start_test_date=self.settings['start_test_date'], purpose='fit_by_setted_params')
        self.data['results']['df_err'][num][model_name]        = self.df_err
        self.data['results']['trained_model'][num][model_name] = self.model
        self.data['results']['history'][num][model_name]       = self.history
        del self.df_err, self.model, self.history

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
        Введите модели, для их ансамблирования из списка ['catboost','lgbm','fc_nn','lstm']. \n
        По сравнению результатов расчетов моделей автоматически выбираются модели, из которых производится ансамблирование.\n
        В случае принудительного занесения аргумента "models" в функцию, данная опция отключается и ансамблирование производится по заданным пользователем моделям. \n
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

    def plot_lerning_process(self, num, from_iter=1):
        import matplotlib.pyplot as plt
        for model in self.models['models']:
            print(f'            -------   model: {model}, num: {num}     -------')
            if model=='catboost':
                history_train = list(self.data['results']['history'][num][model]['train'].values())[0][from_iter:]
                history_test = list(self.data['results']['history'][num][model]['test'].values())[0][from_iter:]
                plt.plot(history_train)
                plt.plot(history_test)
                plt.legend(['train','valid'])
                plt.xlabel('Номер итерации')
                plt.ylabel(f"Ошибка ({list(self.data['results']['history'][num][model]['test'].keys())[0]})")
                plt.grid()
                plt.show()
            elif model=='lgbm':
                history_test = list(self.data['results']['history'][num][model]['valid_0'].values())[0][from_iter:]
                plt.plot(history_test)
                plt.legend(['valid'])
                plt.xlabel('Номер итерации')
                plt.ylabel(f"Ошибка ({list(self.data['results']['history'][num]['lgbm']['valid_0'].keys())[0]})")
                plt.grid()
                plt.show()
            elif model in ['fc_nn','lstm']:
                history_train = self.data['results']['history'][num][model]['loss'][from_iter:]
                history_test = self.data['results']['history'][num][model]['val_loss'][from_iter:]
                plt.plot(history_train)
                plt.plot(history_test)
                plt.legend(['train','valid'])
                plt.xlabel('Номер итерации')
                if self.models['best_params'][model] is None:
                    loss_func = self.default_params_per_model[model]['loss_func']
                else:
                    loss_func = self.models['best_params'][model]['loss_func']
                plt.ylabel(f"Ошибка ({loss_func})")
                plt.grid()
                plt.show()
            print()

    def describe(self, df_err=True, plots=True):
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
        if df_err:
            print(f'                            --------- Результат расчетов моделей прогнозирования ВЭС {self.name_gtp} --------- ')
            try:
                display(df_all.describe())
            except:
                print(df_all.describe())
            print()
        if plots:
            N = px.line(df_all[[x for x in df_all.columns if 'N_' in x]], height=500, width=1250, title='N, МВт')
            err = px.line(df_all[[x for x in df_all.columns if 'err_' in x]].rolling(24).mean(), height=500, width=1250, title='err, %')
            print(f'                            --------- Фактические и прогнозные данные выработки ЭЭ по ВЭС "{self.name_gtp}" --------- ')
            N.show()
            print()
            print(f'                            --------- График ошибок предсказания ЭЭ по ВЭС "{self.name_gtp}" по моделям --------- ')
            err.show()

    def upseart_best_params_to_db(self):
        df_to_insert = pd.DataFrame([[pd.to_datetime(self.time_tune), self.gtp_id, 48, str(self.models['best_params'][model]).replace('\'', '"'), model] for model in self.models['models']], columns = ['init_date', 'gtp_id', 'hour_distance', 'model_name', 'optuna_data'])
        values =  ','.join([str(i) for i in list(df_to_insert.to_records(index=False))])#.replace('"', "'")
        query_string1 = f"""
        INSERT INTO wind_best_params(init_date, gtp_id, hour_distance, optuna_data, model_name)
        VALUES {values}
        ON CONFLICT (init_date, gtp_id, hour_distance, model_name)
        DO UPDATE SET optuna_data = excluded.optuna_data
        """
        with self.eng.connect() as connection:
            result = connection.execute(query_string1)
            result.close()
            print(f'upseart_best_params_to_db result.rowcount={result.rowcount}, len(df_to_insert) = {len(df_to_insert)}')
        return result.rowcount == len(df_to_insert)


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
    pipeline = Pipeline_wind_forecast(['catboost','lgbm','fc_nn','lstm'])
    pipeline.prep_all_data(loc_points, eng, name_gtp='GUK_3')
    pipeline.form_dict_prep_data()
    pipeline.form_dict_fit_predict(pipeline.nums[0], test_size=0.175, purpose='test')
    print(pipeline.__dict__.keys())
    pipeline.plot_lerning_process('06', 3)