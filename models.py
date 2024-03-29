from sklearn.model_selection import train_test_split
import tensorflow as tf
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
import lightgbm as lgb
import time
import sklearn
import optuna
import os


def train_test_split_by_date(x, y, start_date):
    """
    Функция, которая делит датафрейм на трейн и тест части
    На входе: 
        - x - датафрейм с признаками
        - y - таргет
        - start_date - дата-время - старт тестовых данных
    """
    ind = y[:start_date].shape[0]
    X_traintest = x[:ind-1]
    X_holdout   = x[ind:]
    y_traintest = y[:ind-1]
    y_holdout   = y[ind:]
    return X_traintest, X_holdout, y_traintest, y_holdout

def solve_model_fc_nn(x, y, num, params, epoches, early_stopping_rounds, scally, Ncap, tune=False, tune_params = {'n_trials':50, 'timeout_secunds':3600*3}, random_seed = 42, verbose_=1, test_size=None, start_test_date=None):
    """
    Функция для обучения модели нейронной полносвязной сети
    На входе:
        - x - датайрейм с признаками для обучения модели
        - y - датайрейм с таргетами для обучения модели
        - num - номер ветряка 
        - params - гиперпараметры для прогнозирования (заданные). В случае, если tune=True не будут учитываться
        - epoches - число эпох предсказания
        - early_stopping_rounds - число эпох, в случае, если нет улучшения за этот период, обучение останавливается
        - scally - шкалирование таргета
        - Ncap - коэффициент для оценки точности (обычно максимальная выработка по всем ветрякам ГТП)
        - tune - является ли тип расчета с подбором гиперпараметров
        - tune_params - параметры характеризуют, какое количество или сколько времени уделить на подбор гиперпараметров для данной модели
        - random_seed - функция создания модели прогнозирования задает фиксированное значение random seed. Это необходимо для более контролируемого процесса получения и сравнения результатов обучения модели
        - verbose_ - отображать процесс обучения или нет (0 - не отображать)
        - test_size - размер тестовой выборки (опционально - выбрать либо test_size либо start_test_date)
        - start_test_date - старт тестирования данных (опционально - выбрать либо test_size либо start_test_date)
    """
    os.environ['PYTHONHASHSEED'] = str(random_seed)

    # Setting the seed for numpy-generated random numbers
    np.random.seed(random_seed)

    # Setting the graph-level random seed.
    tf.random.set_seed(random_seed)
    if test_size is not None:
        X_traintest, X_holdout, y_traintest, y_holdout = train_test_split(x, y, shuffle=False, test_size=test_size)
    elif start_test_date is not None:
        X_traintest, X_holdout, y_traintest, y_holdout = train_test_split_by_date(x, y, start_test_date)
    t_initial0 = time.time()
    X_train, X_test, y_train, y_test = train_test_split(X_traintest, y_traintest,
                                                        test_size=test_size,
                                                        random_state=42,
                                                        shuffle=False)
    if tune:
        def objective(trial):
            best_params = {
                            'activate1': trial.suggest_categorical('activate1',['selu','relu','linear']),
                            'activate2': trial.suggest_categorical('activate2',['selu','relu','linear']),
                            'activate3': trial.suggest_categorical('activate3',['selu','relu','linear']),
                            'activate4': trial.suggest_categorical('activate4',['selu','relu','linear']),
                            'activate5': trial.suggest_categorical('activate5',['selu','relu','linear']),
                            'units0': trial.suggest_int('units0', 40,160),
                            'units1': trial.suggest_int('units1', 15,80),
                            'units2': trial.suggest_int('units2', 4,45),
                            'add_layer': trial.suggest_categorical('add_layer',[True, False]),
                            'add_units3': trial.suggest_int('add_units3', 4,45),
                            'loss_func': trial.suggest_categorical('loss_func', ['mae','mse'])
                            }
            t_initial = time.time()
            activate1 = best_params['activate1']
            activate2 = best_params['activate2']
            activate3 = best_params['activate3']
            activate4 = best_params['activate4']
            activate5 = best_params['activate5']
            units0 = best_params['units0']
            units1 = best_params['units1']
            units2 = best_params['units2']
            add_layer = best_params['add_layer']
            add_units3 = best_params['add_units3']
            loss_func = best_params['loss_func']

            earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=early_stopping_rounds, verbose=0, mode='min')
            mcp_save = tf.keras.callbacks.ModelCheckpoint('tmp_callback.hdf5', save_best_only=True, monitor='val_loss', mode='min')
            reduce_lr_loss = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=33, verbose=0, mode='min')
            callbacks = [earlyStopping, mcp_save, reduce_lr_loss]

            nn1 = tf.keras.models.Sequential()
            nn1.add(tf.keras.layers.Input(shape=(X_train.shape[1:])))
            nn1.add(tf.keras.layers.Dense(units=units0,
                                        activation=activate1,
                                        input_shape = X_train.shape[1:]))
            nn1.add(tf.keras.layers.Dense(units=units1,
                                        activation=activate2))
            nn1.add(tf.keras.layers.BatchNormalization())
            nn1.add(tf.keras.layers.Dense(units=units2,
                                        activation=activate5))
            if add_layer:
                nn1.add(tf.keras.layers.Dense(units=add_units3,
                                              activation=activate3))
            nn1.add(tf.keras.layers.Dense(units=24,
                                        activation=activate4))

            weights = nn1.weights
            zeros_weights = []
            rng=np.random.RandomState(42)

            for i in range(len(weights)):
                shape = weights[i].shape
                zeros_weights.append(rng.rand(*shape))
            nn1.set_weights(zeros_weights)
            opt = tf.optimizers.Adam(learning_rate=1e-2)
            nn1.compile(optimizer=opt,
                    loss = loss_func)
            nn1.fit(X_train,
                    y_train,
                    batch_size=X_train.shape[0]//10,
                    epochs=epoches,
                    verbose=0,
                    callbacks=callbacks, # Early stopping
                    validation_data=(X_test, y_test))
            res = nn1.predict(X_test)
            targ = y_test.copy()
            result = np.mean(np.mean(np.abs(targ-res)))*100/Ncap
            # res = nn1.evaluate(df_test_x1, y_test)
            print(f'Полученная ошибка - {round(result,4)}, затрачено на 1 итерацию {time.time() - t_initial:3.3f} c')    
            return result
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=tune_params['n_trials'], 
                       timeout=tune_params['timeout_secunds'])
        activate1 = study.best_params['activate1']      
        activate2 = study.best_params['activate2']      
        activate3 = study.best_params['activate3']      
        activate4 = study.best_params['activate4']      
        activate5 = study.best_params['activate5']      
        units0 = study.best_params['units0']            
        units1 = study.best_params['units1']            
        units2 = study.best_params['units2']            
        add_layer = study.best_params['add_layer']      
        add_units3 = study.best_params['add_units3']   
        loss_func = study.best_params['loss_func']
        best_params = study.best_params
        print("Подбор гиперпараметров окончен.")
        print('best_params=', best_params)
    else:
        activate1 = params['activate1']      
        activate2 = params['activate2']      
        activate3 = params['activate3']      
        activate4 = params['activate4']      
        activate5 = params['activate5']      
        
        units0 = params['units0']            
        units1 = params['units1']            
        units2 = params['units2']            
        add_layer = params['add_layer']      
        add_units3 = params['add_units3']   
        loss_func = params['loss_func']
        best_params = None

    earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=early_stopping_rounds, verbose=0, mode='min')
    mcp_save = tf.keras.callbacks.ModelCheckpoint('tmp_callback.hdf5', save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr_loss = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=33, verbose=0, mode='min')
    callbacks = [earlyStopping, mcp_save, reduce_lr_loss]

    nn1 = tf.keras.models.Sequential()
    nn1.add(tf.keras.layers.Input(shape=(X_train.shape[1:])))

    nn1.add(tf.keras.layers.Dense(units=units0,
                                  activation=activate1,
                                  input_shape = X_train.shape[1:]))
    nn1.add(tf.keras.layers.Dense(units=units1,
                                  activation=activate2))
    nn1.add(tf.keras.layers.BatchNormalization())
    nn1.add(tf.keras.layers.Dense(units=units2,
                                  activation=activate5))
    if add_layer:
        nn1.add(tf.keras.layers.Dense(units=add_units3,
                                      activation=activate3))
    nn1.add(tf.keras.layers.Dense(units=24,
                                  activation=activate4))

    weights = nn1.weights
    zeros_weights = []
    rng=np.random.RandomState(random_seed)

    for i in range(len(weights)):
        shape = weights[i].shape
        zeros_weights.append(rng.rand(*shape))
    nn1.set_weights(zeros_weights)
    opt = tf.optimizers.Adam(learning_rate=1e-2)
    nn1.compile(optimizer=opt,
               loss = loss_func)
    history = nn1.fit(X_train,
            y_train,
            batch_size=500,
            epochs=epoches,
            verbose=verbose_,
            callbacks=callbacks, # Early stopping
            validation_data=(X_test, y_test))
    
    y_holdout_pred = nn1.predict(X_holdout)
    df_pred_holdout = pd.DataFrame(data = y_holdout_pred, 
                                    index=X_holdout.index,)
                          # columns = [f'N_pred_{num}'])
    df_err = pd.concat([y_holdout,df_pred_holdout], axis=1)
    df_err = df_err[df_err.index.hour == 0]
    daterange = []
    for date in df_err.index:
        tmp_lst = [date+pd.to_timedelta(f'{hour}h') for hour in range(24)]
        daterange.extend(tmp_lst)
    tmp_lst = []
    for i in range(24):
        tmp_lst.extend([f'N_targ_{i}', i])
    df_err = df_err[tmp_lst]
    df_err = pd.DataFrame(np.array(df_err).reshape(-1,2), index=daterange, columns=[f'N_{num}',f'N_pred_{num}'])    
    df_err = df_err.reindex(daterange) 
    df_err[f'N_{num}'] = scally.inverse_transform(df_err[[f'N_{num}']])
    df_err[f'N_pred_{num}'] = scally.inverse_transform(df_err[[f'N_pred_{num}']])
    max_N = round(df_err[f'N_{num}'].max(),2)
    df_err[f'N_pred_{num}'] = df_err[f'N_pred_{num}'].apply(lambda x: max_N if (x>= max_N) else x)
    df_err = df_err.reindex(pd.date_range(df_err.index[0], df_err.index[-1],freq='1h'))
    df_err[f'N_naiv_{num}'] = df_err[f'N_{num}'].shift(48)
    df_err['err'] = (df_err[f'N_{num}']-df_err[f'N_pred_{num}'])
    df_err['naiv_err'] = df_err[f'N_{num}'] - df_err[f'N_naiv_{num}']
    df_err['abs_err']  = df_err['err'].abs()/Ncap*100
    df_err['abs_naiv_err'] = df_err['naiv_err'].abs()/Ncap*100
    # df_err = df_err.dropna()
    print(f'Ошибка прогноза составляет {round(df_err.abs_err.mean(), 3)}')
    print(f'Ошибка наивной модели составляет {round(df_err.abs_naiv_err.mean(), 3)}')
    return df_err, nn1, history.history, best_params

def solve_model_lstm(x, y, num, params, epoches, early_stopping_rounds, scally, Ncap, 
                     tune = False, 
                     tune_params = { 'n_trials': 50,'timeout_secunds': 3600 * 3 }, 
                     random_seed = 42, 
                     verbose_=0, 
                     test_size=None, 
                     start_test_date=None):
    """
    Функция для обучения модели LSTM типа МО нейронными сетями
    На входе:
        - x - датайрейм с признаками для обучения модели
        - y - датайрейм с таргетами для обучения модели
        - num - номер ветряка 
        - params - гиперпараметры для прогнозирования (заданные). В случае, если tune=True не будут учитываться
        - epoches - число эпох предсказания
        - early_stopping_rounds - число эпох, в случае, если нет улучшения за этот период, обучение останавливается
        - scally - шкалирование таргета
        - Ncap - коэффициент для оценки точности (обычно максимальная выработка по всем ветрякам ГТП)
        - tune - является ли тип расчета с подбором гиперпараметров
        - tune_params - параметры характеризуют, какое количество или сколько времени уделить на подбор гиперпараметров для данной модели
        - random_seed - функция создания модели прогнозирования задает фиксированное значение random seed. Это необходимо для более контролируемого процесса получения и сравнения результатов обучения модели
        - verbose_ - отображать процесс обучения или нет (0 - не отображать)
        - test_size - размер тестовой выборки (опционально - выбрать либо test_size либо start_test_date)
        - start_test_date - старт тестирования данных (опционально - выбрать либо test_size либо start_test_date)
    """
    os.environ['PYTHONHASHSEED'] = str(random_seed)

    # Setting the seed for numpy-generated random numbers
    np.random.seed(random_seed)

    # Setting the graph-level random seed.
    tf.random.set_seed(random_seed)

    if test_size is not None:
        train_test_range = int(len(x)*(1-test_size))
        X_traintest, X_holdout, y_traintest, y_holdout = x[:train_test_range], x[train_test_range:], y[:train_test_range], y[train_test_range:]    
    elif start_test_date is not None:
        X_traintest, X_holdout, y_traintest, y_holdout = train_test_split_by_date(x, y, start_test_date)
    t_initial0 = time.time()
    train_range = int(len(X_traintest)*(0.8))
    X_train, X_test, y_train, y_test = X_traintest[:train_range], X_traintest[train_range:], y_traintest[:train_range], y_traintest[train_range:]
    if tune:
        def objective(trial):
            t_initial = time.time()
            earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=early_stopping_rounds, verbose=0, mode='min')
            reduce_lr_loss = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=33, verbose=0, mode='min')
            callbacks = [earlyStopping, reduce_lr_loss]
            best_params = {
                'units0': trial.suggest_int('units0', 20, 80),
                'activate1': trial.suggest_categorical('activate1', ['sigmoid']),
                'activate3': trial.suggest_categorical('activate3', ['selu','relu','linear']),
                'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.03),
                'loss_func': trial.suggest_categorical('loss_func', ['mae','mse'])
            }
            units0 = best_params['units0']
            activate1 = best_params['activate1']
            learning_rate0 = best_params['learning_rate']
            loss_func0 = best_params['loss_func']
            activate3 = best_params['activate3']
            nn1 = tf.keras.models.Sequential()
            nn1.add(tf.keras.layers.LSTM(units=units0, activation=activate1, input_shape = X_train.shape[1:]))
            nn1.add(tf.keras.layers.Dense(units=24, activation=activate3))
            opt = tf.optimizers.Adam(learning_rate=learning_rate0)
            nn1.compile(optimizer=opt,
                    loss = loss_func0)
            
            weights = nn1.weights
            zeros_weights = []
            rng=np.random.RandomState(random_seed)
            for i in range(len(weights)):
                shape = weights[i].shape
                zeros_weights.append(rng.rand(*shape))
            nn1.set_weights(zeros_weights)
            nn1.fit(X_train, y_train, batch_size=X_train.shape[0]//10, epochs=epoches, verbose=verbose_, callbacks=callbacks, validation_data=(X_test, y_test))
            res = nn1.predict(X_test)
            targ = y_test.copy()
            result = np.mean(np.mean(np.abs(targ-res)))*100/Ncap
            # res = nn1.evaluate(df_test_x1, df_test_y1)
            print(f'Полученная ошибка - {round(result,4)}, затрачено на 1 итерацию {time.time() - t_initial:3.3f} c')    
            return result
        
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=tune_params['n_trials'], 
                        timeout=tune_params['timeout_secunds'])
        best_params = study.best_params
        units0 = best_params['units0']
        activate1 = best_params['activate1']
        activate3 = best_params['activate3']
        learning_rate0 = best_params['learning_rate']
        loss_func0 = best_params['loss_func']
        print("Подбор гиперпараметров окончен.")
        print('best_params=', best_params)
    else:
        units0 = params['units0']
        activate1 = params['activate1']
        activate3 = params['activate3']
        learning_rate0 = params['learning_rate']
        loss_func0 = params['loss_func']
        best_params = None
    earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=early_stopping_rounds, verbose=0, mode='min')
    mcp_save = tf.keras.callbacks.ModelCheckpoint('tmp_callback.hdf5', save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr_loss = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=33, verbose=0, mode='min')
    callbacks = [earlyStopping, mcp_save, reduce_lr_loss]

    nn1 = tf.keras.models.Sequential()
    nn1.add(tf.keras.layers.LSTM(units=units0, activation=activate1, input_shape = X_train.shape[1:]))
    nn1.add(tf.keras.layers.Dense(units=24, activation=activate3))
    opt = tf.optimizers.Adam(learning_rate=learning_rate0)
    nn1.compile(optimizer=opt,
            loss = loss_func0)
    
    weights = nn1.weights
    zeros_weights = []
    rng=np.random.RandomState(random_seed)
    for i in range(len(weights)):
        shape = weights[i].shape
        zeros_weights.append(rng.rand(*shape))
    nn1.set_weights(zeros_weights)
    history = nn1.fit(X_train, y_train, batch_size=X_train.shape[0]//10, epochs=epoches, verbose=verbose_, callbacks=callbacks, validation_data=(X_test, y_test))
    y_holdout_pred = nn1.predict(X_holdout)
    df_pred_holdout = pd.DataFrame(data = y_holdout_pred, index=y_holdout.index)
                          # columns = [f'N_pred_{num}'])
    df_err = pd.concat([y_holdout, df_pred_holdout], axis=1)
    df_err = df_err[df_err.index.hour == 0]
    daterange = []
    for date in df_err.index:
        tmp_lst = [date+pd.to_timedelta(f'{hour}h') for hour in range(24)]
        daterange.extend(tmp_lst)
    tmp_lst = []
    for i in range(24):
        tmp_lst.extend([f'N_targ_{i}', i])
    df_err = df_err[tmp_lst]
    df_err = pd.DataFrame(np.array(df_err).reshape(-1,2), index=daterange, columns=[f'N_{num}',f'N_pred_{num}'])    
    df_err = df_err.reindex(daterange) 
    df_err[f'N_{num}'] = scally.inverse_transform(df_err[[f'N_{num}']])
    df_err[f'N_pred_{num}'] = scally.inverse_transform(df_err[[f'N_pred_{num}']])
    max_N = round(df_err[f'N_{num}'].max(),2)
    df_err[f'N_pred_{num}'] = df_err[f'N_pred_{num}'].apply(lambda x: max_N if (x>= max_N) else x)
    df_err = df_err.reindex(pd.date_range(df_err.index[0], df_err.index[-1],freq='1h'))
    df_err = df_err.reindex(pd.date_range(df_err.index[0], df_err.index[-1],freq='1h'))
    df_err[f'N_naiv_{num}'] = df_err[f'N_{num}'].shift(48)
    df_err['err'] = (df_err[f'N_{num}']-df_err[f'N_pred_{num}'])
    df_err['naiv_err'] = df_err[f'N_{num}'] - df_err[f'N_naiv_{num}']
    df_err['abs_err']  = df_err['err'].abs()/Ncap*100
    df_err['abs_naiv_err'] = df_err['naiv_err'].abs()/Ncap*100
    # df_err = df_err.dropna()
    print(f'Ошибка прогноза составляет {round(df_err.abs_err.mean(), 3)}')
    print(f'Ошибка наивной модели составляет {round(df_err.abs_naiv_err.mean(), 3)}')
    return df_err, nn1, history.history, best_params

def solve_model_catboost(x,y, num, params, epoches, early_stopping_rounds, scally, Ncap, 
                     tune = False, 
                     tune_params = { 'n_trials': 50,'timeout_secunds': 3600 * 3 },
                     verbose_=0, 
                     test_size=None, 
                     start_test_date=None):
    """
    Функция для обучения модели CatBoost типа МО - бустинг 
    На входе:
        - x - датайрейм с признаками для обучения модели
        - y - датайрейм с таргетами для обучения модели
        - num - номер ветряка 
        - params - гиперпараметры для прогнозирования (заданные). В случае, если tune=True не будут учитываться
        - epoches - число эпох предсказания
        - early_stopping_rounds - число эпох, в случае, если нет улучшения за этот период, обучение останавливается
        - scally - шкалирование таргета
        - Ncap - коэффициент для оценки точности (обычно максимальная выработка по всем ветрякам ГТП)
        - tune - является ли тип расчета с подбором гиперпараметров
        - tune_params - параметры характеризуют, какое количество или сколько времени уделить на подбор гиперпараметров для данной модели
        - random_seed - функция создания модели прогнозирования задает фиксированное значение random seed. Это необходимо для более контролируемого процесса получения и сравнения результатов обучения модели
        - verbose_ - отображать процесс обучения или нет (0 - не отображать)
        - test_size - размер тестовой выборки (опционально - выбрать либо test_size либо start_test_date)
        - start_test_date - старт тестирования данных (опционально - выбрать либо test_size либо start_test_date)
    """
    t_initial0 = time.time()
    if test_size is not None:
        X_traintest, X_holdout, y_traintest, y_holdout = train_test_split(x, y, shuffle=False, test_size=test_size)
    elif start_test_date is not None:
        X_traintest, X_holdout, y_traintest, y_holdout = train_test_split_by_date(x, y, start_test_date)
    X_train, X_test, y_train, y_test = train_test_split(X_traintest, y_traintest,
                                                        test_size=0.20,
                                                        random_state=42,
                                                        shuffle=False)
    train_pool = Pool(X_train, label=y_train)
    test_pool = Pool(X_test, label=y_test)

    if tune:
        def objective(trial):
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.03),
                'l2_leaf_reg':   trial.suggest_float('l2_leaf_reg', 0.1, 10, log=True),
                'depth':         trial.suggest_int('depth', 3,10),
                'loss_func':     trial.suggest_categorical('loss_func', ['MAE','MAPE','RMSE'])
            }
            model_with_early_stop = CatBoostRegressor(
                iterations=epoches,
                random_seed=63,
                learning_rate=params['learning_rate'],
                l2_leaf_reg=params['l2_leaf_reg'],
                depth = params['depth'],
                loss_function=params['loss_func'],
                early_stopping_rounds=early_stopping_rounds
            )

            model_with_early_stop.fit(
                train_pool,
                eval_set=test_pool,
                verbose=verbose_,
                plot=False,
                early_stopping_rounds=early_stopping_rounds
            )
            # err = model_with_early_stop.best_score_['validation'][params['loss_func']]
            err = model_with_early_stop.predict(X_test)
            targ = y_test.copy()
            result = np.mean(np.mean(np.abs(targ-np.array([err]).T)))*100
            print(f'Полученная ошибка - {round(result,4)}, затрачено на 1 итерацию {time.time() - t_initial0:3.3f} c')    
            return result
        
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=tune_params['n_trials'], 
                        timeout=tune_params['timeout_secunds'])
        best_params = study.best_params
        params = best_params
        print("Подбор гиперпараметров окончен.")
        print('best_params=', best_params)
    else:
        best_params = None


    model_with_early_stop = CatBoostRegressor(
        iterations=epoches,
        random_seed=63,
        learning_rate=params['learning_rate'],
        l2_leaf_reg=params['l2_leaf_reg'],
        depth = params['depth'],
        loss_function=params['loss_func'],
        early_stopping_rounds=early_stopping_rounds
    )

    history = model_with_early_stop.fit(
        train_pool,
        eval_set=test_pool,
        verbose=verbose_,
        plot=False,
        early_stopping_rounds=early_stopping_rounds
    )
    # Evaluate the model using CatBoost's log loss and F1 score
    metrics_train = model_with_early_stop.eval_metrics(train_pool, 
                                                        metrics=params['loss_func'], 
                                                        plot=False)
    metrics_test = model_with_early_stop.eval_metrics(test_pool, 
                                                        metrics=params['loss_func'], 
                                                        plot=False)
    df_err = y_holdout.copy()
    df_err[f'N_pred_{num}'] = model_with_early_stop.predict(pd.DataFrame(X_holdout))
    df_err[f'N_{num}'] = scally.inverse_transform(df_err[[f'N_{num}']])
    df_err[f'N_pred_{num}'] = scally.inverse_transform(df_err[[f'N_pred_{num}']])
    df_err = df_err.reindex(pd.date_range(df_err.index[0],df_err.index[-1], freq='H'), axis=0)
    df_err[f'N_naiv_{num}'] = df_err[f'N_{num}'].shift(48)
    df_err['err'] = (df_err[f'N_{num}']-df_err[f'N_pred_{num}'])
    df_err['naiv_err'] = df_err[f'N_{num}'] - df_err[f'N_naiv_{num}']
    df_err['abs_err']  = df_err['err'].abs()/Ncap*100
    df_err['abs_naiv_err'] = df_err['naiv_err'].abs()/Ncap*100
    # df_err = df_err.dropna()
    print(f'Ошибка прогноза составляет {round(df_err.abs_err.mean(), 3)}')
    print(f'Ошибка наивной модели составляет {round(df_err.abs_naiv_err.mean(), 3)}')
    history = {'train':metrics_train, 'test':metrics_test}
    return df_err, model_with_early_stop, history, best_params

def solve_model_lgbm(x,y, num, params, epoches, early_stopping_rounds, scally, Ncap, 
                     tune = False, 
                     tune_params = { 'n_trials': 50,'timeout_secunds': 3600 * 3 }, 
                     verbose_=0,
                     test_size=None, 
                     start_test_date=None):
    """
    Функция для обучения модели LightGBM типа МО - бустинг 
    На входе:
        - x - датайрейм с признаками для обучения модели
        - y - датайрейм с таргетами для обучения модели
        - num - номер ветряка 
        - params - гиперпараметры для прогнозирования (заданные). В случае, если tune=True не будут учитываться
        - epoches - число эпох предсказания
        - early_stopping_rounds - число эпох, в случае, если нет улучшения за этот период, обучение останавливается
        - scally - шкалирование таргета
        - Ncap - коэффициент для оценки точности (обычно максимальная выработка по всем ветрякам ГТП)
        - tune - является ли тип расчета с подбором гиперпараметров
        - tune_params - параметры характеризуют, какое количество или сколько времени уделить на подбор гиперпараметров для данной модели
        - random_seed - функция создания модели прогнозирования задает фиксированное значение random seed. Это необходимо для более контролируемого процесса получения и сравнения результатов обучения модели
        - verbose_ - отображать процесс обучения или нет (0 - не отображать)
        - test_size - размер тестовой выборки (опционально - выбрать либо test_size либо start_test_date)
        - start_test_date - старт тестирования данных (опционально - выбрать либо test_size либо start_test_date)
    """
    t_initial0 = time.time()
    if test_size is not None:
        X_traintest, X_holdout, y_traintest, y_holdout = train_test_split(x, y, shuffle=False, test_size=test_size)
    elif start_test_date is not None:
        X_traintest, X_holdout, y_traintest, y_holdout = train_test_split_by_date(x, y, start_test_date)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X_traintest, y_traintest,
                                                                                    test_size=0.20,
                                                                                    random_state=42,
                                                                                    shuffle=False)


    if tune:
        def objective(trial):
            params = {
                'num_leaves':    trial.suggest_int('num_leaves',30, 150),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.03),
                'lambda_l1':     trial.suggest_float('lambda_l1', 0, 10, step=1),
                'lambda_l2':     trial.suggest_float('lambda_l2', 0, 10, step=1),
                'linear_lambda': trial.suggest_float('linear_lambda', 0.1, 15, log=True),
                'linear_tree':   trial.suggest_categorical('linear_tree', [True, False]),
                'loss_function': trial.suggest_categorical('loss_function', ['mae','mape','rmse','mse'])
            }
            params_train = {'objective': 'regression', 'seed': 0, 
                            'num_leaves': params['num_leaves'], 
                            'learning_rate': params['learning_rate'], 
                            'lambda_l1': params['lambda_l1'], 
                            'lambda_l2':params['lambda_l2'],
                            'metric': params['loss_function'],
                            'verbose': verbose_, 
                            'linear_lambda': params['linear_lambda'], 
                            'linear_tree': params['linear_tree'],
                            'verbose':-1}
            train_data = lgb.Dataset(X_train, label=y_train, params={'verbose': -1})
            test_data = lgb.Dataset(X_test, label=y_test, params={'verbose': -1})
            time_arr = []
            t0 = time.time()
            timer_callback = lambda env: time_arr.append(time.time() - t0)
            history_callback = {}
            early_stopping_callback = lgb.early_stopping(early_stopping_rounds, first_metric_only=False, verbose=True, min_delta=0.0)
            est = lgb.train(params_train, train_data, valid_sets=test_data, num_boost_round=epoches,
                            callbacks=[timer_callback, lgb.record_evaluation(history_callback), early_stopping_callback, lgb.log_evaluation(period=verbose_)])       #categorical_feature=category_indices
            err = est.predict(X_test)
            targ = y_test.copy()
            result = np.mean(np.mean(np.abs(targ-np.array([err]).T)))*100
            print(f'Полученная ошибка - {round(result,4)}, затрачено на 1 итерацию {time.time() - t_initial0:3.3f} c')    
            return result
        
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=tune_params['n_trials'], 
                        timeout=tune_params['timeout_secunds'])
        best_params = study.best_params
        params_train = {'objective': 'regression', 'seed': 0, 
                        'num_leaves':     best_params['num_leaves'], 
                        'learning_rate':  best_params['learning_rate'], 
                        'lambda_l1':      best_params['lambda_l1'], 
                        'lambda_l2':      best_params['lambda_l2'],
                        'metric':         best_params['loss_function'],
                        'verbose':        verbose_, 
                        'linear_lambda':  best_params['linear_lambda'], 
                        'linear_tree':    best_params['linear_tree'],
                        'verbose':-1}
        print("Подбор гиперпараметров окончен.")
        print('best_params=', best_params)
    else:
        best_params = None
        params_train = {'objective': 'regression', 'seed': 0, 
                        'num_leaves':    params['num_leaves'], 
                        'learning_rate': params['learning_rate'], 
                        'lambda_l1':     params['lambda_l1'], 
                        'lambda_l2':     params['lambda_l2'],
                        'metric':        params['loss_function'],
                        'verbose':       verbose_, 
                        'linear_lambda': params['linear_lambda'], 
                        'linear_tree':   params['linear_tree'],
                        'verbose':-1}

    train_data = lgb.Dataset(X_train, label=y_train, params={'verbose': -1})
    test_data = lgb.Dataset(X_test, label=y_test, params={'verbose': -1})

    time_arr = []
    t0 = time.time()
    timer_callback = lambda env: time_arr.append(time.time() - t0)
    history_callback = {}
    early_stopping_callback = lgb.early_stopping(early_stopping_rounds, first_metric_only=False, verbose=True, min_delta=0.0)
    est = lgb.train(params_train, train_data, valid_sets=test_data, num_boost_round=epoches,
                    callbacks=[timer_callback, lgb.record_evaluation(history_callback), early_stopping_callback, lgb.log_evaluation(period=verbose_)])       #categorical_feature=category_indices
    df_err = y_holdout.copy()
    df_err[f'N_pred_{num}'] = est.predict(pd.DataFrame(X_holdout))
    df_err[f'N_{num}'] = scally.inverse_transform(df_err[[f'N_{num}']])
    df_err[f'N_pred_{num}'] = scally.inverse_transform(df_err[[f'N_pred_{num}']])
    df_err = df_err.reindex(pd.date_range(df_err.index[0],df_err.index[-1], freq='H'), axis=0)
    df_err[f'N_naiv_{num}'] = df_err[f'N_{num}'].shift(48)
    df_err['err'] = (df_err[f'N_{num}']-df_err[f'N_pred_{num}'])
    df_err['naiv_err'] = df_err[f'N_{num}'] - df_err[f'N_naiv_{num}']
    df_err['abs_err']  = df_err['err'].abs()/Ncap*100
    df_err['abs_naiv_err'] = df_err['naiv_err'].abs()/Ncap*100
    # df_err = df_err.dropna()
    print(f'Ошибка прогноза составляет {round(df_err.abs_err.mean(), 3)}')
    print(f'Ошибка наивной модели составляет {round(df_err.abs_naiv_err.mean(), 3)}')
    return df_err, est, history_callback, best_params