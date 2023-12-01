from sklearn.model_selection import train_test_split
import tensorflow as tf
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import lightgbm as lgb
import time
import sklearn


def solve_model_fc_nn(x, y, num, params, epoches, scally, Ncap, random_seed = 42, verbose_=1, test_size=0.175):
    x_traintest, X_holdout, y_traintest, y_holdout = train_test_split(x, y, shuffle=False, test_size=0.175)
    # x_traintest, X_holdout, y_traintest, y_holdout = df2.drop(columns=[f'N_{num}'])[:start_test_date], df2.drop(columns=[f'N_{num}'])[start_test_date:], df2[[f'N_{num}']][:start_test_date], df2[[f'N_{num}']][start_test_date:]
    
    # dropout = trial.suggest_float("dropout", 0.01, 0.09, step=0.02)
    activate1 = params['activate1']      
    activate2 = params['activate2']      
    activate3 = params['activate3']      
    activate4 = params['activate4']      
    activate5 = params['activate5']      
    
    units0 = params['units0']            
    units1 = params['units1']            
    units2 = params['units2']            
    add_layer = params['add_layer']      
    add_units3 = params['units3']    
    
    df_train_x1, df_test_x1, df_train_y1, df_test_y1 = train_test_split(x_traintest, y_traintest,
                                                                                    test_size=test_size,
                                                                                    random_state=42,
                                                                                    shuffle=False)
    earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=150, verbose=0, mode='min')
    mcp_save = tf.keras.callbacks.ModelCheckpoint('tmp_callback.hdf5', save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr_loss = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=33, verbose=0, mode='min')
    callbacks = [earlyStopping, mcp_save, reduce_lr_loss]

    nn1 = tf.keras.models.Sequential()
    nn1.add(tf.keras.layers.Input(shape=(df_train_x1.shape[1:])))

    nn1.add(tf.keras.layers.Dense(units=units0,
                                  activation=activate1,
                                  input_shape = df_train_x1.shape[1:]))
    # nn1.add(tf.keras.layers.BatchNormalization())
    # nn1.add(tf.keras.layers.Dropout(study.params['dropout'], seed=1))
    nn1.add(tf.keras.layers.Dense(units=units1,
                                  activation=activate2))
    # nn1.add(tf.keras.layers.Dropout(study.params['dropout'], seed=1))
    nn1.add(tf.keras.layers.BatchNormalization())
    nn1.add(tf.keras.layers.Dense(units=units2,
                                  activation=activate5))
    # nn1.add(tf.keras.layers.Dropout(study.params['dropout'], seed=1))
    # nn1.add(tf.keras.layers.BatchNormalization())
    if add_layer:
        nn1.add(tf.keras.layers.Dense(units=add_units3,activation=activate3))
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
               loss = 'mae')
    history = nn1.fit(df_train_x1,
            df_train_y1,
            batch_size=500,
            epochs=epoches,
            verbose=verbose_,
            callbacks=callbacks, # Early stopping
            validation_data=(df_test_x1, df_test_y1))
    
    y_holdout_pred = nn1.predict(X_holdout)
    df_pred_holdout = pd.DataFrame(data = y_holdout_pred, 
                                    index=X_holdout.index,)
                          # columns = [f'N_pred_{num}'])
    df_err = pd.concat([y_holdout,df_pred_holdout], axis=1)
    # df_err = (df_err-2)*Ncap/2+Ncap/2
    # df_err['err'] = (df_err[f'N_{num}'] - df_err[f'N_pred_{num}']).abs()*100/Ncap
    # # df_err = df_err.loc[(df_err['err']<100)&(df_err['err']>-100)]
    # print(f'Точность модели: {abs(df_err["err"]).mean()}%')
    df_err = df_err[df_err.index.hour == 0]
    daterange = pd.date_range(df_err.index[0], str(df_err.index[-1]) + ' 23:00', freq='1h')
    tmp_lst = []
    for i in range(24):
        tmp_lst.extend([f'N_targ_{i}', i])
    df_err = df_err[tmp_lst]
    df_err = pd.DataFrame(np.array(df_err).reshape(-1,2), columns=[f'N_{num}',f'N_pred_{num}'])    
    df_err.index = daterange    
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
    df_err = df_err.dropna()
    print(f'Ошибка прогноза составляет {round(df_err.abs_err.mean(), 3)}')
    print(f'Ошибка наивной модели составляет {round(df_err.abs_naiv_err.mean(), 3)}')
    return df_err, nn1, history


def solve_model_lstm(x, y, num, epoches, scally, Ncap, random_seed = 42, verbose_=0, test_size=0.175):
    # x_traintest, X_holdout, y_traintest, y_holdout = df2.drop(columns=[f'N_{num}'])[:start_test_date], df2.drop(columns=[f'N_{num}'])[start_test_date:], df2[[f'N_{num}']][:start_test_date], df2[[f'N_{num}']][start_test_date:]
    # x_traintest, X_holdout, y_traintest, y_holdout = train_test_split(x, y, shuffle=False, test_size=0.2)
    train_test_range = int(len(x)*(1-test_size))
    x_traintest, X_holdout, y_traintest, y_holdout = x[:train_test_range], x[train_test_range:], y[:train_test_range], y[train_test_range:]
    train_range = int(len(x_traintest)*(0.8))
    X_train, X_test, y_train, y_test = x_traintest[:train_range], x_traintest[train_range:], y_traintest[:train_range], y_traintest[train_range:]
    
    earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=150, verbose=0, mode='min')
    mcp_save = tf.keras.callbacks.ModelCheckpoint('tmp_callback.hdf5', save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr_loss = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=33, verbose=0, mode='min')
    callbacks = [earlyStopping, mcp_save, reduce_lr_loss]
    
    nn1 = tf.keras.models.Sequential()
    nn1.add(tf.keras.layers.LSTM(units=32, activation='sigmoid', input_shape = X_train.shape[1:]))
    nn1.add(tf.keras.layers.Dense(units=24, activation='linear'))
    opt = tf.optimizers.Adam(learning_rate=0.0015)
    nn1.compile(optimizer=opt,
               loss = 'mae')
    
    weights = nn1.weights
    zeros_weights = []
    rng=np.random.RandomState(random_seed)
    for i in range(len(weights)):
        shape = weights[i].shape
        zeros_weights.append(rng.rand(*shape))
    nn1.set_weights(zeros_weights)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    history = nn1.fit(X_train, y_train, batch_size=400, epochs=epoches, verbose=verbose_, callbacks=callbacks, validation_data=(X_test, y_test))
    
    y_holdout_pred = nn1.predict(X_holdout)
    df_pred_holdout = pd.DataFrame(data = y_holdout_pred, index=y_holdout.index)
                          # columns = [f'N_pred_{num}'])
    df_err = pd.concat([y_holdout, df_pred_holdout], axis=1)
    df_err = df_err[df_err.index.hour == 0]
    daterange = pd.date_range(df_err.index[0], str(df_err.index[-1]) + ' 23:00', freq='1h')
    tmp_lst = []
    for i in range(24):
        tmp_lst.extend([f'N_targ_{i}', i])
    df_err = df_err[tmp_lst]
    df_err = pd.DataFrame(np.array(df_err).reshape(-1,2), columns=[f'N_{num}',f'N_pred_{num}'])    
    df_err.index = daterange    
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
    df_err = df_err.dropna()
    print(f'Ошибка прогноза составляет {round(df_err.abs_err.mean(), 3)}')
    print(f'Ошибка наивной модели составляет {round(df_err.abs_naiv_err.mean(), 3)}')
    return df_err, nn1, history


def solve_model_catboost(x,y, num, params, epoches, scally, Ncap, verbose_, test_size=0.175):
    # # #({'learning_rate': 0.1, 'l2_leaf_reg': 6.0, 'depth': 8}, 755.6520656574232)
    x_traintest, X_holdout, y_traintest, y_holdout = train_test_split(x, y, shuffle=False, test_size=test_size)
    X_train, X_test, y_train, y_test = train_test_split(x_traintest, y_traintest,
                                                                                    test_size=0.20,
                                                                                    random_state=42,
                                                                                    shuffle=False)
    model_with_early_stop = CatBoostRegressor(
        iterations=epoches,
        random_seed=63,
        learning_rate=params['learning_rate'],
        l2_leaf_reg=params['l2_leaf_reg'],
        depth = params['depth'],
        loss_function='MAPE',
        early_stopping_rounds=10
    )

    history = model_with_early_stop.fit(
        X_train, y_train,
        eval_set=(X_test.values, y_test.values),
        verbose=verbose_,
        plot=False
    )
    
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
    df_err = df_err.dropna()
    print(f'Ошибка прогноза составляет {round(df_err.abs_err.mean(), 3)}')
    print(f'Ошибка наивной модели составляет {round(df_err.abs_naiv_err.mean(), 3)}')
    return df_err, model_with_early_stop, history


def solve_model_lgbm(x,y, num, params, epoches, scally, Ncap, verbose_):
    num_leaves = params['num_leaves']
    learning_rate = params['learning_rate']
    linear_lambda = params['linear_lambda']
    linear_tree = params['linear_tree']
    
    params = {'objective': 'regression', 'seed': 0, 'num_leaves': num_leaves, 'learning_rate': learning_rate,
              'metric': 'mape', 'verbose': 0, 'linear_lambda': linear_lambda, 'linear_tree': linear_tree, 
             'verbose':-1}
    x_traintest, X_holdout, y_traintest, y_holdout = train_test_split(x, y, shuffle=False, test_size=0.175)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(x_traintest, y_traintest,
                                                                                    test_size=0.20,
                                                                                    random_state=42,
                                                                                    shuffle=False)
    train_data = lgb.Dataset(X_train, label=y_train, params={'verbose': -1})
    test_data = lgb.Dataset(X_test, label=y_test, params={'verbose': -1})

    time_arr = []
    t0 = time.time()
    timer_callback = lambda env: time_arr.append(time.time() - t0)
    history_callback = {}
    early_stopping_callback = lgb.early_stopping(50, first_metric_only=False, verbose=True, min_delta=0.0)
    est = lgb.train(params, train_data, valid_sets=test_data, num_boost_round=epoches,
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
    df_err = df_err.dropna()
    print(f'Ошибка прогноза составляет {round(df_err.abs_err.mean(), 3)}')
    print(f'Ошибка наивной модели составляет {round(df_err.abs_naiv_err.mean(), 3)}')
    return df_err, est, history_callback