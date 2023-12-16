from main import *
import warnings
warnings.filterwarnings('ignore')

from sqlalchemy import create_engine

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
num = '26'
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


# import lightgbm as lgb
# model = Model()
# model.prep_all_data(loc_points, eng, name_gtp='GUK_3')
# x_trees, y_trees, x_fcnn, y_fcnn, x_lstm, y_lstm, scallx, scally = model.prep_data(model.df_all, model.nums[0])
# df_err, model1, history, best_params = model.fit_predict(x_trees, y_trees, model.nums[0], params, model.model_name, purpose='test')
# print(model1)
# print(df_err)
# print(best_params)


# # LGBM
# import lightgbm as lgb
# model_name = 'lgbm'
# model = Model(model_name)
# model.prep_all_data(loc_points, eng, name_gtp='GUK_3')
# x_trees, y_trees, x_fcnn, y_fcnn, x_lstm, y_lstm, scallx, scally = model.prep_data(model.df_all, model.nums[0])
# df_err, model1, history, best_params = model.fit_predict(x_trees, y_trees, model.nums[0], params, model_name, purpose='test')
# print(model1)
# print(df_err)
# print(best_params)


model_name = 'fc_nn'
model = Model()
model.prep_all_data(loc_points, eng, name_gtp='GUK_3')
x_trees, y_trees, x_fcnn, y_fcnn, x_lstm, y_lstm, scallx, scally = model.prep_data(model.df_all, model.nums[0])
df_err, model1, history, best_params = model.fit_predict(x_fcnn, y_fcnn, model.nums[0], params, model_name, purpose='test')
# print(df_err)
print(model1)
print(best_params)


# model_name = 'lstm'
# model = Model(model_name)
# model.prep_all_data(loc_points, eng, name_gtp='GUK_3')
# x_trees, y_trees, x_fcnn, y_fcnn, x_lstm, y_lstm, scallx, scally = model.prep_data(model.df_all, num)
# df_err, model1, history, best_params = model.fit_predict(x_lstm, y_lstm, num, params, model_name, purpose='test')
# # df_err, model1, history, best_params = model.df_err, model.model, model.history, model.best_params
# print(df_err)
# print(model1)
# print(best_params)


