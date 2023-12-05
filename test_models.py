from main import *

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



model = Model()
model.prep_all_data(loc_points, eng, name_gtp='GUK_3')
model.prep_data(num)
model.fit_predict(num, purpose='tune_params')
df_err, model1, history, best_params = model.df_err, model.model, model.history, model.best_params
# print(df_err)
print(model1)
print(best_params)
print(history.evals_result_['learn'][best_params['loss_func']])
# print(history.evals_result_['learn']['MAPE'])


# LGBM
import lightgbm as lgb

model = Model('lgbm')
model.prep_all_data(loc_points, eng, name_gtp='GUK_3')
model.prep_data(num)
model.fit_predict(num, purpose='tune_params')
df_err, model1, history, best_params = model.df_err, model.model, model.history, model.best_params
print(model1)
# print(df_err)
print(best_params)
print(history['valid_0'][best_params['loss_function']])


model = Model('fc_nn')
model.prep_all_data(loc_points, eng, name_gtp='GUK_3')
model.prep_data(num)
model.fit_predict(num, purpose='tune_params')
df_err, model1, history, best_params = model.df_err, model.model, model.history, model.best_params
# print(df_err)
print(model1)
print(best_params)
print(history.history['loss'][20:])
print(history.history['val_loss'][20:])


model = Model('lstm')
model.prep_all_data(loc_points, eng, name_gtp='GUK_3')
model.prep_data(num)
model.fit_predict(num, purpose='tune_params')
df_err, model1, history, best_params = model.df_err, model.model, model.history, model.best_params
# print(df_err)
print(model1)
print(best_params)
print(history.history['loss'][20:])
print(history.history['val_loss'][20:])