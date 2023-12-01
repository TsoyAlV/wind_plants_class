from main import *


# model = Model(debug=True)
# model.prep_data()
# model.fit_predict()
# df_err, model1, history = model.df_err, model.model, model.history
# print(df_err)
# print(model1)
# print(history.evals_result_['learn']['MAPE'])


# import lightgbm as lgb

# model = Model('lgbm','mae', debug=True)
# model.prep_data()
# model.fit_predict()
# df_err, model1, history = model.df_err, model.model, model.history
# print(model1)
# print(df_err)
# print(history['valid_0']['mape'])


# model = Model('fc_nn', debug=True)
# model.prep_data()
# model.fit_predict()
# df_err, model1, history = model.df_err, model.model, model.history
# print(df_err)
# print(model1)
# print(history.history['loss'][20:])
# print(history.history['val_loss'][20:])

model = Model('lstm', debug=True)
model.prep_data()
model.fit_predict()
df_err, model1, history = model.df_err, model.model, model.history
print(df_err)
print(model1)
print(history.history['loss'][20:])
print(history.history['val_loss'][20:])