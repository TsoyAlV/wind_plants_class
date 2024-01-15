from dash import Dash, html, Input, State, Output, ctx, callback, dcc, dash_table
import dash_bootstrap_components as dbc
import dash
# import dash_html_components as html
import plotly.express as px
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from datetime import date
from sqlalchemy import create_engine

from main import Pipeline_wind_forecast

# backend
eng = create_engine('postgresql://postgres:achtung@192.168.251.133:5432/fortum_wind')
pipeline = Pipeline_wind_forecast()
available_gtps = pipeline.get_available_GTPs(eng)
available_gtps.sort()
app = Dash(name="dash app wind forecasting")


# header
# Отображение графика выработки по каждому ветряку
select_gtp_from_dd = html.Div([
                dcc.Dropdown(available_gtps, available_gtps[0], id='demo-dropdown-gtp'), 
            ])
header_1_0 = html.Div([html.H1('Прогнозирование выработки ЭЭ на ВЭС')], style={'text-align': 'center',
                                                                                      'margin-left':'auto',
                                                                                      'margin-right':'auto'})
header_1_1 = html.H4("""Выберите ГТП из списка:""", 'children')
header_1_button = html.Div([html.Button('Показать график', id='button-get-graph-N-by-nums', 
                                        n_clicks=0,  style={'text-align': 'center', 'margin-left':'auto', 'margin-right':'auto',
                                                            'font-size':18, 'height':30, 'margin-bottom':0, 'margin-top':25}),
                                                            dcc.Loading(id="ls-loading-0", children=[html.Div(id="ls-loading-start-graph")], type="default", style={'float':'left', 'margin-bottom':20, 'margin-top':0})])  
                                                                                                        
header_1_graph = html.Div(id='fig_N_by_num')
header_1 = html.Div([header_1_0, 
                     header_1_1, 
                     select_gtp_from_dd,
                     header_1_button, 
                     header_1_graph])
@callback(Output('fig_N_by_num', 'children'),
          Output('ls-loading-start-graph', 'children'),
          Input('button-get-graph-N-by-nums', 'n_clicks'),
          State('demo-dropdown-gtp', 'value'),
          prevent_initial_call=True)
def render_fig_N_by_num(btn, gtp):
    # Выборка location
    print(f'Нажата кнопка {btn}')
    # Выполнение prep функций
    pipeline.prep_all_data(eng, gtp_name=gtp)
    pipeline.form_dict_prep_data()
    fig_W_by_wind_num = px.line(pipeline.df_all[[x for x in pipeline.df_all.columns if 'N_' in x]])
    if 'button-get-graph-N-by-nums' == ctx.triggered_id:
        res = dcc.Graph(figure=fig_W_by_wind_num, animate=True, style={'height':400})
    else:
        res = dcc.Graph(figure=px.line(), style={'height':400})
    return res, ''

_start_date = dcc.DatePickerSingle(id='date-picker-single', display_format='Y-M-D', date=date(2022, 5, 10))
start_date = html.Div([html.H4('Старт тестируемых данных:'), _start_date])
_purpose = dcc.Dropdown(['test', 'fit_by_setted_params', 'tune_params'], 'test', id='demo-dropdown-purpose')
_do_results_button = html.Div([html.Button('Выполнить обучение', id='button-start-pipeline', n_clicks=0, 
                                           style={'text-align': 'center',
                                           'margin-left':'auto',
                                           'margin-right':'auto',
                                           'font-size':18, 'height':30, 'margin-bottom':0, 'margin-top':0, 'float':'top'}), 
                                           dcc.Loading(id="ls-loading-1", children=[html.Div(id="ls-loading-output-learn")], type="default", style={'float':'top', 'margin-bottom':0, 'margin-top': 0, 'height':100})], style={'margin-bottom':20})
description_of_types = html.Div([
    html.P('test - производится обучение в 25 итераций (предназначается для отладки приложения и теста его работы)'),
    html.P('fit_by_setted_params - производится обучение с заданными по дефолту гиперпараметрами'),
    html.P('tune_params - производится обучение с подбором гиперпараметров и создается отдельная переменная класса self.best_params. На обучение уходит значительно больше времени!!! (1-2ч)'),
], style={'margin-left':40, 'margin-top':20, 'margin-bottom':20})

do_results = html.Div([html.H4('Выберите тип расчета:'), 
                       description_of_types,
                       _purpose, html.Br(),
                       _do_results_button,
                       html.Br()])
# head
layer_header = html.Div([header_1,
                        start_date,
                        do_results,], style={
                            'margin': 0,
                            # 'position': 'absolute',
                            # 'top': 20,
                            'left': '40%',
                            'margin-right': 'auto',
                            'margin-left': 'auto',
                            'width':900,
                            })


print('start_date:', _start_date.date)
layer_results = html.Div([dcc.Tabs(id="res-tabs", value='tab-1', children=[
                                    dcc.Tab(label='Results', value='tab-1'),
                                    dcc.Tab(label='Learning processes', value='tab-2'),
                                    dcc.Tab(label='Save results', value='tab-3'),
    ]),
    html.Div(id='tabs-content')], style={
                            'margin': 0,
                            # 'position': 'relative',
                            'top': 30,
                            'left': '40%',
                            'margin-right': 30,
                            'margin-left': 30
                            ,})

@callback(Output('tabs-content', 'children'),
          Input('res-tabs', 'value'),
          prevent_initial_call=True)
def render_content(value):
    print(f'Нажата кнопка {value}')
    if value == 'tab-1':
        try:
            pipeline.settings
            descr, N, err = pipeline.get_description()
            descr = descr.round(4)
            descr['description'] = descr.index
            descr = pd.concat([descr[['description']], descr.iloc[:,:-1]], axis=1)
            table = html.Div(dash_table.DataTable(descr.to_dict('records'), [{"name": i, "id": i} for i in descr.columns]), style={'width':1000, 'margin-left':40})
            fig_N = dcc.Graph(id='fig_N', figure=N, animate=True)
            fig_err = dcc.Graph(id='fig_err', figure=err, animate=True)
            return html.Div([html.H3('Основные результаты'), html.Div([table, fig_N, fig_err])])
        except:
            return html.Div([html.P('Модели еще не обучены. Настройте и выполните обучение')],  style={'margin': 0,'text-align': 'center',
                                                                                                        'margin-left':'auto',
                                                                                                        'margin-right':'auto',
                                                                                                        'margin-top':200})
    elif value == 'tab-2':
        try:
            models = pipeline.models['models']
            dict_of_learning_results = dict()
            for i in range(len(models)):
                model = models[i]
                if model=='catboost':
                    tmp_df = pd.DataFrame()
                    for num in pipeline.nums:
                        history_train = list(pipeline.data['results']['history'][num][model]['train'].values())[0][20:]
                        history_test = list(pipeline.data['results']['history'][num][model]['test'].values())[0][20:]
                        tmp_df_2 = pd.DataFrame(history_test, columns=[f'{model}_{num}'])
                        tmp_df = pd.concat([tmp_df, tmp_df_2], axis=1)
                elif model=='lgbm':
                    tmp_df = pd.DataFrame()
                    for num in pipeline.nums:
                        history_test = list(pipeline.data['results']['history'][num][model]['valid_0'].values())[0][20:]
                        tmp_df_2 = pd.DataFrame(history_test, columns=[f'{model}_{num}'])
                        tmp_df = pd.concat([tmp_df, tmp_df_2], axis=1)
                elif model in ['fc_nn','lstm']:
                    tmp_df = pd.DataFrame()
                    for num in pipeline.nums:
                        history_train = pipeline.data['results']['history'][num][model]['loss'][20:]
                        history_test = pipeline.data['results']['history'][num][model]['val_loss'][20:]
                        tmp_df_2 = pd.DataFrame(history_test, columns=[f'{model}_{num}'])
                        tmp_df = pd.concat([tmp_df, tmp_df_2], axis=1)
                tmp_df.index.name = 'epoches'
                dict_of_learning_results[model] = tmp_df
            # dict_of_learning_results = pipeline.plot_lerning_process()

            graph_to_display = html.Div([dcc.Graph(id=f"learning_process_plot_{model}", figure=px.line(dict_of_learning_results[model], title=model), animate=True) for model in pipeline.models['models']])

            # Добавляю кнопку переобучения модели
            select_model_to_relearn = html.Div([html.P('Введите модель для переобучения:'), dcc.Dropdown(pipeline.models['models'], id='select_model_to_relearn')],  style={'margin-top':120})
            select_num_to_relearn = html.Div([html.P('Введите номер ветряка для переобучения:'), dcc.Dropdown(pipeline.nums, id='select_num_to_relearn')])
            input_epoches = html.Div([html.P('Установите число эпох для предсказания. По дефолту установлено 1500'), dcc.Slider(0, 3000, 300, value=1500, id='input_epoches')])
            input_early_stopping_rounds = html.Div([html.P('Установите early_stopping_rounds для обучения. По дефолту установлено 150'), dcc.Slider(10, 310, 20, value=150, id='input_early_stopping_rounds')])
            button_relearn_model = html.Div([html.Button('Выполнить переобучение', id='button-relearn-model', n_clicks=0,  style={'text-align': 'center', 'float':'left',
                                                                                                                    'margin-left':'auto',
                                                                                                                    'margin-right':'auto',
                                                                                                                    'font-size':18, 'height':30, 'margin-top':40}), 
                                                                                                        dcc.Loading(id="ls-loading-2", children=[html.Div(id="ls-loading-output-relearn")], type="default", style={'float':'left', 'margin-bottom':0, 'margin-top':120})])
            return_relearning_status = html.Div(id='return-relearning-status', style={'float':'left', 'margin-left': 5, 'margin-top':0})
            
            return html.Div([html.Div([html.H3('Процесс обучения',), graph_to_display], style={'width':900,'float':'left'}),
                      html.Div([html.H3('Переобучение модели (если необходимо)'), 
                                select_model_to_relearn, 
                                select_num_to_relearn, 
                                input_epoches, 
                                input_early_stopping_rounds, 
                                button_relearn_model,
                                return_relearning_status], 
                                style={'float':'left', 'width':700,'margin-left':100,})], )
        except:
            return html.Div([html.P('Модели еще не обучены. Настройте и выполните обучение')],  style={'margin': 0,'text-align': 'center',
                                                                                                        'margin-left':'auto',
                                                                                                        'margin-right':'auto',
                                                                                                        'margin-top':200})

    elif value == 'tab-3':
        try:
            df_to_insert = pd.DataFrame([[str(pipeline.time_tune)[:19], pipeline.gtp_name, 48, str(pipeline.models['best_params'][model]).replace('\'', '"'), 
                                          model, pipeline.data['results']['df_err_windfarm'][model].describe()[['err']].iloc[[1],[0]].values[0][0], 
                                          len(pipeline.data['results']['df_err_windfarm'][model])] for model in pipeline.models['models']], 
                                          columns = ['init_date', 'gtp_name', 'hour_distance', 'optuna_data','model_name', 'error', 'count_rows'])
            df_to_insert['error'] = df_to_insert['error'].round(4)
            df_to_insert['optuna_data'] = df_to_insert['optuna_data'].apply(lambda x: x if len(x)<22 else x[:12]+'...'+x[-8:])
            table = html.Div(dash_table.DataTable(df_to_insert.to_dict('records'), [{"name": i, "id": i} for i in df_to_insert.columns]), style={'width':1000})
            left_part_tab_3 = html.Div([html.H3('Данные по моделям:'), html.Div([table])], style={'float':'left'})
            button_save_best_params = html.Div(html.Button('Save best params', id='btn-save-best-params', n_clicks=0))
            save_best_params = html.Div([html.H5('Для сохранения "Best_params" нажмите на кнопку "Save best params":'), 
                                         button_save_best_params, 
                                         html.Div(id='is-complited-save-best-params')])
            button_save_models = html.Div([html.Button("Download models", id="btn-save-models"),
                                        dcc.Download(id="download-models")])
            save_models =  html.Div([html.H5('Для получения обученных моделей нажмите на кнопку "Download models":'), button_save_models])
            button_other_data = html.Div([html.Button("Download other data", id="btn-save-other-data"), 
                                        dcc.Download(id="download-other-data")])
            save_other_data =  html.Div([html.H5('Для получения остальных данных нажмите на кнопку "Download other data":'), button_other_data])
            right_part_tab_3 = html.Div([html.H3('Сохранение результатов:'), html.Div([save_best_params, save_models, save_other_data])], style={'float':'left', 'width':700, 'margin-left': 40, 'margin-bottom':400})
            return html.Div([left_part_tab_3,
                             right_part_tab_3])
        except:
            return html.Div([html.P('Модели еще не обучены. Настройте и выполните обучение')],  style={'margin': 0,'text-align': 'center',
                                                                                                        'margin-left':'auto',
                                                                                                        'margin-right':'auto',
                                                                                                        'margin-top':200})


# layout
app.layout = html.Div([
    layer_header,
    layer_results
])

@callback(
    Output('ls-loading-output-learn', 'children'),
    Output('res-tabs', 'value'),
    # Output('tabs-content', 'children'),
    Input('button-start-pipeline', 'n_clicks'),
    State('demo-dropdown-purpose', 'value'),
    State('date-picker-single', 'date'),
    # demo-dropdown-purpose
    prevent_initial_call=True)
def button_do_learn(btn1, purpose, date):
    if "button-start-pipeline" == ctx.triggered_id: 
        from callbacks import callback_congratulations_message_to_telegram

        if purpose == 'test':
            pipeline.form_dict_fit_predict(models=['lstm'], start_test_date=date, purpose=purpose)
        else:
            # pipeline.form_dict_fit_predict(models=['fc_nn','lstm'],start_test_date=date, purpose=purpose)
            pipeline.form_dict_fit_predict(start_test_date=date, purpose=purpose)
        pipeline.do_ansamble()
        page = 'tab-1'
        callback_congratulations_message_to_telegram()
    return '', page


@callback(
    Output('return-relearning-status', 'children'),
    Output('ls-loading-output-relearn', 'children'),
    # Output('tabs-content', 'children'),
    Input('button-relearn-model', 'n_clicks'),
    State('select_model_to_relearn', 'value'),
    State('select_num_to_relearn', 'value'),
    State('input_epoches', 'value'),
    State('input_early_stopping_rounds', 'value'),
    prevent_initial_call=True)
def button_do_relearn(btn1, model, num, epoches, early_stopping_rounds):
    msg = ' '
    pipeline.relearn_model(model, num, int(epoches), int(early_stopping_rounds))
    if "button-relearn-model" == ctx.triggered_id: 
        msg = f'Для модели {model} и ветряка №{num} было произведеено переобучение'
    return html.Div(html.P(msg)), ' '

@callback(
    Output('is-complited-save-best-params','children'),
    Output("download-models", "data"),
    Output("download-other-data", "data"),
    State('demo-dropdown-purpose', 'value'),
    Input("btn-save-best-params", "n_clicks"),
    Input("btn-save-models", "n_clicks"),
    Input("btn-save-other-data", "n_clicks"),
    prevent_initial_call=True,
)
def save_buttons(purpose, bests, models, data):
    status_save = ''
    down_models = ''
    down_data = ''
    if "btn-save-best-params" == ctx.triggered_id:
        if pipeline.models['best_params'][pipeline.models['models'][0]] is not None:
            pipeline.upseart_best_params_to_db()
            msg = 'Данные сохранены в БД'
        else:
            msg = 'Расчет выполнен без подбора гиперпараметров. Для сохранения в БД "best_params" необходимо обучить с подбором гиперпараметов - тип расчета "tune_params"'
        status_save = html.P(msg)
    elif "btn-save-models" == ctx.triggered_id:
        path_to_file = pipeline.save_models()
        down_models = dcc.send_file(f"{path_to_file}")
    elif "btn-save-other-data" == ctx.triggered_id:
        path_to_file = pipeline.save_data()
        down_models = dcc.send_file(f"{path_to_file}")    
    return status_save, down_models, down_data


if __name__ == '__main__':
    app.run_server(debug=True, port=8010, host='192.168.252.225') # Run the Dash app