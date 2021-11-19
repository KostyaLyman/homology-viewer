import pandas as pd
import networkx as nx

import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

import plotly.graph_objs as go
import plotly.express as px

import numpy as np
from textwrap import dedent
import json

import simcomplex as sc


sc.MAIN_RANGE = [-0.1, 300.1]
axis_style_free = dict(showgrid=False, zeroline=False, showticklabels=True,
                       range=sc.MAIN_RANGE,
                       tickvals=np.linspace(*np.around(sc.MAIN_RANGE), 7)
                       )

sc.main_layout = go.Layout(
    title=sc.MAIN_TITLE,
    height=sc.HEIGHT, width=sc.WIDTH,
    margin=sc.MAIN_MARGIN,
    xaxis=axis_style_free,
    yaxis=axis_style_free,
    autosize=True,
    hovermode='closest',
    clickmode='event+select',
    # clickmode='event',
    # dragmode="select+lasso"
    dragmode="select",
    shapes=[]
)
sc.main_figure = sc.empty_fig(sc.main_layout, sc.MAIN_SC, 'markers')

# -----------------------------------------------------------------------------
#                    ===== DASH SETUP =====
# -----------------------------------------------------------------------------
# import the css template, and pass the css template into dash
# app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
# app = dash.Dash(__name__, external_stylesheets=[dbc.themes.VAPOR])
# app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SANDSTONE])
# app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SLATE])
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
# app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
# app = dash.Dash(__name__)
app.title = "2D Simplex Viewer"
# main_figure = px.scatter(x=[], y=[], hovertext=[], text=[],
#                          title="Main Figure",
#                          )

styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}
main_checklist_options = [
    {"label": "Vertices\t", "value": "verts"},
    {"label": "Edges\t", "value": "edges"},
    {"label": "Triangles\t", "value": "tris"},
]



# -----------------------------------------------------------------------------
#                 ===== CALLBACKS =====
# -----------------------------------------------------------------------------
app.layout = html.Div([
    # Title -----------------------------------------------
    dbc.Row([html.H1("2D Simplex Viewer")],
            style={'textAlign': "center"}),
    # Main Layout -----------------------------------------
    dbc.Row(
        children=[
            # Main Graph ----------------------------------
            dbc.Col(
                width=7,
                children=[
                    dcc.Graph(id="main-graph")
                ]),
            # Right Side ----------------------------------
            dbc.Col(
                width=4,
                children=[
                    # Button ------------------------------
                    dbc.Row(
                        children=[
                            dbc.Col(
                                width=4,
                                children=[
                                    html.Button("Randomize", id='random-button',
                                                className='btn btn-info'),
                                    dcc.Input(id='random-size-input', type="number",
                                              placeholder='number of vertices')
                                ]),
                            dbc.Col(
                                width={"size": 3, "offset": 2},
                                children=[
                                    html.Button("Magic", id='magic-button',
                                                className='btn btn-primary'),
                                    html.Button("Science", id='sci-button',
                                                className='btn btn-warning',
                                                style={'text-align': 'center'})
                                ])
                        ]),
                    # Checklist ---------------------------
                    html.P(),
                    dbc.Row(html.Div(
                        # style={'border': 'thin lightgrey solid', 'width': '400px'},
                        style={'border': 'thin lightgrey solid'},
                        # className="form-check form-switch",
                        children=[dcc.Checklist(
                            id='main-checklist',
                            options=main_checklist_options,
                            value=[],
                            # className="form-check form-switch",
                            # className="form-check",
                            # labelStyle={"display": "block"},
                            labelStyle={"display": "inline-block",
                                        # 'justify-content': 'space-between',
                                        'width': '25%',
                                        'margin': '.5em'},
                            inputClassName='form-check-input',
                            labelClassName='form-check-label',
                            # className="form-check form-switch"
                            className="form-check"
                        )])
                    ),
                    # Click Data --------------------------
                    html.P(),
                    dbc.Row(html.Div(
                        style={'height': '800px', 'width': '400px'},
                        children=[
                            dcc.Markdown(dedent("""
                            **Click Data**
                            """)),
                            html.Pre(id='click-data', style=styles['pre'])
                        ]))
                ])
        ])
])
@app.callback(
    Output('click-data', 'children'),
    Input('main-graph', 'clickData'),
    Input('main-graph', 'selectedData'),
    Input('magic-button', 'n_clicks')
)
def display_click_data(clickData, selectData, magicData):
    # return json.dumps(clickData, indent=2) + json.dumps({"magic": n_clicks})
    return json.dumps(clickData, indent=2) + "@@@\n" + \
           json.dumps(selectData, indent=2) + "@@@\n" +\
           json.dumps({"magic": magicData}) + "@@@"


@app.callback(
    Output('main-graph', 'figure'),
    inputs=dict(
        rnd_click_data=(Input('random-button', 'n_clicks'), State('random-size-input', 'value')),
        magic_click=Input('magic-button', 'n_clicks'),
        sci_click=Input('sci-button', 'n_clicks')
    ))
def refresh_fig(rnd_click_data, magic_click, sci_click):
    print(">>> refresh fig")
    ctx = dash.callback_context

    if not ctx.triggered:
        button_id = 'No clicks yet'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'random-button':
        rnd_click, rnd_size = rnd_click_data
        print(f">>> random button: {rnd_click} : {rnd_size}")
        # main_figure.data = [go.Scatter(x=[200], y=[200])]
        figure = go.Figure(
            data=[go.Scatter(name=sc.MAIN_SC, x=[200], y=[200], mode='markers+lines')],
            layout=sc.main_layout
        )
        sc.main_figure = figure
        return sc.get_main_figure()

    if button_id == 'magic-button':
        print(f">>> magic button: {magic_click}")
        figure = sc.empty_fig(sc.main_layout, sc.MAIN_SC)
        sc.main_figure = figure
        return sc.get_main_figure()

    if button_id == 'sci-button':
        print(f">>> sci button: {sci_click}")
        # tr = sc.main_figure.select_traces(dict(name=MAIN_SC))
        # sc.main_figure.add_trace(go.Bar(x=[1, 2, 3]))
        # tr = sc.main_figure.select_traces(dict(type='scatter'))
        tr = sc.main_figure.select_traces(dict(name='main_sc'))
        print(f"*** tr: {tr}")
        tr = list(tr)
        print(f"*** tr_list: {tr}")

        tr[0].update(dict(x=[100, 200], y=[200, 100]))
        print(f"*** tr_upd: {tr}")
        print(f"*** tr_x: {tr[0].x}")
        print(f"*** tr_x_type: {type(tr[0].x)}")
        tr_x = list(tr[0].x)
        tr_x[0] = 133
        tr[0].x = tr_x
        print(f"*** tr_upd_upd: {tr}")
        # sc.main_figure.update_traces()
        #     patch=dict(x=[100, 200], y=[200, 100]),
        #     selector=dict(name=MAIN_SC)
        # )

        return sc.get_main_figure()

    print(">>> no button")
    return sc.get_main_figure()


# -----------------------------------------------------------------------------
#                 ===== RUN DASH SERVER =====
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    app.run_server(debug=True)
