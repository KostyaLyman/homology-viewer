import pandas as pd
import networkx as nx

import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
from dash_extensions.enrich import DashProxy, MultiplexerTransform, NoOutputTransform
# from dash_extensions.enrich import Output, State, Input

import dash_bootstrap_components as dbc

import plotly.graph_objs as go
import plotly.express as px

import numpy as np
from textwrap import dedent
import json

import simcomplex as scx


scx.MAIN_RANGE = [-0.1, 300.1]
axis_style_free = dict(showgrid=False, zeroline=False, showticklabels=True,
                       range=scx.MAIN_RANGE,
                       tickvals=np.linspace(*np.around(scx.MAIN_RANGE), 7)
                       )

scx.main_layout = go.Layout(
    title=scx.MAIN_TITLE,
    height=scx.HEIGHT, width=scx.WIDTH,
    margin=scx.MAIN_MARGIN,
    xaxis=axis_style_free,
    yaxis=axis_style_free,
    autosize=True,
    hovermode='closest',
    clickmode='event+select',
    showlegend=False,
    # clickmode='event',
    # dragmode="select+lasso"
    dragmode="select",
    # transition={
    #     'duration': 500,
    #     'easing': 'cubic-in-out'
    # },
    shapes=[]
)
scx.main_figure = scx.setup_fig(scx.main_layout)


# -----------------------------------------------------------------------------
#                    ===== DASH SETUP =====
# -----------------------------------------------------------------------------
# import the css template, and pass the css template into dash
# app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
# app = dash.Dash(__name__, external_stylesheets=[dbc.themes.VAPOR])
# app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SANDSTONE])
# app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SLATE])
# app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
# app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
# app = dash.Dash(__name__)

# app = dash.Dash(__name__,
#                 external_stylesheets=[dbc.themes.FLATLY],
#                 prevent_initial_callbacks=False
#                 )
app = DashProxy(__name__,
                external_stylesheets=[dbc.themes.FLATLY],
                # external_stylesheets=[dbc.themes.VAPOR],
                prevent_initial_callbacks=False,
                transforms=[MultiplexerTransform()]
                )
app.title = "2D Homology Viewer"


# TODO: gather all styles here
styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}

main_checklist_options = [
    {"label": "Points\t", "value": scx.ST_POINT},
    {"label": "Edges\t", "value": scx.ST_EDGE},
    {"label": "Triangles\t", "value": scx.ST_TRI},
]



# -----------------------------------------------------------------------------
#                 ===== CALLBACKS =====
# -----------------------------------------------------------------------------
app.layout = html.Div([
    # Title -----------------------------------------------
    dbc.Row([html.H1("2D Homology Viewer")],
            style={'textAlign': "center"}),
    # Main Layout -----------------------------------------
    dbc.Row(
        children=[
            # Main Graph ----------------------------------
            dbc.Col(
                width=7,
                children=[
                    dcc.Graph(id="main-figure")
                ], style={'overflowY': 'scroll', 'overflowX': 'scroll', 'height': 600}),
            # Right Side ----------------------------------
            dbc.Col(
                width=4,
                children=[
                    # Buttons -----------------------------
                    dbc.Row(
                        children=[
                            dbc.Col(
                                width=4,
                                children=[
                                    html.Button("Random cloud", id='random-button',
                                                className='btn btn-info'),
                                    dcc.Input(id='random-size-input', type="number",
                                              placeholder='number of vertices')
                                ]),
                            dbc.Col(
                                width={"size": 3, "offset": 2},
                                children=[
                                    html.Button("Reset", id='reset-button',
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
                        children=[
                            dcc.Checklist(
                                id='main-checklist',
                                options=main_checklist_options,
                                value=[],
                                # labelStyle={"display": "block"},
                                labelStyle={"display": "inline-block",
                                            # 'justify-content': 'space-between',
                                            'width': '25%',
                                            'margin': '.5em'},
                                inputClassName='form-check-input',
                                labelClassName='form-check-label',
                                # className="form-check form-switch"
                                className="form-check"
                            ),
                            dcc.RadioItems(
                                id='main-radio',
                                options=main_checklist_options,
                                value=scx.ST_POINT,
                                labelStyle={"display": "inline-block",
                                            # 'justify-content': 'space-between',
                                            'width': '25%',
                                            'margin': '.5em'},
                                inputClassName='form-check-input',
                                labelClassName='form-check-label',
                                className="form-check"
                            )
                        ])),
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
    output=[Output('click-data', 'children'),
            Output('main-figure', 'figure')],
    inputs=[Input('main-figure', 'clickData'),
            Input('main-figure', 'selectedData'),
            Input('reset-button', 'n_clicks'),
            Input('main-radio', 'value')]
)
def display_click_data(clickData, selectData, reset_click, stype):
    print(f"%%% select data %%% {selectData and selectData['points']}")
    if selectData and selectData['points']:
        snames = [pts['id'] for pts in selectData['points']]
        print(f"%%% select data %%% {snames}")

        if stype == scx.ST_POINT:
            pnames = set(scx.filter_stype(snames, scx.ST_POINT))
            scx.highlight_points(pnames)

        if stype == scx.ST_EDGE:
            enames = set(scx.filter_stype(snames, scx.ST_EDGE))
            scx.highlight_edges(enames)

        if stype == scx.ST_TRI:
            tnames = set(scx.filter_stype(snames, scx.ST_TRI))
            scx.highlight_triangles(tnames)

    else:
        scx.clear_highlighting()

    json_dump = "@@@ click @@@" + \
                json.dumps(clickData, indent=2) + \
                "\n@@@ select @@@\n" + \
                json.dumps(selectData, indent=2) + \
                "\n@@@ magic @@@\n" + \
                json.dumps({"magic": reset_click})

    return json_dump, scx.get_main_figure()


@app.callback(
    output=[Output('main-figure', 'figure'), Output('main-figure', 'selectedData')],
    inputs=Input('main-radio', 'value')
)
def radio_callback(radio_value):
    print(f"%%% radio %%% {radio_value}")
    # scx.clear_highlighting()
    # scx.show_hide_points(radio_value)
    scx.show_obscure_points(radio_value)
    return scx.get_main_figure(), dict(points=[])


@app.callback(
    output=[Output('main-figure', 'figure'), Output('main-radio', 'value')],
    inputs=[Input('random-button', 'n_clicks'), State('random-size-input', 'value')]
)
def random_cloud(rnd_click, rnd_size):
    print(f"\n================\n" +
          f" [{rnd_click}] : random cloud({rnd_size})" +
          f"\n================\n")
    if rnd_click is not None:
        rnd_size = rnd_size if rnd_size else 15
        # scx.random_cloud(rnd_size, xlim=(0.0, 1.0), ylim=(0.0, 1.0))
        scx.random_cloud(rnd_size, xlim=(0.0, 300.0), ylim=(0.0, 300.0))
        # scx.show_hide_points(scx.ST_POINT)

    return scx.get_main_figure(), scx.ST_POINT


@app.callback(
    output=Output('main-figure', 'figure'),
    inputs=[Input('reset-button', 'n_clicks')]
)
def reset_figure(reset_click):
    print(f"\n================\n" +
          f" [{reset_click}] : reset_figure " +
          f"\n================\n")
    scx.main_figure = scx.reset_fig()
    return scx.get_main_figure()


@app.callback(
    output=Output('main-figure', 'figure'),
    inputs=[Input('sci-button', 'n_clicks')]
)
def triangulate_cloud(sci_click):
    print(f"\n================\n" +
          f" [{sci_click}] : triangulate " +
          f"\n================\n")

    if sci_click is not None:
        scx.triangulate()

    return scx.get_main_figure()

# -----------------------------------------------------------------------------
#                 ===== RUN DASH SERVER =====
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    app.run_server(debug=True)
