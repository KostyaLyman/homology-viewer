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
                external_stylesheets=[dbc.themes.CERULEAN],
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
#                 ===== LAYOUT / COMPONENTS =====
# -----------------------------------------------------------------------------
buttons_row = dbc.Row(children=[
    dbc.Col(width=3,
            children=[
                # html.P(),
                dbc.Button("Random", id='random-button', color='primary', class_name='me-1'),
                html.P(),
                dbc.Input(id='random-size-input', type="number",
                          value=15, min=3, max=50,
                          # style={'width': 100}
                          class_name='me-1'
                          ),
                # html.P()
            ], align='center'),
    dbc.Col(width={"size": 2},
            children=[
                # html.P(),
                dbc.Button("Reset", id='reset-button', class_name='me-1', color='danger',
                            # style={'width': 83}
                            ),
                html.P(),
                dbc.Button("Science", id='sci-button', class_name='me-1', color='info',
                            style={'text-align': 'center'}),
                # html.P(),
            ], align='center'),
    dbc.Col(width={"size": 2},
            children=[
                # html.P(),
                dbc.Button("Load", id='load-button', color='warning'),
                html.P(),
                dbc.Button("Save", id='save-button', color='success'),
                # html.P()
            ], align='center')
], justify="around", class_name='mt-0 mb-0')

buttons_card = dbc.Card(children=[
    dbc.CardHeader("Point cloud"),
    dbc.CardBody([buttons_row])
], class_name='mt-0 mb-0', color="primary", outline=True)


check_radio_row = dbc.Row(children=[
    dbc.Col(children=[
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
    ])],
    style={'border': 'thin lightgrey solid'}
)

points_tab_row = dbc.Row(children=[
    dbc.Col(width=4, children=[
        # html.Div(id='selected-points-div', className='alert-inf', children=["asd"])
        html.P(),
        dbc.Card(id='selected-points-div', children=[
            dbc.CardHeader("selected"),
            dbc.CardBody([
                # html.H6("Points selected", className="card-title"),
                html.P(id='selected-points-size', className='card-text', children=["# "])
            ])
        ])
    ]),
    dbc.Col(width={'size': 4}, children=[
        html.P(),
        dbc.Button("Delete points", id='delete-points-button', color='danger')
    ])
], class_name='mb-0')

edges_tab_row = dbc.Row(children=[
    dbc.Col(width=4, children=[
        # html.Div(id='selected-edges-div', className='alert alert-inf', children=["qwe"])
        html.P(),
        dbc.Card(id='selected-edges-div', children=["qwe"], body=True)
    ]),
    dbc.Col(width=4, children=[
        html.P(),
        dbc.Button("Make chain", id='make-chain-button', color='primary')
    ])
], className='mb-0')

tris_tab_row = dbc.Row(children=[
    dbc.Col(width=8, children=[
        html.P(),
        dbc.Card(children=[
            dbc.CardHeader("Stats"),
            dbc.CardBody(id='tris-stats-card')
        ])
    ]),
    dbc.Col(width=4, children=[
        html.P(),
        dbc.Button("Make holes", id='make-holes-button', color='primary')
    ])
], className='mb-0')

holes_tab_row = dbc.Row(children=[
    dbc.Col(width=8, children=[
        html.P(),
        dbc.Card(children=[
            dbc.CardHeader("Stats"),
            dbc.CardBody(id='holes-stats-card')
        ])
    ]),
    dbc.Col(width=4, children=[
        html.P(),
        dbc.Button("hole button", id='hole-button', color='primary')
    ])
], className='mb-0')

simplex_tabs_row = dbc.Row([dbc.Col([
    # dbc.Card([
        dbc.Tabs(id='simplex-tabs', active_tab=scx.ST_POINT, class_name='nav-pills nav-justified',
                 # className='nav nav-tabs',  parent_className='nav nav-tabs',
                 children=[
                     dbc.Tab(label='Points', tab_id=scx.ST_POINT, children=[points_tab_row],
                             # active_tab_class_name='nav-item warning'
                             # active_tab_style={"background-color": "green", "color":  "#17a2b8"}
                             ),
                     dbc.Tab(label='Edges', tab_id=scx.ST_EDGE, children=[edges_tab_row]),
                     dbc.Tab(label='Triangles', tab_id=scx.ST_TRI, children=[tris_tab_row]),
                     dbc.Tab(label='Holes', tab_id=scx.ST_HOLE, children=[holes_tab_row], disabled=False)
                 ])
    # ])
])])

simplex_tabs_card = dbc.Card(children=[
    # dbc.CardHeader(tabs_row),
    # dbc.CardBody(html.Div(id='tabs-content-div'))
    dbc.CardBody(simplex_tabs_row)
], className='mt-0 mb-0', color="info", outline=True)

main_tabs_content_dict = {
    'main-tab-scx': [
        buttons_card,
        html.P(),
        simplex_tabs_card,
    ],
    'main-tab-ntk': [
        html.P("some content")
    ]
}

main_tabs_row = dbc.Row([dbc.Col([
    dbc.Tabs(id='main-tabs', active_tab='main-tab-scx', class_name='nav-justified',
             children=[
                 dbc.Tab(label='2D Complex', tab_id='main-tab-scx',
                         # class_name='mr-0 ml-0 pr-0 pl-0',
                         # label_class_name='mr-0 ml-0 pr-0 pl-0',
                         label_class_name='text-warning',
                         # style={'width': '100px', 'padding': '0'}
                         ),
                 dbc.Tab(label='Network', tab_id='main-tab-ntk', label_class_name='text-success')
             ])
])])

main_tabs_card = dbc.Card([
    dbc.CardHeader(main_tabs_row, class_name="mb-0 ml-6 pl-5"),
    # dbc.CardBody(html.Div(id='main-tabs-content-div'))
    dbc.CardBody([dbc.Row([dbc.Col([
        dbc.Collapse(id='main-tab-scx-collapse', is_open=True, children=main_tabs_content_dict['main-tab-scx']),
        dbc.Collapse(id='main-tab-ntk-collapse', is_open=False, children=main_tabs_content_dict['main-tab-ntk'])
    ])])])
], className='mt-0 mb-0', color="success", outline=True)

click_data_row = dbc.Row(html.Div(
    children=[
        dcc.Markdown(dedent("""**Click Data**""")),
        html.Pre(id='click-data', style=styles['pre'])
    ],
    style={'height': '800px', 'width': '400px'}
))

debug_tabs = ["points", "mids",
              "edges", "edges_plotly", "edges_plotly_mid",
              "tris", "tris_plotly", "tris_plotly_mid",
              "holes", "holes_plotly", "holes_plotly_mid", "holes_bd",
              "cache_bd", "cache_plotly_hl"
              ]
debug_data_row = dbc.Row([dbc.Col([dbc.Card([
        dbc.CardHeader(dbc.Row([dbc.Col([
            dbc.Tabs(id='debug-tabs', active_tab='points', class_name='nav-pills nav-justified',
                     children=[
                         dbc.Tab(label=dt, tab_id=dt) for dt in debug_tabs
                     ])
        ])]), class_name="mb-0 ml-6 pl-5"),
        dbc.CardBody([dbc.Row([dbc.Col([
            html.Div([dbc.Switch(
                id='debug-customdata-switch', label='customdata', value=True
            )]),
            html.P(),
            html.Div(id='debug-data')
        ])], style={'overflowY': 'scroll', 'overflowX': 'scroll'})
        ])
    ], className='mt-0 mb-0', color="primary", outline=True)
])])

# -----------------------------------------------------------------------------
#                 ===== LAYOUT / APP =====
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
                width=8,
                children=[
                    dbc.Row(
                        [dcc.Graph(id="main-figure")],
                        # style={'overflowY': 'scroll', 'overflowX': 'scroll', 'height': 750, 'width': 750}
                        style={'overflowY': 'scroll', 'overflowX': 'scroll'}
                    ),
                    debug_data_row
                # ], style={'overflowY': 'scroll', 'overflowX': 'scroll', 'height': 800, 'width': 800},
                # ], style={'height': 800, 'width': 800},
                ],
                className='pr-4'
            ),
            # Right Side ----------------------------------
            dbc.Col(
                width=4,
                children=[
                    # Buttons -----------------------------
                    # buttons_card,
                    # Checklist ---------------------------
                    # html.P(),
                    # check_radio_row,
                    # Tabs --------------------------------
                    # html.P(),
                    # simplex_tabs_card,
                    main_tabs_card,
                    # Click Data --------------------------
                    html.P(),
                    click_data_row
                ])
        ], justify="evenly")
])


# -----------------------------------------------------------------------------
#                 ===== CALLBACKS =====
# -----------------------------------------------------------------------------
@app.callback(
    output=[Output('click-data', 'children'),
            Output('main-figure', 'figure'),
            Output('selected-points-size', 'children'),
            # Output('tris-stats-card', 'children')
            ],
    inputs=[Input('main-figure', 'clickData'),
            Input('main-figure', 'selectedData'),
            State('reset-button', 'n_clicks'),
            State('simplex-tabs', 'active_tab')]
)
def display_click_data(clickData, selected, reset_click, stype):
    print(f"%%% selected data %%% {selected and selected['points']}")
    seleced_points = 0
    tris_table = dbc.Table()
    if selected and selected['points']:
        snames = [pts['id'] for pts in selected['points']]
        print(f"%%% selected names %%% {snames}")

        if stype == scx.ST_POINT:
            pnames = set(scx.filter_by_stype(snames, scx.ST_POINT))
            seleced_points = len(pnames)
            scx.highlight_points(pnames)

        if stype == scx.ST_EDGE:
            enames = set(scx.filter_by_stype(snames, scx.ST_EDGE))
            scx.highlight_edges(enames)

        if stype == scx.ST_TRI:
            tnames = set(scx.filter_by_stype(snames, scx.ST_TRI))
            scx.highlight_triangles(tnames)
            # tris_stats = pd.DataFrame(dict(stats={
            #     '# triangles': len(tnames),
            #     'area': 10
            # }))
            # tris_table = dbc.Table.from_dataframe(tris_stats, bordered=True, hover=True, index=True)
        if stype == scx.ST_HOLE:
            hnames = set(scx.filter_by_stype(snames, scx.ST_HOLE))
            scx.highlight_holes(hnames)
            pass

    else:
        scx.clear_highlighting()

    json_dump = "@@@ click @@@" + \
                json.dumps(clickData, indent=2) + \
                "\n@@@ select @@@\n" + \
                json.dumps(selected, indent=2) + \
                "\n@@@ magic @@@\n" + \
                json.dumps({"magic": reset_click})

    return json_dump, scx.get_main_figure(), seleced_points


@app.callback(
    output=[Output('main-figure', 'figure'),
            Output('main-figure', 'selectedData')],
    inputs=Input('simplex-tabs', 'active_tab')
)
def show_simplices_tabs(active_tab):
    print(f"%%% radio %%% {active_tab}")
    scx.clear_highlighting()
    scx.show_hide_points(active_tab, mode='opacity')

    # scx.show_obscure_points(radio_value)
    return scx.get_main_figure(), dict(points=[])


# @app.callback(
#     output=Output('tabs-content-div', 'children'),
#     inputs=Input('simplex-tabs', 'active_tab')
# )
# def stat_tabs_to_content(active_tab):
#     tabs_content = {
#         scx.ST_POINT: points_tab_row,
#         scx.ST_EDGE: edges_tab_row,
#         scx.ST_TRI: tris_tab_row
#     }
#     return tabs_content.get(active_tab, [])

@app.callback(
    output=[Output('main-tab-scx-collapse', 'is_open'),
            Output('main-tab-ntk-collapse', 'is_open')],
    inputs=Input('main-tabs', 'active_tab')
)
def show_main_tabs_content(active_tab):
    tab_content = {
        'main-tab-scx': (True, False),
        'main-tab-ntk': (False, True)
    }

    tab_scx, tab_ntk = tab_content.get(active_tab, (False, False))
    return tab_scx, tab_ntk

@app.callback(
    output=Output('debug-data', 'children'),
    inputs=[Input('debug-tabs', 'active_tab'),
            Input('debug-customdata-switch', 'value')]
)
def show_debug_data(debug_tab, show_customdata):
    debug_data = scx.get_data_df(debug_tab)
    if not show_customdata and 'customdata' in debug_data.columns:
        debug_data = debug_data.drop(columns='customdata')

    debug_table = dbc.Table.from_dataframe(debug_data, bordered=True, hover=True, index=False)
    return debug_table


@app.callback(
    output=[Output('main-figure', 'figure'),
            Output('simplex-tabs', 'active_tab')],
    inputs=[Input('random-button', 'n_clicks'),
            State('random-size-input', 'value')]
)
def random_cloud(rnd_click, rnd_size):
    print(f"\n================\n" +
          f" [{rnd_click}] : random cloud({rnd_size})" +
          f"\n================\n")
    if rnd_click is not None:
        rnd_size = rnd_size if rnd_size else 15
        # scx.random_cloud(rnd_size, xlim=(0.0, 1.0), ylim=(0.0, 1.0))
        scx.RandomCloud(rnd_size, xlim=(0.0, 300.0), ylim=(0.0, 300.0))
        scx.Triangulate()
        # scx.show_hide_points(scx.ST_POINT)

    return scx.get_main_figure(), scx.ST_POINT


@app.callback(
    output=[Output('main-figure', 'figure'),
            Output('simplex-tabs', 'active_tab')],
    inputs=[Input('sci-button', 'n_clicks')]
)
def triangulate_cloud(sci_click):
    print(f"\n================\n" +
          f" [{sci_click}] : sci click " +
          f"\n================\n")

    # if sci_click is not None:
    #     scx.Triangulate()

    return scx.get_main_figure(), scx.ST_POINT


@app.callback(
    output=[Output('main-figure', 'figure')],
    inputs=[Input('make-holes-button', 'n_clicks'),
            State('main-figure', 'selectedData'),
            State('simplex-tabs', 'active_tab')]
)
def make_holes(mkholes_click, selected, active_tab):
    print(f"\n================\n" +
          f" [{mkholes_click}] : make holes " +
          f"\n================\n")
    if mkholes_click and selected and selected['points']:
        snames = [pts['id'] for pts in selected['points']]
        print(f"%%% mk_holes :: selected names %%% {snames}")
        _, _, tnames, hnames = scx.filter_by_stypes(snames, by_points=False, by_edges=False)
        snames = list(set(tnames + hnames))
        print(f"%%% mk_holes :: filtered names %%% {snames}")
        scx.MakeHoles(snames)
        scx.show_hide_points(active_tab, mode='opacity')

    return scx.get_main_figure()


@app.callback(
    output=[Output('main-figure', 'figure'),
            Output('simplex-tabs', 'active_tab')],
    inputs=[Input('reset-button', 'n_clicks')]
)
def reset_figure(reset_click):
    print(f"\n================\n" +
          f" [{reset_click}] : reset_figure " +
          f"\n================\n")
    scx.main_figure = scx.reset_fig()
    return scx.get_main_figure(), scx.ST_POINT


# -----------------------------------------------------------------------------
#                 ===== RUN DASH SERVER =====
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    app.run_server(debug=True)
