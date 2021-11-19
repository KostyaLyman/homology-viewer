import scipy as sp
from scipy.spatial import Delaunay
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from textwrap import dedent


def empty_fig(layout, name, mode='markers'):
    figure = go.Figure(
        data=[go.Scatter(name=name, x=[], y=[], mode=mode)],
        layout=layout)
    return figure


# -----------------------------------------------------------------------------
HEIGHT, WIDTH = 800, 800
MAIN_SC = 'main_sc'
SC_VERTEX, SC_EDGE, SC_TRI = 'sc_vertex', 'sc_edge', 'sc_tri'


MAIN_TITLE = "Main figure"
MAIN_MARGIN = {'b': 40, 'l': 40, 'r': 40, 't': 40}
MAIN_RANGE = [-0.1, 300.1]
MAIN_TICKVALS_NUM = 7
MAIN_TICKVALS = np.linspace(*np.around(MAIN_RANGE), MAIN_TICKVALS_NUM)

COLORS = ['#C0223B', '#404ca0', 'rgba(173,216,230, 0.5)']  # colors for vertices, edges and 2-simplexes

EMPTY_LAYOUT = go.Layout(
        title=MAIN_TITLE,
        height=HEIGHT, width=WIDTH,
        margin=MAIN_MARGIN
)


axis_style_box = dict(showline=True, showticklabels=True, mirror=True,
                      zeroline=False, showgrid=False,
                      range=MAIN_RANGE,
                      tickvals=MAIN_TICKVALS,
                      ticklen=5
                      )

axis_style_free = dict(showticklabels=True,
                       showgrid=False, zeroline=False,
                       range=MAIN_RANGE,
                       tickvals=MAIN_TICKVALS
                       )

main_layout = EMPTY_LAYOUT
main_figure = empty_fig(main_layout, MAIN_SC)
sc_data = dict()

# -----------------------------------------------------------------------------
def get_main_figure() -> go.Figure:
    print(">>> get main figure")
    return main_figure



def gen_random_points(n, xlim=(0, 1), ylim=(0, 1)):
    x_min, x_max = xlim if type(xlim) is tuple else (0, xlim)
    y_min, y_max = ylim if type(ylim) is tuple else (0, ylim)
    print(x_min, x_max)
    print(y_min, y_max)

    x = (x_max - x_min) * np.random.random_sample(n) + x_min
    y = (y_max - y_min) * np.random.random_sample(n) + y_min
    return x, y


def plotly_data(points, simplices):
    # points are the given data points,
    # complex_s is the list of indices in the array of points defining 2-simplexes(triangles)
    # in the simplicial complex to be plotted
    X = []
    Y = []
    for s in simplices:
        X += [points[s[k]][0] for k in [0, 1, 2, 0]] + [None]
        Y += [points[s[k]][1] for k in [0, 1, 2, 0]] + [None]
    return X, Y


def make_trace(x, y, point_color=COLORS[0], line_color=COLORS[1]):
    return go.Scatter(mode='markers+lines',  # set vertices and edges of the alpha-complex
                      name='',
                      x=x,
                      y=y,
                      marker=go.Marker(size=6.5, color=point_color),
                      line=go.Line(width=1.25, color=line_color),
                      )


def make_triangulation(x, y):
    print(f">>>> {len(x)}")
    points = np.array([x, y])
    tri = Delaunay(points)

    X, Y = plotly_data(points, tri.simplices)  # get data for Delaunay triangulation
    main_figure.data[0] = make_trace(X, Y)

    for s in tri.simplices:
        A = points[s[0]]
        B = points[s[1]]
        C = points[s[2]]
        s_path = f"M {A[0]}, {A[1]} L {B[0]}, {B[1]} L {C[0]}, {C[1]} Z"
        main_figure.layout.shapes.append(dict(
            path=s_path,
            fillcolor=COLORS[2],
            line=go.Line(color=COLORS[1], width=1.25),
            xref='x2', yref='y2'
        ))

    return main_figure



# not used
# def gen_random_triangulation(n, xlim=(0, 1), ylim=(0, 1)):
#     x, y = gen_random_points(n, xlim, ylim)
#     return make_triangulation(x, y)


# not used
def clean_fig():
    main_figure.data = []
    return main_figure



# ------- wish list ----
def highlight_edge():
    pass

def highlight_tri():
    pass

def orient_edges():
    pass

def show_edge_orientations():
    pass


def make_hole():
    pass

# -----------------------------------------------------------------------------
#                 ===== RUN / TEST =====
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    x, y = gen_random_points(10, 1, 1)
    print(np.array([x, y]))
    # gen_random_points(100, (1, 20), (13, 40))
