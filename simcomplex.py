import scipy as sp
from scipy.spatial import Delaunay
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import json




# -----------------------------------------------------------------------------
HEIGHT, WIDTH = 800, 800
MAIN_SC = 'main_sc'
MARKER_SIZE = 6
EDGE_WIDTH = 1.5

MAIN_TITLE = "Main figure"
MAIN_MARGIN = {'b': 40, 'l': 40, 'r': 40, 't': 40}
MAIN_RANGE = (-0.1, 300.1)
MAIN_TICKVALS_NUM = 7
MAIN_TICKVALS = np.linspace(*np.around(MAIN_RANGE), MAIN_TICKVALS_NUM)

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

EMPTY_LAYOUT = go.Layout(
        title=MAIN_TITLE,
        height=HEIGHT, width=WIDTH,
        margin=MAIN_MARGIN,
        showlegend=False
)

SIMPLEX_TYPES = ST_POINT, ST_EDGE, ST_TRI, ST_HOLE = ("p", "e", "t", "h")

# MID = mid point, HL = highlighting
SC_POINTS = 'sc_points'
SC_EDGES, SC_EDGES_MID, SC_EDGES_HL, SC_EDGES_NEG_HL = 'sc_edges', 'sc_edges_mid', 'sc_edges_hl', 'sc_edges_neg_hl'
SC_TRIS, SC_TRIS_MID, SC_TRIS_HL = 'sc_tris', 'sc_tris_mid', 'sc_tris_hl'
SC_HOLES = "sc_holes"

SC_TRACES = \
    SC_POINTS, \
    SC_EDGES, SC_EDGES_MID, SC_EDGES_HL, SC_EDGES_NEG_HL, \
    SC_TRIS, SC_TRIS_MID, SC_TRIS_HL, \
    SC_HOLES

SC_MIDS = {ST_EDGE: SC_EDGES_MID, ST_TRI: SC_TRIS_MID, ST_HOLE: SC_HOLES}
MID_TYPES = SC_MIDS.keys()
MID_CONFIG = dict(marker_size=10, opacity=0.3)

# colors for vertices, edges and 2-simplexes
COLORS = ['rgba(235, 52, 134, 1)', 'rgba(39, 53, 150, 1)', 'rgba(235, 174, 52, 1)']
COLORS_DICT = {
    SC_POINTS: 'rgba(25, 72, 94, 1)',           # hsl = 199°, 58%, 23%
    SC_EDGES: 'rgba(39, 53, 150, 1)',           # hsl = 232°, 59%, 37%
    SC_EDGES_MID: 'rgba(39, 53, 150, 1)',       # hsl = 232°, 59%, 37%
    SC_TRIS: 'rgba(235, 174, 52, 1)',           # hsl = 40°, 82%, 56%
    SC_TRIS_MID: 'rgba(49, 33, 2, 1)',          # hsl = 40°, 92%, 10%
    SC_EDGES_HL: ('rgba(39, 53, 150, 1)', 'rgba(14, 153, 21, 1)'),  # (neutral, pos) = hue(232°, 123°)
    SC_EDGES_NEG_HL: 'rgba(199, 44, 181, 1)',   # hsl = 307°, 64%, 48%
    SC_TRIS_HL: 'rgba(168, 63, 19, 1)'          # hsl = 18°, 80%, 37%
}

# model
EMPTY_DATA = dict(
    tris=dict(sc_names=[SC_TRIS, SC_TRIS_HL, SC_TRIS_MID]),
    edges=dict(sc_names=[SC_EDGES, SC_EDGES_HL, SC_EDGES_NEG_HL, SC_EDGES_MID]),
    points=dict(sc_names=[SC_POINTS])
)
main_data = EMPTY_DATA

# view
main_layout = EMPTY_LAYOUT
main_figure = go.Figure(layout=main_layout)


# -----------------------------------------------------------------------------
#               UTILITY
# -----------------------------------------------------------------------------
def as_range(lim):
    lim = (0, lim) if type(lim) is int or type(lim) is float else lim
    l_min, l_max = min(lim), max(lim)
    return l_min, l_max


def get_stype(name: str):
    # if type(name) is not str:
    #     raise Exception("name is not string", name)

    stype = name[0]
    if stype not in SIMPLEX_TYPES:
        raise Exception("extracted type is not known", stype, name)

    return stype


def filter_stype(names: list, stype) -> list:
    filtered_names = list(filter(lambda name: get_stype(name) == stype, names))
    return filtered_names


def get_sname(sid, stype):
    sname = f"{stype}{sid:04d}"
    return sname


def get_pname(pid):
    pname = "{stype}{pid:04d}".format(stype=ST_POINT, pid=pid)
    return pname

def get_ename(eid):
    ename = f"{ST_EDGE}{eid:04d}"
    return ename

def get_tname(tid):
    tname = f"{ST_TRI}{tid:04d}"
    return tname


# -----------------------------------------------------------------------------
#               GETTER / SETTER
# -----------------------------------------------------------------------------
def get_main_figure() -> go.Figure:
    global main_figure
    return main_figure


def get_main_data() -> dict:
    global main_data
    return main_data


def get_data(stype) -> dict:
    global main_data
    skey = {ST_POINT: 'points', ST_EDGE: 'edges', ST_TRI: 'tris'}[stype]
    return main_data[skey]


def set_margin():
    """handle change of margins"""
    pass


def reset_fig() -> go.Figure:
    """reset the figure to 3 traces"""
    print("--- reset fig ---")
    global main_figure, main_data

    main_data = EMPTY_DATA
    sc_names = [scn for data in main_data.values() for scn in data['sc_names']]
    empty_sc = [go.Scatter(name=scn) for scn in sc_names]

    # go.Figure creates a lot of supporting data for each trace
    # for this reason we can't assign a list of new traces to ``figure.data``
    # it checks a list of traces that we try to assign to ``figure.data``:
    # >>> there shouldn't be traces with new uid, i.e., new traces are not allowed
    # so we need either manipulate the data of each existing trace
    # or wipe out all of them and add new traces
    main_figure.data = []
    for sc in empty_sc:
        main_figure.add_trace(sc)

    return main_figure


def setup_fig(layout=main_layout) -> go.Figure:
    global main_figure, main_data

    main_data = EMPTY_DATA
    sc_names = [scn for data in main_data.values() for scn in data['sc_names']]
    empty_sc = [go.Scatter(name=scn) for scn in sc_names]

    main_figure = go.Figure(
        data=empty_sc,
        layout=layout
    )
    return main_figure


def clear_highlighting() -> go.Figure:
    global main_figure, main_data

    for sc_hl in (SC_EDGES_HL, SC_EDGES_NEG_HL, SC_TRIS_HL):
        main_figure.update_traces(dict(
            ids=[], x=[], y=[], line_color=None
        ), selector=dict(name=sc_hl))

    pts_data = main_data['points']
    if 'marker_size' in pts_data and 'n' in pts_data:
        marker_size = [pts_data['marker_size']] * pts_data['n']
    else:
        marker_size = []

    main_figure.update_traces(dict(
        marker_size=marker_size
    ), selector=dict(name=SC_POINTS))

    return get_main_figure()



# -----------------------------------------------------------------------------
#           SIMPLICIAL DATA HANDLING / PROCESSING / GENERATING
# -----------------------------------------------------------------------------
def gen_random_points(n, xlim=(0, 1), ylim=(0, 1), padding=0.05):
    # print(f"{xlim} - {ylim} - {padding}")
    # x_min, x_max = xlim if type(xlim) is tuple else (0, xlim)
    # y_min, y_max = ylim if type(ylim) is tuple else (0, ylim)
    x_min, x_max = as_range(xlim)
    y_min, y_max = as_range(ylim)

    x_len = x_max - x_min
    y_len = y_max - y_min
    x = x_len * (1 - padding) * np.random.random_sample(n) + x_min + x_len * padding / 2
    y = y_len * (1 - padding) * np.random.random_sample(n) + y_min + y_len * padding / 2
    return x, y


def gen_triangulation(x, y):
    points = np.array([x, y])
    tri = Delaunay(points)
    return tri


def _log_data_row(data_row: pd.DataFrame) -> str:
    return str(data_row.loc[:, data_row.columns != 'customdata'])


def gen_points_data(points, color=COLORS_DICT[SC_POINTS]) -> dict:
    """
        generating/populating ``main_data.points``
        :param points: tuple(x, y)
        :param color: color id
        :return: `points` dict
    """

    def gen_points_df(x, y, n, color) -> pd.DataFrame:
        pids = list(range(1, n + 1))
        pnames = [f"p{pid:04d}" for pid in pids]
        colors = [color] * n
        customdata = [
            json.dumps(dict(id=pid, name=f"p{pid:04d}", sc_name=SC_POINTS))
            for pid in pids
        ]

        df = pd.DataFrame(dict(
            id=pids, name=pnames,            # plotly_row
            x=x, y=y,                        # plotly_row
            color=colors,                    # plotly_row
            customdata=customdata,           # plotly_row
            mid_point=None, mid_type=None
            # ), index=pids)
        ))
        return df

    # -------------
    x, y = points
    n = len(x)
    print(f"+++ gen_points: n={n}, x0={x[0]:.3f}, y0={y[0]:.3f}")

    pts_df = gen_points_df(x, y, n, color)

    # in case of points there is no difference between
    # the model data of points
    # the plotly data (required to make a scatter)
    # ----
    # the only "bad" thing is that we have ``customdata`` in the model data
    pts_data = dict(data=pts_df, plotly=pts_df,
                    n=n, color=color, marker_size=MARKER_SIZE)
    return pts_data


def gen_triangulation_df(points):
    """
    Triangulates the point cloud in ``points``
    and generates ``edge_dict`` and ``tri_dict`` data
        :param points: data frame[id, x, y]
        :return: (edge_df, tri_df, mid_points_df)
    """
    tri = Delaunay(points[["x", "y"]])
    print(f"tri: {len(tri.simplices)}")

    # edge_df -> plotly_edges[id=eid], plotly_edge_midpoints[id=eid]
    edges_df = pd.DataFrame(dict(
        id=[],                              # edge id
        name=[],                            # edge name: e0012, e0034
        pid_1=[], pid_2=[],                 # pids of endpoints: order?
        cof_pos=[], cof_neg=[],             # coface id (triangle/hole) where this edge is positive / negative
        cof_pos_type=[], cof_neg_type=[],   # coface type: triangle or hole
        mid_point=[],                       # pid of midpoints
        length=[],                          # euclidean length : np.linalg.norm(v)
        customdata=[]
    ))

    # tri_df -> plotly_tris[id=tid], plotly_tri_midpoints[id=tid]
    tris_df = pd.DataFrame(dict(
        id=[],                         # triangle id
        name=[],                       # triangle name: t0045, t0123
        pid_1=[], pid_2=[], pid_3=[],  # pids of vertices: order?
        eid_1=[], eid_2=[], eid_3=[],  # eids of edges: order?
        ort_1=[], ort_2=[], ort_3=[],  # orientation of edges: -1 or +1
        mid_point=[],                  # pid of midpoints
        area=[],                       # area
        customdata=[]
    ))

    mids_df = pd.DataFrame(dict(
        id=[], name=[], x=[], y=[],
        color=[], customdata=[],
        mid_point=[], mid_type=[]
    ))

    def calc_mid(X, Y):
        mid_x, mid_y = np.mean(X), np.mean(Y)
        return mid_x, mid_y

    def orient_edges(X, Y, pids):
        if len(X) != 3 or len(Y) != 3 or len(pids) != 3:
            raise Exception("orient edges: not a triangle", X, Y, pids)

        A, B, C = pids
        # AB_x, AB_y = X[1] - X[0], Y[1] - Y[0]
        # BC_x, BC_y = X[2] - X[1], Y[2] - Y[0]
        # print(f">> orient >> {A} : {B} : {C}")
        edges_x = AB_x, BC_x = X.iloc[1] - X.iloc[0], X.iloc[2] - X.iloc[1]
        edges_y = AB_y, BC_y = Y.iloc[1] - Y.iloc[0], Y.iloc[2] - Y.iloc[1]

        cross = np.cross(edges_x, edges_y)
        if cross < 0:  # anti-clockwise
            edges = [(A, B), (B, C), (C, A)]
            print(f">> orient >> ({A} : {B} : {C}) >> [Anti-CC] >> {edges}")
            return edges
        else:
            edges = [(A, C), (C, B), (B, A)]
            print(f">> orient >> ({A} : {B} : {C}) >> [CC] >> {edges}")
            return edges
        pass

    def calc_area(X, Y) -> float:
        if len(X) != 3 or len(Y) != 3:
            raise Exception("area: not a triangle", X, Y)

        X, Y = list(X), list(Y)
        a = X[0] * (Y[1] - Y[2])
        b = X[1] * (Y[2] - Y[0])
        c = X[2] * (Y[0] - Y[1])
        area = abs(0.5 * (a + b + c))
        return area

    def calc_length(X, Y) -> float:
        if len(X) != 2 or len(Y) != 2:
            raise Exception("length: not an edge", X, Y)
        X, Y = list(X), list(Y)
        x, y = X[1] - X[0], Y[1] - Y[0]
        length = np.linalg.norm([x, y])
        return length

    def add_mid_point(mdf, mpid, x, y, sid, stype):
        if stype not in SC_MIDS.keys():
            raise Exception("wrong simplex type", stype)

        sname = f"{stype}{sid:04d}"
        mdf = mdf.append(dict(
            id=int(mpid), name=sname, x=x, y=y,
            mid_point=sid,
            mid_type=stype,
            color=None,
            customdata=json.dumps(dict(
                id=int(mpid), name=sname,
                mid_point=sid, mid_type=stype,
                sc_name=SC_MIDS[stype]
            ))
        ), ignore_index=True)
        # print(f">> add mid >> {mpid} >> {sname} >> {len(mdf.index)}")
        return mpid + 1, mpid, mdf

    points = points.set_index("id", drop=False)
    eid = 0
    mpid = np.max(points.index) + 1
    for tid, s in enumerate(tri.simplices, start=1):
        # points: A < B < C
        # tri: A-B-C
        # edges: A-B, B-C, C-A
        pid_A, pid_B, pid_C = sorted(points.iloc[s]["id"].to_list())
        print(f"\n>> points[{tid}] >> A={pid_A}, B={pid_B}, C={pid_C}")

        # 0) make triangle dict
        ABC = points.loc[[pid_A, pid_B, pid_C]]
        tmid_x, tmid_y = calc_mid(ABC["x"], ABC["y"])
        mpid, mpid_old, mids_df = add_mid_point(mids_df, mpid, tmid_x, tmid_y, tid, stype=ST_TRI)
        tname = f"{ST_TRI}{tid:04d}"
        tri_row = dict(id=int(tid), name=tname,
                       pid_1=int(pid_A), pid_2=int(pid_B), pid_3=int(pid_C),
                       eid_1=None, eid_2=None, eid_3=None,
                       ort_1=None, ort_2=None, ort_3=None,
                       mid_point=int(mpid_old),
                       area=calc_area(ABC["x"], ABC["y"]))

        # 1) check existence of edges in edge_df
        #   *) if exist then get eid and orientation (negative)
        #   *) id don't - create a new eid and a new edge_row (positive orient)
        # edges = [(pid_A, pid_B), (pid_B, pid_C), (pid_A, pid_C)]
        edges = orient_edges(ABC["x"], ABC["y"], ABC["id"])
        for e, edge in enumerate(edges, 1):
            is_CA, is_not_CA = e == 3, e != 3
            anti_edge = edge[::-1]
            if edge not in edges_df.index and anti_edge not in edges_df.index:
                eid += 1
                # print(f">> no edge[{e}] >> {edge} >> eid={eid}")
                edge_points = ABC.loc[list(edge)]
                ename = f"{ST_EDGE}{eid:04d}"
                emid_x, emid_y = calc_mid(edge_points["x"], edge_points["y"])
                mpid, mpid_old, mids_df = add_mid_point(mids_df, mpid, emid_x, emid_y, eid, stype=ST_EDGE)
                edge_dict = dict(
                    id=int(eid), name=ename,
                    pid_1=edge[0], pid_2=edge[1],
                    cof_pos=int(tid), cof_neg=0,
                    cof_pos_type=ST_TRI, cof_neg_type=None,
                    mid_point=int(mpid_old),
                    length=calc_length(edge_points["x"], edge_points["y"])
                )
                edge_customdata = edge_dict.copy()
                edge_customdata["sc_name"] = SC_EDGES
                edge_dict["customdata"] = json.dumps(edge_customdata)
                edge_row = pd.DataFrame(edge_dict, index=[(min(edge), max(edge))])
                edges_df = edges_df.append(edge_row, ignore_index=False)
                # print(f">> new edge row >> edges.len={len(edges_df.index)} >> \n {_log_data_row(edge_row)}")

                tri_row[f"eid_{e}"] = eid
                tri_row[f"ort_{e}"] = +1

            else:
                # edge_row = edge_df.loc[[edge]] if edge in edge_df.index else edge_df.loc[[anti_edge]]
                edge = edge if edge in edges_df.index else anti_edge
                edge_row = edges_df.loc[[edge]]
                # print(f"\n>> edge[{e}] >> {edge} >> \n {_log_data_row(edge_row)}")
                if not edge_row["cof_pos"][0] or edge_row["cof_neg"][0]:
                    raise Exception("cof_pos/neg", edge_row)

                edge_customdata = json.loads(edge_row["customdata"][0])
                edges_df.loc[[edge], "cof_neg"] = edge_customdata["cof_neg"] = int(tid)
                edges_df.loc[[edge], "cof_neg_type"] = edge_customdata["cof_neg_type"] = ST_TRI
                edges_df.loc[[edge], "customdata"] = json.dumps(edge_customdata)

                tri_row[f"eid_{e}"] = int(edge_row["id"][0])
                tri_row[f"ort_{e}"] = -1
        # end for(edges)

        # 2) meke/add triangle row
        tri_row["customdata"] = json.dumps(tri_row)
        tris_df = tris_df.append(tri_row, ignore_index=True)
        # print(f">> new tri row >> tid= {tid} >> tris_df.len= {len(edges_df.index)} >> \n {_log_data_row(edge_row)}")
    # end for(tri.simplices)

    edges_df = edges_df.astype({"id": int,
                                "pid_1": int, "pid_2": int,
                                "cof_pos": int, "cof_neg": int,
                                "mid_point": int, "length": float
                                }, copy=False)
    tris_df = tris_df.astype({"id": int,
                              "pid_1": int, "pid_2": int, "pid_3": int,
                              "eid_1": int, "eid_2": int, "eid_3": int,
                              "ort_1": int, "ort_2": int, "ort_3": int,
                              "mid_point": int, "area": float
                              }, copy=False)
    mids_df = mids_df.astype({"id": int,
                              "x": float, "y": float,
                              "mid_point": int
                              }, copy=False)
    return edges_df, tris_df, mids_df


def gen_boundary(tris: pd.DataFrame, tri_names: list) -> pd.DataFrame:
    def boundary_row(eid=[], name=[], ort=[]):
        return pd.DataFrame(dict(
            id=eid, name=name, ort=ort
        ), index=[eid])

    boundary_df = boundary_row()

    tris = tris.set_index('name', drop=False).loc[tri_names]
    for index, tri in tris.iterrows():
        for e in range(1, 4):
            eid, ort = tri[f"eid_{e}"], tri[f"ort_{e}"]
            ename = get_ename(eid)

            if eid in boundary_df.index:
                boundary_df.loc[eid, 'ort'] += ort
            else:
                boundary_df = pd.concat([
                    boundary_df, boundary_row([int(eid)], [ename], [int(ort)])
                ])

    boundary_df = boundary_df.loc[boundary_df.ort != 0]
    boundary_df = boundary_df.astype(dict(id=int, ort=int))
    return boundary_df


def plotly_row(id=[], name=[], x=[], y=[], color=[], customdata=[]):
    return pd.DataFrame(dict(
        id=id, name=name, x=x, y=y, color=color, customdata=customdata
    ), index=[id])


def plotly_none_row(id=None, name=None):
    return plotly_row(id=id, name=name, x=None, y=None, color=None, customdata=None)


def gen_plotly_edges(points, edges, edge_colors=(COLORS_DICT[SC_EDGES], COLORS_DICT[SC_EDGES_MID])):
    edge_color, edge_mid_color = edge_colors

    points = points.set_index("id", drop=False)
    plotly_edges = plotly_row()
    plotly_edges_mid = plotly_row()

    for index, edge in edges.iterrows():
        p1 = points.loc[edge['pid_1']]
        p2 = points.loc[edge['pid_2']]
        mp = points.loc[edge['mid_point']]
        if mp['mid_point'] != edge['id'] or edge['mid_point'] != mp['id']:
            raise Exception("mid point ids are wrong", edge, mp)

        plotly_edges = pd.concat([
            plotly_edges,
            plotly_row(edge['id'], edge['name'], p1['x'], p1['y'], edge_color, edge['customdata']),
            plotly_row(edge['id'], edge['name'], p2['x'], p2['y'], edge_color, edge['customdata']),
            plotly_none_row(edge['id'], edge['name'])
         ])

        plotly_edges_mid = pd.concat([
            plotly_edges_mid,
            # plotly_row(mp['id'], edge['name'], mp['x'], mp['y'], 'SpringGreen', edge['customdata'])
            plotly_row(mp['id'], edge['name'], mp['x'], mp['y'], edge_mid_color, edge['customdata'])
        ])

    return plotly_edges, plotly_edges_mid


def gen_plotly_tris(points, tris, tri_colors=(COLORS_DICT[SC_TRIS], COLORS_DICT[SC_TRIS_MID])):
    tri_color, tri_mid_color = tri_colors
    # print(f"<<gen_plotly_tris>> : {tri_color, tri_mid_color}")

    points = points.set_index("id", drop=False)
    plotly_tris = plotly_row()
    plotly_tris_mid = plotly_row()

    for index, tri in tris.iterrows():
        p1 = points.loc[tri['pid_1']]
        p2 = points.loc[tri['pid_2']]
        p3 = points.loc[tri['pid_3']]
        mp = points.loc[tri['mid_point']]
        if mp['mid_point'] != tri['id'] or tri['mid_point'] != mp['id']:
            raise Exception("mid point ids are wrong", tri, mp)
        plotly_tris = pd.concat([
            plotly_tris,
            # plotly_row(tri['id'], tri['name'], p1['x'], p1['y'], tri_color, tri['customdata']),
            # plotly_row(tri['id'], tri['name'], p2['x'], p2['y'], tri_color, tri['customdata']),
            # plotly_row(tri['id'], tri['name'], p3['x'], p3['y'], tri_color, tri['customdata']),
            plotly_row(p1['name'], tri['name'], p1['x'], p1['y'], tri_color, tri['customdata']),
            plotly_row(p2['name'], tri['name'], p2['x'], p2['y'], tri_color, tri['customdata']),
            plotly_row(p3['name'], tri['name'], p3['x'], p3['y'], tri_color, tri['customdata']),
            plotly_none_row(tri['id'], tri['name'])
        ])

        plotly_tris_mid = pd.concat([
            plotly_tris_mid,
            plotly_row(mp['id'], tri['name'], mp['x'], mp['y'], tri_mid_color, tri['customdata'])
        ])

    return plotly_tris, plotly_tris_mid


def gen_plotly_hl(names, plotly_df, hl_color):
    plotly_df = plotly_df.set_index("name", drop=False)
    plotly_df = plotly_df.loc[names]
    if not plotly_df.empty:
        plotly_df.loc[:, "color"] = hl_color
    return plotly_df


# -----------------------------------------------------------------------------
#           MAKE TRACES / PLOTS
# -----------------------------------------------------------------------------
def get_trace(trace_name) -> go.Scatter:
    global main_figure, main_data
    if trace_name not in SC_TRACES:
        raise Exception("wrong trace name", trace_name)

    tr = main_figure.select_traces(dict(name=trace_name))
    tr = list(tr)
    if not tr or len(tr) == 0:
        reset_fig()
        return get_trace(trace_name)

    return tr[0]


def get_traces(trace_names) -> list:
    tr_list = [get_trace(name) for name in trace_names]
    return tr_list


def upd_trace_points(ids=None) -> go.Figure:
    """
        :param pts_data: current ``points`` dict
        :param ids: points which trace data need to be updated
        :return: main_figure
    """
    global main_figure, main_data
    pts_data = main_data['points']
    plotly_points = pts_data['plotly']
    plotly_points = plotly_points[plotly_points['mid_point'].isna()]

    main_figure.update_traces(dict(
        ids=plotly_points['name'],
        x=plotly_points['x'], y=plotly_points['y'],
        mode='markers', marker_opacity=1,
        marker_size=[pts_data['marker_size']] * pts_data['n'],
        marker_color=plotly_points['color'],
        hoverinfo='text', hovertext=plotly_points['name'],
        customdata=plotly_points['customdata']
    ), selector=dict(name=SC_POINTS))

    # print(f"+++ upd hvr: {tr.hovertext[0]}")
    return get_main_figure()


# noinspection PyTypeChecker
def upd_trace_edges(ids=None) -> go.Figure:
    global main_figure, main_data
    edges_data = main_data['edges']
    plotly_edges = main_data['edges']['plotly']

    # tr_edges = get_trace(SC_EDGES)
    # tr_edges_mid = get_trace(SC_EDGES_MID)
    main_figure.update_traces(dict(
        ids=plotly_edges['name'],
        x=plotly_edges['x'], y=plotly_edges['y'],
        mode='lines', marker=None,
        line_width=edges_data['edge_width'],
        line_color=edges_data['color'],
        # line_color=plotly_edges['color'],
        hovertext=[], hoverinfo="none"
        ),
        selector=dict(name=SC_EDGES)
    )

    # tr_edges = get_trace(SC_EDGES)
    # print(f"+++ upd edges +++ : {tr_edges.mode} : {tr_edges.x[0:3]}")
    return get_main_figure()


def upd_trace_edges_mid(ids=None) -> go.Figure:
    global main_figure, main_data
    plotly_mids = main_data['edges']['plotly_mids']

    main_figure.update_traces(dict(
        ids=plotly_mids['name'],
        x=plotly_mids['x'], y=plotly_mids['y'],
        mode='markers',
        marker_size=MID_CONFIG['marker_size'], marker_opacity=MID_CONFIG['opacity'],
        marker_color=plotly_mids['color'],
        hoverinfo='text',
        hovertext=plotly_mids['name'],
        # hoverlabel_font_color="white",
        customdata=plotly_mids['customdata']
        ),
        selector=dict(name=SC_EDGES_MID)
    )

    # tr_edges_mid = get_trace(SC_EDGES_MID)
    # print(f"+++ upd edges mid +++ : {tr_edges_mid.mode} : {tr_edges_mid.x[0:3]}")
    return get_main_figure()


def upd_trace_edges_hl(sc_name, plotly_hl, hl_color) -> go.Figure:
    global main_figure, main_data
    edges_data = main_data['edges']

    main_figure.update_traces(dict(
        ids=plotly_hl['name'],
        x=plotly_hl['x'], y=plotly_hl['y'],
        mode='lines', marker=None,
        line_width=edges_data['edge_width'] * 3,
        line_color=hl_color,
        opacity=0.5,
        # line_color=plotly_edges['color'],
        hovertext=[], hoverinfo="none"
    ), selector=dict(name=sc_name))

    return get_main_figure()


def upd_trace_tris(ids=None) -> go.Figure:
    global main_figure, main_data
    tris_data = main_data['tris']
    plotly_tris = tris_data['plotly']

    print("\n>>> plotly tris <<<")
    print(f"{plotly_tris['id']}")
    print("\n-------------------")

    main_figure.update_traces(dict(
        ids=plotly_tris['id'],
        x=plotly_tris['x'], y=plotly_tris['y'],
        mode='lines', marker=None,
        fill="toself", fillcolor=tris_data['color'], opacity=0.75,
        line_width=0,
        hovertext=[], hoverinfo="none"
    ), selector=dict(name=SC_TRIS))

    # tr_tris = get_trace(SC_TRIS)
    # print(f"+++ upd tris +++ : {tr_tris.mode} : {tr_tris.x[0:3]}")

    return get_main_figure()


def upd_trace_tris_mid(ids=None):
    global main_figure, main_data
    tris_data = main_data['tris']
    plotly_mids = tris_data['plotly_mids']

    main_figure.update_traces(dict(
        ids=plotly_mids['name'],
        x=plotly_mids['x'], y=plotly_mids['y'],
        mode='markers',
        marker_size=MID_CONFIG['marker_size'], marker_opacity=MID_CONFIG['opacity'],
        marker_color=plotly_mids['color'],
        hoverinfo='text',
        hovertext=plotly_mids['name'],
        hoverlabel_font_color="white",
        customdata=plotly_mids['customdata']
    ), selector=dict(name=SC_TRIS_MID))

    tr_tris_mid = get_trace(SC_TRIS_MID)
    print(f"+++ upd tris mid +++ : {tr_tris_mid.mode} : {tr_tris_mid.x[0:3]}")
    return get_main_figure()


def upd_trace_tris_hl(plotly_hl, hl_color) -> go.Figure:
    global main_figure, main_data
    tris_data = main_data['tris']

    main_figure.update_traces(dict(
        ids=plotly_hl['name'],
        x=plotly_hl['x'], y=plotly_hl['y'],
        mode='lines', marker=None,
        fill="toself", fillcolor=hl_color, opacity=0.5,
        line_width=0,
        hovertext=[], hoverinfo="none"
    ), selector=dict(name=SC_TRIS_HL))
    return get_main_figure()


def make_trace_edges():
    pass

def make_trace_tris():
    pass

# deprecated / do not use
def make_plotly_data(points, simplices):
    """deprecated / do not use"""
    # points are the given data points,
    # complex_s is the list of indices in the array of points defining 2-simplexes(triangles)
    # in the simplicial complex to be plotted
    X = []
    Y = []
    for s in simplices:
        X += [points[s[k]][0] for k in [0, 1, 2, 0]] + [None]
        Y += [points[s[k]][1] for k in [0, 1, 2, 0]] + [None]
    return X, Y

# deprecated / do not use
def make_trace(x, y, point_color=COLORS[0], line_color=COLORS[1]):
    """deprecated / not use"""
    return go.Scatter(mode='markers+lines',  # set vertices and edges of the alpha-complex
                      name='',
                      x=x,
                      y=y,
                      marker=go.Marker(size=6.5, color=point_color),
                      line=go.Line(width=1.25, color=line_color),
                      )

# deprecated / do not use
def make_triangulation(x, y):
    """deprecated / do not use"""
    print(f">>>> {len(x)}")
    points = np.array([x, y])
    tri = Delaunay(points)

    X, Y = make_plotly_data(points, tri.simplices)  # get data for Delaunay triangulation
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




# -----------------------------------------------------------------------------
#           USAGE / SCENARIOS / CONTROL
# -----------------------------------------------------------------------------
def random_cloud(n, **kwargs) -> go.Figure:
    """
    Generate ``n`` random points and
    put them on the figure
        :param n: number of points in the cloud
        :param kwargs:
        :return: main_figure
    """
    xlim = as_range(kwargs.get("xlim", (0, 1)))
    ylim = as_range(kwargs.get("ylim", (0, 1)))
    padding = kwargs.get("padding", 0.05)
    color = kwargs.get("color", COLORS_DICT[SC_POINTS])

    global main_figure, main_data
    reset_fig()
    main_figure.update_layout(
        xaxis_range=xlim, yaxis_range=ylim,
        xaxis_tickvals=np.linspace(*np.around(xlim), 7),
        yaxis_tickvals=np.linspace(*np.around(xlim), 7)
    )

    x, y = gen_random_points(n, xlim, ylim, padding)
    pts_upd = gen_points_data((x, y), color)
    main_data['points'].update(pts_upd)
    upd_trace_points()

    return get_main_figure()


def triangulate(**kwargs) -> go.Figure:
    """
    Generate a triangulation for existing cloud
    and draw it on the figure
        :param kwargs:
        :return: main_figure
    """
    edge_color = kwargs.get("edge_color", COLORS_DICT[SC_EDGES])
    edge_mid_color = kwargs.get("edge_mid_color", COLORS_DICT[SC_EDGES_MID])
    edge_colors = (edge_color, edge_mid_color)
    edge_width = kwargs.get("edge_width", 1.5)
    tri_color = kwargs.get("tri_color", COLORS_DICT[SC_TRIS])
    tri_mid_color = kwargs.get("tri_mid_color", COLORS_DICT[SC_TRIS_MID])
    tri_colors = (tri_color, tri_mid_color)

    global main_figure, main_data
    if 'data' not in main_data['points']:
        return get_main_figure()

    # TODO: should it be extracted into a new function?
    points_df = main_data['points']['data']
    points_df = points_df[points_df['mid_point'].isna()]
    edges_df, tris_df, mids_df = gen_triangulation_df(points_df)

    main_data['points']['data'] = pd.concat([points_df, mids_df], ignore_index=True)
    points_df = main_data['points']['data']
    # TODO: main_data['points']['plotly'] = points_df, because after prev line it is different from main_data['points']['data']

    plotly_edges, plotly_edges_mid = gen_plotly_edges(points_df, edges_df, edge_colors)
    main_data['edges'].update(dict(
        data=edges_df, plotly=plotly_edges, plotly_mids=plotly_edges_mid,
        color=edge_color, edge_width=edge_width
    ))

    plotly_tris, plotly_tris_mid = gen_plotly_tris(points_df, tris_df, tri_colors)
    main_data['tris'].update(dict(
        data=tris_df, plotly=plotly_tris, plotly_mids=plotly_tris_mid,
        color=tri_color
    ))

    upd_trace_edges()
    upd_trace_edges_mid()
    upd_trace_tris()
    upd_trace_tris_mid()
    # upd_trace_points()

    return get_main_figure()




def show_hide_points(stype, mode='visibility') -> go.Figure:
    """
    Shows  points/mid_points associated with ``stype``,
    hides the rest, by changing trace.visible = True/False
        :param stype: ST_POINT OR MID_TYPES(EDGE, TRI, HOLE)
        :param mode: 'visibility' or 'opacity' / 'v' or 'o'
        :return:
    """
    # inner functions -------------------------------------
    def get_show_hide_sc(stype):
        if stype == ST_POINT:  # turn off all mid-points
            sc_on = SC_POINTS
            sc_off = list(SC_MIDS.values())

        elif stype in SC_MIDS.keys():  # turn off all except a stype's mid point
            sc_on = SC_MIDS[stype]
            sc_off = [SC_POINTS] + [sc for sc in SC_MIDS.values() if sc != sc_on]

        else:  # turn off everything
            sc_on = None
            sc_off = [SC_POINTS] + list(SC_MIDS.values())
        return sc_off, sc_on

    # TODO: when config is implemented, do something with this..
    sc_opacities = {
        SC_POINTS: 1,
        SC_EDGES_MID: MID_CONFIG['opacity'],
        SC_TRIS_MID: MID_CONFIG['opacity'],
        SC_HOLES: MID_CONFIG['opacity']
    }

    def set_opacity(sc_name, visible):
        if sc_name:
            opacity = sc_opacities[sc_name] if visible else 0
            main_figure.update_traces(dict(
                marker_opacity=opacity
            ), selector=dict(name=sc_name))

    def set_visibility(sc_name, visible):
        if sc_name:
            main_figure.update_traces(dict(
                visible=visible
            ), selector=dict(name=sc_name))

    # show / hide functions -------------------------------
    modes = dict(
        opacity=(lambda sn: set_opacity(sn, True), lambda sn: set_opacity(sn, False)),
        visibility=(lambda sn: set_visibility(sn, True), lambda sn: set_visibility(sn, False))
    )
    modes['o'], modes['v'] = modes['opacity'], modes['visibility']

    # show / hide logic -----------------------------------
    global main_figure, main_data
    show, hide = modes[mode]
    sc_off, sc_on = get_show_hide_sc(stype)

    show(sc_on)
    for sc in sc_off:
        hide(sc)

    return get_main_figure()


def highlight_points(pts_names: list, **kwargs) -> go.Figure:
    """
    Highlights points by increasing their size.
        :param pts_names:
        :param kwargs:
        :return:
    """
    hl_mult = kwargs.get("hl_mult", 2)

    global main_figure, main_data
    if 'plotly' not in main_data['points']:
        return get_main_figure()

    pts_data = main_data['points']
    plotly_points = pts_data['plotly']
    plotly_points = plotly_points[plotly_points['mid_point'].isna()].set_index('name')

    marker_size = pd.Series([pts_data['marker_size']] * pts_data['n'], index=plotly_points.index)
    marker_size.loc[pts_names] = pts_data['marker_size'] * hl_mult
    main_figure.update_traces(dict(
        marker_size=marker_size
    ), selector=dict(name=SC_POINTS))

    return get_main_figure()


def highlight_edges(edge_names: list, **kwargs) -> go.Figure:
    """
    Highlights a set of edges with one color:
    all edges are taken with positive orientation
        :param edge_names: edges to highlight
        :param kwargs: hl_color, sc_name
        :return:
    """
    hl_color = kwargs.get("hl_color", COLORS_DICT[SC_EDGES_HL][0])
    sc_name = kwargs.get("sc_name", SC_EDGES_HL)

    global main_figure, main_data
    if 'plotly' not in main_data['edges']:
        return get_main_figure()

    plotly_edges = main_data['edges']['plotly']
    plotly_hl = gen_plotly_hl(edge_names, plotly_edges, hl_color)

    # clear_highlighting()
    upd_trace_edges_hl(sc_name, plotly_hl, hl_color)

    return get_main_figure()


def highlight_triangles(tri_names: list, **kwargs) -> go.Figure:
    """
    Highlight a set of triangles
        :param tri_names:
        :param kwargs:
        :return:
    """
    hl_color = kwargs.get("hl_color", COLORS_DICT[SC_TRIS_HL])

    global main_figure, main_data
    if 'plotly' not in main_data['tris']:
        return get_main_figure()

    plotly_tris = main_data['tris']['plotly']
    plotly_hl = gen_plotly_hl(tri_names, plotly_tris, hl_color)

    upd_trace_tris_hl(plotly_hl, hl_color)
    highlight_boundary(tri_names)

    return get_main_figure()


def highlight_boundary(tri_names: list, **kwargs) -> go.Figure:
    """
    Highlights the border of a list of triangles
        :param tri_names:
        :param kwargs:
        :return:
    """
    neg_color, pos_color = COLORS_DICT[SC_EDGES_NEG_HL], COLORS_DICT[SC_EDGES_HL][1]
    is_Z_2 = kwargs.get("Z_2", False)

    global main_figure, main_data
    if 'data' not in main_data['tris']:
        return get_main_figure()

    tris_df = main_data['tris']['data']
    boundary = gen_boundary(tris_df, tri_names)

    pos_boundary = boundary.loc[boundary.ort == 1]
    neg_boundary = boundary.loc[boundary.ort == -1]
    print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(f"tris= [{tri_names}]")
    print("------------------------------------------")
    print(f"### boundary ### \n {boundary}")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
    highlight_edges(pos_boundary.name,
                    hl_color=pos_color, sc_name=SC_EDGES_HL)
    highlight_edges(neg_boundary.name,
                    hl_color=neg_color, sc_name=SC_EDGES_NEG_HL)

    return get_main_figure()


# ------- wish list ----
def show_edge_orientations():
    pass


def make_hole():
    pass

# -----------------------------------------------------------------------------
#                 ===== RUN / TEST =====
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    # setup_fig(main_layout)
    # random_cloud(10, xlim=11, ylim=(2, 5), color="red")
    # gen_triangulation_data(main_data['points']['data'])
    # (0, 0), (2, 0), (0, 4), (3, 1)
    print("----")
