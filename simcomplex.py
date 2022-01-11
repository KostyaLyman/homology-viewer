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
SC_HOLES, SC_HOLES_MID, SC_HOLES_HL = "sc_holes", "sc_holes_mid", "sc_holes_hl"

SC_TRACES = \
    SC_POINTS, \
    SC_EDGES, SC_EDGES_MID, SC_EDGES_HL, SC_EDGES_NEG_HL, \
    SC_TRIS, SC_TRIS_MID, SC_TRIS_HL, \
    SC_HOLES, SC_HOLES_MID, SC_HOLES_HL

SC_MIDS = {ST_EDGE: SC_EDGES_MID, ST_TRI: SC_TRIS_MID, ST_HOLE: SC_HOLES_MID}
MID_TYPES = SC_MIDS.keys()
MID_CONFIG = dict(marker_size=10, opacity=0.3)

# colors for vertices, edges and 2-simplexes
COLORS = ['rgba(235, 52, 134, 1)', 'rgba(39, 53, 150, 1)', 'rgba(235, 174, 52, 1)']
COLORS_DICT = {
    SC_POINTS: 'rgba(25, 72, 94, 1)',           # hsl = 199°, 58%, 23%

    SC_EDGES: 'rgba(39, 53, 150, 1)',           # hsl = 232°, 59%, 37%
    SC_EDGES_MID: 'rgba(39, 53, 150, 1)',       # hsl = 232°, 59%, 37%
    SC_EDGES_HL: ('rgba(39, 53, 150, 1)', 'rgba(14, 153, 21, 1)'),  # (neutral, pos) = hue(232°, 123°)
    SC_EDGES_NEG_HL: 'rgba(199, 44, 181, 1)',   # hsl = 307°, 64%, 48%

    SC_TRIS: 'rgba(235, 174, 52, 1)',           # hsl = 40°, 82%, 56%
    SC_TRIS_MID: 'rgba(49, 33, 2, 1)',          # hsl = 40°, 92%, 10%
    SC_TRIS_HL: 'rgba(168, 63, 19, 1)',         # hsl = 18°, 80%, 37%

    SC_HOLES: 'rgba(156, 28, 169, 1)',          # hsl = 295°, 84%, 66%
    SC_HOLES_MID: 'rgba(212, 23, 231, 1)',      # hsl = 295°, --------
    SC_HOLES_HL: 'rgba(138, 85, 192, 1)',       # hsl = 270°, --------


}

# model
EMPTY_DATA = dict(
    holes=dict(sc_names=[SC_HOLES, SC_HOLES_HL, SC_HOLES_MID]),
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
        raise ValueError("get_stype: extracted type is not known", stype, name)

    return stype


def filter_by_stype(snames: list, stype) -> list:
    filtered_names = list(filter(lambda name: get_stype(name) == stype, snames))
    return filtered_names


def filter_by_stypes(snames: list, by_points=True, by_edges=True, by_tris=True, by_holes=True) -> list:
    pnames = list(filter(lambda name: get_stype(name) == ST_POINT, snames)) if by_points else []
    enames = list(filter(lambda name: get_stype(name) == ST_EDGE, snames))  if by_edges else []
    tnames = list(filter(lambda name: get_stype(name) == ST_TRI, snames))   if by_tris else []
    hnames = list(filter(lambda name: get_stype(name) == ST_HOLE, snames))  if by_holes else []
    return pnames, enames, tnames, hnames


def get_sname(sid, stype):
    sname = f"{stype}{sid:04d}"
    return sname


def get_pname(pid):
    pname = "{stype}{pid:04d}".format(stype=ST_POINT, pid=pid)
    return pname

def get_ename(eid):
    ename = f"{ST_EDGE}{eid:04d}"
    return ename

def get_enames(eids):
    # enames = []
    # for eid in eids:
    #     # TODO: rewrite without loop (apply, reduce, map)
    #     ename = f"{ST_EDGE}{eid:04d}"
    #     enames.append(ename)
    enames = list(map(get_ename, eids))
    return enames

def get_tname(tid):
    tname = f"{ST_TRI}{tid:04d}"
    return tname

def get_hname(hid):
    hname = f"{ST_HOLE}{hid:04d}"
    return hname

def parse_sname(name):
    try:
        stype = get_stype(name)
        sid = int(name[1:])
    except:
        stype, sid = None, 0
    finally:
        return stype, sid

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
    skey = {ST_POINT: 'points', ST_EDGE: 'edges', ST_TRI: 'tris', ST_HOLE: 'holes'}[stype]
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

    for sc_hl in (SC_EDGES_HL, SC_EDGES_NEG_HL, SC_TRIS_HL, SC_HOLES_HL):
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


def _log_data_row(data_row: pd.DataFrame) -> str:
    return str(data_row.loc[:, data_row.columns != 'customdata'])


def points_row(pid=[], pname=[], x=[], y=[], mp_pointer=[], mp_type=[],
               color=[], customdata=[], sc_name=SC_POINTS):
    pid = [pid] if type(pid) is not list else pid
    points_dict = dict(
        id=pid, name=pname,  # plotly_row
        x=x, y=y,  # plotly_row
        mp_pointer=mp_pointer, mp_type=mp_type,
        color=color,  # plotly_row
        # customdata=customdata,  # plotly_row
    )
    if customdata is None:
        customdata = customdata_from_row(points_dict, sc_name)
    points_dict['customdata'] = customdata
    return pd.DataFrame(points_dict, index=pd.Index(pid, name='pid'))


def edges_row(eid=[], ename=[], pid_1=[], pid_2=[],
              cof_pos=[], cof_neg=[], cof_pos_type=[], cof_neg_type=[],
              mid_point=[], length=[], customdata=[], as_dict=False):
    if not as_dict:
        ename = [ename] if type(ename) is not list else ename
    edges_dict = dict(
        id=eid,                                 # edge id
        name=ename,                             # edge name: e0012, e0034
        pid_1=pid_1, pid_2=pid_2,               # pids of endpoints: order?
        cof_pos=cof_pos, cof_neg=cof_neg,       # coface ``name`` where this edge is positive / negative
        cof_pos_type=cof_pos_type, cof_neg_type=cof_neg_type,   # coface type: triangle or hole or None
        # if the edge is on the outer boundary
        # then: cof = None, cof_type = None   <-- ints cannot be None
        mid_point=mid_point,                    # pid of midpoints
        length=length,                          # euclidean length : np.linalg.norm(v)
        # customdata=customdata
    )
    if customdata is None:
        customdata = customdata_from_row(edges_dict, SC_EDGES)
    edges_dict['customdata'] = customdata

    if as_dict:
        return edges_dict
    return pd.DataFrame(edges_dict, pd.Index(ename, name="ename"))


def tris_row(tid=[], tname=[],
             pid_1=[], pid_2=[], pid_3=[],
             eid_1=[], eid_2=[], eid_3=[],
             ort_1=[], ort_2=[], ort_3=[],
             mid_point=[], area=[], customdata=[], as_dict=False):
    if not as_dict:
        tname = [tname] if type(tname) is not list else tname
    tris_dict = dict(
        id=tid,                                     # triangle id
        name=tname,                                 # triangle name: t0045, t0123
        pid_1=pid_1, pid_2=pid_2, pid_3=pid_3,      # pids of vertices: order?
        eid_1=eid_1, eid_2=eid_2, eid_3=eid_3,      # eids of edges: order?
        ort_1=ort_1, ort_2=ort_2, ort_3=ort_3,      # orientation of edges: -1 or +1
        mid_point=mid_point,                        # pid of midpoints
        area=area,                                  # area
        # customdata=customdata
    )
    if customdata is None:
        customdata = customdata_from_row(tris_dict, SC_TRIS)
    tris_dict['customdata'] = customdata

    if as_dict:
        return tris_dict
    return pd.DataFrame(tris_dict, pd.Index(tname, name="tname"))


def holes_row(hid=[], hname=[], mid_point=[], area=[], customdata=[]):
    hname = [hname] if type(hname) is not list else hname
    holes_dict = dict(
        id=hid, name=hname,
        mid_point=mid_point, area=area,
        # customdata=customdata
    )
    if customdata is None:
        customdata = customdata_from_row(holes_dict, SC_HOLES)
    holes_dict['customdata'] = customdata

    return pd.DataFrame(holes_dict, index=pd.Index(hname, name="hname"))


def boundary_row(eid=[], ename=[], ort=[], sname=[]):
# def boundary_row(eid=[], ename=[], ort=[], sname=[], idx=None, idx_name='eid'):
    # TODO: maybe we don't need to pass ``idx`` and just fix it to be ``eid``
    eid = [eid] if type(eid) is not list else eid
    # idx = eid if idx is None else idx
    # idx = [idx] if type(idx) is not list else idx
    return pd.DataFrame(dict(
        eid=eid, ename=ename, ort=ort, sname=sname
    # ), index=pd.Index(idx, name=idx_name))
    ), index=pd.Index(eid, name='eid'))


def plotly_row(sid=[], sname=[], x=[], y=[], color=[], customdata=[]):
    return pd.DataFrame(dict(
        id=sid, name=sname, x=x, y=y, color=color, customdata=customdata
    ), index=[sid])


def plotly_none_row(sid, sname):
    return plotly_row(sid=sid, sname=sname, x=None, y=None, color=None, customdata=None)


def fix_type_plotly(plotly_df):
    return plotly_df.astype({
        "id": int, "x": float, "y": float
    }, copy=False)


def fix_type_points(points):
    return points.astype({
        "id": int, "x": float, "y": float,
    }, copy=False)


def fix_type_edges(edges):
    return edges.astype({
        "id": int, "pid_1": int, "pid_2": int,
        "mid_point": int, "length": float
    }, copy=False)


def fix_type_tris(tris):
    return tris.astype({
        "id": int,
        "pid_1": int, "pid_2": int, "pid_3": int,
        "eid_1": int, "eid_2": int, "eid_3": int,
        "ort_1": int, "ort_2": int, "ort_3": int,
        "mid_point": int, "area": float
    }, copy=False)


def fix_type_holes(holes):
    return holes.astype({
        "id": int, "mid_point": int, "area": float
    })


def _set_log_index(df, index, log="", inplace=False):
    if df.index.name is not None and df.index.name.endswith(index):
        return df

    print(f"::: SET INDEX [{str(inplace)[0]}]::: <{log}> ::: [{df.index.name}] --> [{index}]")
    if inplace:
        df.set_index(index, drop=False, inplace=True)
        return df
    else:
        return df.set_index(index, drop=False, inplace=False)


def customdata_from_row(row, sc_name):
    if type(row) is pd.Series:
        row = row.to_dict()
    if type(row) is pd.DataFrame:
        row = row.iloc[0].to_dict()
    if type(row) is dict:
        customdata = row.copy()
        customdata["sc_name"] = sc_name
        return json.dumps(customdata)
    return None


def append_mid_point(mids, mpid, x, y, sid, stype):
    sname = get_sname(sid, stype)
    pname = get_pname(mpid)
    # customdata = json.dumps(dict(
    #     id=int(mpid), name=pname, mp_pointer=sname, mp_type=stype, sc_name=SC_MIDS[stype]
    # ))

    mids = mids.append(points_row(
        int(mpid), pname, x, y,
        mp_pointer=sname, mp_type=stype,
        color=None,
        customdata=None, sc_name=SC_MIDS[stype]
    ), ignore_index=False)

    return mids, mpid + 1, mpid


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

        df = points_row(pids, pnames, x, y, None, None, colors, customdata)
        return df

    # -------------
    x, y = points
    n = len(x)
    print(f"+++ gen_points: n={n}, x0={x[0]:.3f}, y0={y[0]:.3f}")

    points_df = gen_points_df(x, y, n, color)
    points_df = _set_log_index(points_df, "id", "gen_pts_data : points_df")
    # in case of points there is no difference between
    # the model data of points
    # the plotly data (required to make a scatter)
    # ----
    # the only "bad" thing is that we have ``customdata`` in the model data
    points_data = dict(data=points_df, plotly=points_df,
                       n=n, color=color, marker_size=MARKER_SIZE)
    return points_data


def gen_triangulation_df(simplices, points_df) -> object:
    """
    Triangulates the point cloud in ``points``
    and generates ``edge_dict`` and ``tri_dict`` data
        :param simplices:
        :param points_df: data frame[id, x, y]
        :return: (edge_df, tri_df, mid_points_df)
    """
    edges_df = edges_row()
    tris_df = tris_row()
    mids_df = points_row()

    def calc_mid(X, Y):
        mid_x, mid_y = np.mean(X), np.mean(Y)
        return mid_x, mid_y

    def orient_edges(X, Y, pids):
        if len(X) != 3 or len(Y) != 3 or len(pids) != 3:
            raise ValueError("orient_edges: not a triangle, too many/not enough dimensions", X, Y, pids)

        A, B, C = pids
        # AB_x, AB_y = X[1] - X[0], Y[1] - Y[0]
        # BC_x, BC_y = X[2] - X[1], Y[2] - Y[0]
        # print(f">> orient >> {A} : {B} : {C}")
        edges_x = AB_x, BC_x = X.iloc[1] - X.iloc[0], X.iloc[2] - X.iloc[1]
        edges_y = AB_y, BC_y = Y.iloc[1] - Y.iloc[0], Y.iloc[2] - Y.iloc[1]

        cross = np.cross(edges_x, edges_y)
        if cross < 0:  # anti-clockwise
            edges = [(A, B), (B, C), (C, A)]
            # print(f">> orient >> ({A} : {B} : {C}) >> [Anti-CC] >> {edges}")
            return edges
        else:
            edges = [(A, C), (C, B), (B, A)]
            # print(f">> orient >> ({A} : {B} : {C}) >> [CC] >> {edges}")
            return edges
        pass

    def calc_area(X, Y) -> float:
        if len(X) != 3 or len(Y) != 3:
            raise ValueError("calc_area: not a triangle, too many/not enough dimensions", X, Y)

        X, Y = list(X), list(Y)
        a = X[0] * (Y[1] - Y[2])
        b = X[1] * (Y[2] - Y[0])
        c = X[2] * (Y[0] - Y[1])
        area = abs(0.5 * (a + b + c))
        return area

    def calc_length(X, Y) -> float:
        if len(X) != 2 or len(Y) != 2:
            raise ValueError("calc_length: not an edge, too many/not enough dimensions", X, Y)
        X, Y = list(X), list(Y)
        x, y = X[1] - X[0], Y[1] - Y[0]
        length = np.linalg.norm([x, y])
        return length

    # points_df = points_df.set_index("id", drop=False)
    points_df = _set_log_index(points_df, "id", "gen_tri_df : points")
    eid = 0
    mpid = np.max(points_df.index) + 1
    for tid, s in enumerate(simplices, start=1):
        # points: A < B < C
        # tri: A-B-C
        # edges: A-B, B-C, C-A
        pid_A, pid_B, pid_C = sorted(points_df.iloc[s]["id"].to_list())
        # print(f"\n>> points[{tid}] >> A={pid_A}, B={pid_B}, C={pid_C}")

        # 0) make triangle dict
        ABC = points_df.loc[[pid_A, pid_B, pid_C]]
        tname = get_tname(tid)
        tmid_x, tmid_y = calc_mid(ABC["x"], ABC["y"])
        mids_df, mpid, mpid_old = append_mid_point(mids_df, mpid, tmid_x, tmid_y, tid, stype=ST_TRI)
        # tri_row = dict(id=int(tid), name=tname,
        #                pid_1=int(pid_A), pid_2=int(pid_B), pid_3=int(pid_C),
        #                eid_1=None, eid_2=None, eid_3=None,
        #                ort_1=None, ort_2=None, ort_3=None,
        #                mid_point=int(mpid_old),
        #                area=calc_area(ABC["x"], ABC["y"]))
        tri_row = tris_row(tid, tname, pid_A, pid_B, pid_C,
                           eid_1=None, eid_2=None, eid_3=None,
                           ort_1=None, ort_2=None, ort_3=None,
                           mid_point=int(mpid_old),
                           area=calc_area(ABC["x"], ABC["y"]),
                           as_dict=True)

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
                ename = get_ename(eid)
                emid_x, emid_y = calc_mid(edge_points["x"], edge_points["y"])
                mids_df, mpid, mpid_old = append_mid_point(mids_df, mpid, emid_x, emid_y, eid, stype=ST_EDGE)
                # edge_dict = dict(
                #     id=int(eid), name=ename,
                #     pid_1=edge[0], pid_2=edge[1],
                #     cof_pos=tname, cof_neg=None,
                #     cof_pos_type=ST_TRI, cof_neg_type=None,
                #     mid_point=int(mpid_old),
                #     length=calc_length(edge_points["x"], edge_points["y"])
                # )
                edge_dict = edges_row(eid, ename,
                                      pid_1=edge[0], pid_2=edge[1],
                                      cof_pos=tname, cof_neg=None,
                                      cof_pos_type=ST_TRI, cof_neg_type=None,
                                      mid_point=int(mpid_old),
                                      length=calc_length(edge_points["x"], edge_points["y"]),
                                      customdata=None,
                                      as_dict=True)
                # edge_customdata = edge_dict.copy()
                # edge_customdata["sc_name"] = SC_EDGES
                # edge_dict["customdata"] = json.dumps(edge_customdata)
                # TODO: edges.index <- enames
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
                    raise ValueError("No cof_pos/neg", edge_row)

                edge_customdata = json.loads(edge_row["customdata"][0])
                edges_df.loc[[edge], "cof_neg"] = edge_customdata["cof_neg"] = tname
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

    edges_df = fix_type_edges(edges_df)
    tris_df = fix_type_tris(tris_df)
    mids_df = fix_type_points(mids_df)

    return edges_df, tris_df, mids_df


def gen_triangulation_data(simplices, points_df, **kwargs):
    """
    Generates triangulation data from a data frome of points
        :param simplices:
        :param points_df: a data frame of ``point``s to triangulate
        :param kwargs:
        :return:  tuple of update dicts
    """
    edge_color = kwargs.get("edge_color", COLORS_DICT[SC_EDGES])
    edge_mid_color = kwargs.get("edge_mid_color", COLORS_DICT[SC_EDGES_MID])
    edge_colors = (edge_color, edge_mid_color)
    edge_width = kwargs.get("edge_width", 1.5)

    tri_color = kwargs.get("tri_color", COLORS_DICT[SC_TRIS])
    tri_mid_color = kwargs.get("tri_mid_color", COLORS_DICT[SC_TRIS_MID])
    tri_colors = (tri_color, tri_mid_color)

    # ----------------------------------------------------------------
    edges_df, tris_df, mids_df = gen_triangulation_df(simplices, points_df)
    points_df = pd.concat([points_df, mids_df], ignore_index=True)

    _set_log_index(points_df, "id", "gen_tri_data : points", inplace=True)
    _set_log_index(edges_df, "name", "gen_tri_data : edges", inplace=True)
    _set_log_index(tris_df, "name", "gen_tri_data : tris", inplace=True)

    plotly_edges, plotly_edges_mid = gen_plotly_edges(points_df, edges_df, edge_colors)
    plotly_tris, plotly_tris_mid = gen_plotly_tris(points_df, tris_df, tri_colors)

    points_dict = dict(
        data=points_df, plotly=points_df
    )

    edges_dict = dict(
        data=edges_df, plotly=plotly_edges, plotly_mids=plotly_edges_mid,
        color=edge_color, edge_width=edge_width
    )

    tris_dict = dict(
        data=tris_df, plotly=plotly_tris, plotly_mids=plotly_tris_mid,
        color=tri_color
    )

    return points_dict, edges_dict, tris_dict


def add_ort_to_boundary_edge(bd, eid, ort, sname):
    # expects: bd.index == eid
    old_ort = bd.loc[eid, 'ort']
    bd.loc[eid, 'ort'] += ort
    new_ort = bd.loc[eid, 'ort']

    if new_ort == 0:
        bd.loc[eid, 'sname'] = bd.loc[eid, 'sname'] + sname
    else:
        # old_ort * new_ort < 0  -- orientation flipped
        bd.loc[eid, 'sname'] = sname if old_ort * new_ort < 0 else bd.loc[eid, 'sname']
    return bd


def get_tris_boundary(tnames: list, tris, drop_zeros=True, **kwargs) -> pd.DataFrame:
    tris = _set_log_index(tris, "name", "gen_tris_bd : tris")
    if tnames:
        # tris = tris.set_index('name', drop=False).loc[tnames]
        tris = tris.loc[tnames]

    bd = boundary_row()
    for tname, tri in tris.iterrows():
        for e in range(1, 4):
            eid, ort = tri[f"eid_{e}"], tri[f"ort_{e}"]

            if eid in bd.index:
                add_ort_to_boundary_edge(bd, eid, ort, tname)
            else:
                ename = get_ename(eid)
                bd = bd.append(boundary_row([eid], [ename], [ort], [tname]))

    if drop_zeros:
        bd = bd.loc[bd.ort != 0]

    bd = bd.astype(dict(eid=int, ort=int))
    return bd



def gen_boundaries(snames, tris, holes_bd, **kwargs):
    if type(snames) is tuple:
        tnames, hnames = snames
    else:
        tnames = filter_by_stype(snames, ST_TRI)
        hnames = filter_by_stype(snames, ST_HOLE)

    # holes_bd = holes_bd.set_index("sname", drop=False).loc[hnames].set_index("eid", drop=False)
    holes_bd = _set_log_index(holes_bd, "sname", "gen_bd : holes_bd").loc[hnames].set_index("eid", drop=False)
    tris_bd = get_tris_boundary(tnames, tris, **kwargs)
    bd = add_boundaries(tris_bd, holes_bd, **kwargs)

    return bd


def add_boundaries(bd_1, bd_2, ignore_zeros=True, drop_zeros=True):
    # we expect boundaries to have index == 'eid'
    # bd = bd_1.set_index("eid", drop=False)
    # bd_2 = bd_2.set_index("eid", drop=False)

    if ignore_zeros:
        bd_1 = bd_1.loc[bd_1.ort != 0]
        bd_2 = bd_2.loc[bd_2.ort != 0]

    for eid, bd_row in bd_2.iterrows():
        if eid in bd_1.index:
            add_ort_to_boundary_edge(bd_1, eid, bd_row['ort'], bd_row['sname'])
        else:
            bd_1 = bd_1.append(bd_row)

    if drop_zeros:
        bd_1 = bd_1.loc[bd_1.ort != 0]

    bd_1 = bd_1.astype(dict(eid=int, ort=int))
    return bd_1


def gen_holes_df(snames, points, edges, tris, holes=None, holes_bd=None):
    """
    Generates holes out of a list of triangles and holes.
    If there are holes in ``snames``, then merges those holes with created ones.
    Note that we can't merge with the outer void.

        :param snames: a list of triangles and/or holes
        :param points:
        :param edges:
        :param tris:
        :param holes:   to merge holes with triangle or another hole
        :param holes_bd:
        :return: holes_df, holes_bd, holes_mids,
                 to_remove = (pnames, enames, tnames, hnames)
    """
    # hole: {
    #   id, name: hid, hname
    #   mid_point: pid                          <-- one tris' mid_points
    #   area: float                             <-- sum of tris.area
    #   boundary: list of edges + orientation   <-- get_boundary
    # }
    #
    # How to detect holes:
    # BFS or UNION/FIND? --> BFS looks better
    #
    snames = list(dict.fromkeys(snames))        # remove duplicates if any, preserve order
    tnames = filter_by_stype(snames, ST_TRI)
    hnames = filter_by_stype(snames, ST_HOLE)

    if not holes or not holes_bd:
        holes = holes_row()
        holes_bd = boundary_row()
        hnames = []
        snames = tnames

    if not tnames and not hnames:
        return None

    # n = len(snames)
    HID_MAX = holes["id"].max()
    HID_MAX = HID_MAX + 1 if HID_MAX and pd.notna(HID_MAX) else 1
    PID_MAX = points['id'].max()
    PID_MAX = PID_MAX + 1 if PID_MAX and pd.notna(PID_MAX) else 1

    # mids = points[points['mp_pointer'].notna()].set_index("id", drop=False)
    # points = points[points['mp_pointer'].isna()].set_index("id", drop=False)
    # edges = edges.set_index("name", drop=False)
    # tris = tris.set_index("name", drop=False).loc[tnames]
    # holes = holes.set_index("name", drop=False).loc[hnames]

    mids = _set_log_index(points[points['mp_pointer'].notna()], "id", "gen_holes : mids")
    # points = set_my_index(points[points['mp_pointer'].isna()], "id", "gen_holes : pts")
    edges = _set_log_index(edges, "name", "gen_holes : edges")
    tris = _set_log_index(tris, "name", "gen_holes : tris").loc[tnames]
    holes = _set_log_index(holes, "name", "gen_holes : holes").loc[hnames]

    # construct boundaries ------------------------------------------
    # holes_bd = holes_bd.set_index("sname", drop=False).loc[hnames].set_index("eid", drop=False)
    # tris_bd = gen_tris_boundary(None, tris, drop_zeros=False)
    # bd = add_boundaries(tris_bd, holes_bd, drop_zeros=False, ignore_zeros=False)
    bd = gen_boundaries((tnames, hnames), tris, holes_bd, drop_zeros=False, ignore_zeros=False)
    bd_all_points = get_points(bd['ename'], edges)
    bd_inner = bd.loc[bd.ort == 0]
    bd = bd.loc[bd.ort != 0]
    bd_points = get_points(bd['ename'], edges)
    print(f"[[bd]] : inner={bd_inner.shape} : outer={bd.shape}")

    # to remove -----------------------------------------------------
    to_remove = dict(
        points=[], mids=[], edges=[], tris=tnames, holes=hnames
    )
    to_remove['edges'] = bd_inner['ename'].tolist()
    inner_points = get_points(to_remove['edges'], edges)
    # to_remove['points'] = list(set(bd_all_points) - set(inner_points))
    to_remove['points'] = list(set(inner_points) - set(bd_points))
    # to_remove['mids'] = get_mid_points(to_remove['edges'], edges) + \
    #                     get_mid_points(tnames, tris) + get_mid_points(hnames, holes)
    to_remove['mids'] = get_mid_points_by_stype(to_remove['edges'] + tnames + hnames, edges, tris, holes)

    # group tris and holes into new holes ---------------------------
    to_newholes = {sname: hid for hid, sname in enumerate(snames, start=HID_MAX)}
    for eid, ename in bd_inner['ename'].iteritems():
        cfp, cfn = edges.loc[ename, ["cof_pos", "cof_neg"]]
        print(f"[[bd_inner]] : [{ename}] : cf_pos={cfp} : cf_neg={cfn}")
        if not cfp or not cfn:
            raise ValueError(f"Inner edge[{eid}] doesn't have a coface", ename, cfp, cfn)
        if cfp not in to_newholes.keys() or cfn not in to_newholes.keys():
            raise ValueError(f"Cofaces of an inner edge[{eid}] are not in the list of 2D simplices",
                             ename, cfp, cfn, snames)

        hp, hn = to_newholes[cfp], to_newholes[cfn]
        hp, hn = min(hp, hn), max(hp, hn)
        to_newholes[cfp], to_newholes[cfn] = hp, hp     # now both cofaces are in one hole

    newholes_dict = dict()
    for sname, hid in to_newholes.items():
        newholes_dict.setdefault(hid, []).append(sname)

    # construct newholes and newholes_bd ----------------------------
    newholes = holes_row()
    newholes_bd = boundary_row()
    newholes_mids = points_row()
    for hid, snames in newholes_dict.items():
        hmids = get_mid_points_by_stype(snames, edges, tris, holes)
        hmid = get_centralish_point(hmids, mids)
        newholes_mids, PID_MAX, mpid = append_mid_point(
            newholes_mids, PID_MAX, hmid['x'], hmid['y'], hid, ST_HOLE
        )

        harea = get_total_area(snames, tris, holes)
        newholes = newholes.append(holes_row(
            int(hid), get_hname(hid), int(mpid), harea, None
        ))
        newholes_bd = newholes_bd.append(
            gen_boundaries(snames, tris, holes_bd, drop_zeros=True, ignore_zeros=True)
        )
        pass

    newholes = fix_type_holes(newholes)
    newholes_mids = fix_type_points(newholes_mids)
    return newholes, newholes_bd, newholes_mids, to_remove



def get_points(snames, simp_df):
    """
    Get points of edges or tris.
        :param snames: enames or tnames
        :param simp_df: edges or tris
        :return: list of pids
    """
    # simp_df = simp_df.set_index('name', drop=False)
    simp_df = _set_log_index(simp_df, 'name', "get_points : simp_df")
    pid_1 = simp_df.loc[snames, 'pid_1'].unique().tolist()
    pid_2 = simp_df.loc[snames, 'pid_2'].unique().tolist()
    pid_3 = [] if 'pid_3' not in simp_df.columns else simp_df.loc[snames, 'pid_3'].unique().tolist()
    # pids = list(set(pid_1 + pid_2))
    pids = list(dict.fromkeys(pid_1 + pid_2 + pid_3))
    return pids


def get_points_holes(hnames, holes_bd, edges):
    # enames = holes_bd.set_index("name", drop=False).loc[hnames, "ename"].tolist()
    enames = _set_log_index(holes_bd, "name", "get_pts_holes : holes_bd").loc[hnames, "ename"].tolist()
    return get_points(enames, edges)


def get_points_by_stype(snames, edges, tris, holes_bd, as_dict=False, as_tuple=False):
    _, enames, tnames, hnames = filter_by_stypes(snames)

    epoints = get_points(enames, edges)
    tpoints = get_points(tnames, tris)
    hpoints = get_points_holes(tnames, holes_bd, edges)

    if as_dict:
        return {ST_EDGE: epoints, ST_TRI: tpoints, ST_HOLE: hpoints}
    if as_tuple:
        return epoints, tpoints, hpoints
    return epoints + tpoints + hpoints


def get_mid_points(snames, simp_df):
    """
    Get mid points of edges, tris, and holes
        :param snames: enames, tnames, hnames
        :param simp_df: edges, tris, holes
        :return: pids of mids
    """
    if 'mid_point' not in simp_df.columns:
        raise ValueError("get_mid_point: simd_df doesn't have \'mid_point\' column", simp_df.columns)

    # simp_df = simp_df.set_index('name', drop=False)
    simp_df = _set_log_index(simp_df, 'name', "get_mid_pts : simp_df")
    mids = simp_df.loc[snames, 'mid_point'].tolist()
    return mids


def get_mid_points_by_stype(snames, edges, tris, holes, as_dict=False, as_tuple=False):
    _, enames, tnames, hnames = filter_by_stypes(snames, by_points=False)

    emids = get_mid_points(enames, edges)
    tmids = get_mid_points(tnames, tris)
    hmids = get_mid_points(hnames, holes)

    if as_dict:
        return {ST_EDGE: emids, ST_TRI: tmids, ST_HOLE: hmids}
    if as_tuple:
        return emids, tmids, hmids
    return emids + tmids + hmids


def get_centralish_point(pids, points):
    # expects: points.index == pid
    points = points.loc[pids]
    x, y = points["x"], points["y"]
    x0, y0 = x.mean(), y.mean()
    dist = (x - x0)**2 + (y - y0)**2
    pid = dist.idxmin()
    return points.loc[pid]


def get_areas(snames, simp_df):
    if 'area' not in simp_df.columns:
        raise ValueError("get_areas: simd_df doesn't have \'area\' column", simp_df.columns)

    # simp_df = simp_df.set_index('name', drop=False)
    simp_df = _set_log_index(simp_df, 'name', "get_areas : simp_df")
    areas = simp_df.loc[snames, 'area'].tolist()
    return areas


def get_areas_by_stype(snames, tris, holes, as_dict=False, as_tuple=False):
    _, _, tnames, hnames = filter_by_stypes(snames, by_points=False, by_edges=False)

    tareas = get_areas(tnames, tris)
    hareas = get_areas(hnames, holes)

    if as_dict:
        return {ST_TRI: tareas, ST_HOLE: hareas}
    if as_tuple:
        return tareas, hareas
    return tareas + hareas


def get_total_area(snames, tris, holes):
    areas = get_areas_by_stype(snames, tris, holes)
    total_area = np.sum(areas)
    return total_area


def get_neighbours(sname, edges, tris, holes, holes_bd, from_snames=None, allow_outer=True) -> dict:
    """
    For a triangle or/and hole ``sname``
    returns a list  of their neighbors from the ``from_snames`` list.
    If ``from_snames`` is None then can use any triangles or holes
        :param sname:
        :param edges:       edges df
        :param tris:        tris df
        :param holes:       holes df ... or their boundaries????
        :param holes_bd:    boundaries of holes
        :param from_snames:
        :param allow_outer:
        :return: a dict of neighboring holes and triangles indexed by the shared edge
    """

    # a simplex/cell given by ``sname`` shouldn't count
    # None - outer void
    # if not from_snames:
    #     from_snames = tris['name'].to_list() + holes['name'].to_list()
    if from_snames:
        outer = {None} if allow_outer else set()
        from_snames = set(from_snames) - {sname} | outer


    def get_tri_neighbors(tname, edges, tris, from_snames):
        tri = _set_log_index(tris, 'name', "get_tri_nbr : tris").loc[tname]
        tri_edges = tri[["eid_1", "eid_2", "eid_3"]]
        eid_1, eid_2, eid_3 = tri_edges[0], tri_edges[1], tri_edges[2]
        eid_1, eid_2, eid_3 = get_enames([eid_1, eid_2, eid_3])

        # edges = edges.set_index("name", drop=False)
        edges = _set_log_index(edges, "name", "get_tri_nbr : edges")
        cofaces = edges.loc[[eid_1, eid_2, eid_3], ["name", "cof_pos", "cof_neg"]]

        tri_neighbors = dict()
        for index, cf_row in cofaces.iterrows():
            ename, cf_pos, cf_neg = cf_row['name'], cf_row['cof_pos'], cf_row['cof_neg']
            cf = cf_pos if cf_pos != tname else cf_neg
            if not from_snames or cf in from_snames:
                tri_neighbors[ename] = cf

        return tri_neighbors


    def get_hole_neighbors(hname, edges, holes_bd, from_snames):
        # TODO: hole neighbors
        return None

    dispatcher_dict = {ST_TRI: get_tri_neighbors, ST_HOLE: get_hole_neighbors}

    # ---------------------------------------------------------------
    stype = get_stype(sname)
    if stype in dispatcher_dict.keys():
        return dispatcher_dict[stype](sname, edges, tris, from_snames)

    return {}






def gen_plotly_edges(points, edges, edge_colors=(COLORS_DICT[SC_EDGES], COLORS_DICT[SC_EDGES_MID])):
    edge_color, edge_mid_color = edge_colors

    # points = points.set_index("id", drop=False)
    points = _set_log_index(points, "id", "gen_plotly_edges : points")
    plotly_edges = plotly_row()
    plotly_edges_mid = plotly_row()

    for index, edge in edges.iterrows():
        p1 = points.loc[edge['pid_1']]
        p2 = points.loc[edge['pid_2']]
        mp = points.loc[edge['mid_point']]
        if mp['mp_pointer'] != edge['name'] or edge['mid_point'] != mp['id']:
            raise ValueError("mid_point ids/pointers are wrong", edge, mp)

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

    plotly_edges = fix_type_plotly(plotly_edges)
    plotly_edges_mid = fix_type_plotly(plotly_edges_mid)
    return plotly_edges, plotly_edges_mid


def gen_plotly_tris(points, tris, tri_colors=(COLORS_DICT[SC_TRIS], COLORS_DICT[SC_TRIS_MID])):
    tri_color, tri_mid_color = tri_colors
    # print(f"<<gen_plotly_tris>> : {tri_color, tri_mid_color}")

    # points = points.set_index("id", drop=False)
    points = _set_log_index(points, "id", "gen_plotly_tris : points")
    plotly_tris = plotly_row()
    plotly_tris_mid = plotly_row()

    for index, tri in tris.iterrows():
        p1 = points.loc[tri['pid_1']]
        p2 = points.loc[tri['pid_2']]
        p3 = points.loc[tri['pid_3']]
        mp = points.loc[tri['mid_point']]
        if mp['mp_pointer'] != tri['name'] or tri['mid_point'] != mp['id']:
            raise Exception("mid point ids are wrong", tri, mp)
        plotly_tris = pd.concat([
            plotly_tris,
            plotly_row(tri['id'], tri['name'], p1['x'], p1['y'], tri_color, tri['customdata']),
            plotly_row(tri['id'], tri['name'], p2['x'], p2['y'], tri_color, tri['customdata']),
            plotly_row(tri['id'], tri['name'], p3['x'], p3['y'], tri_color, tri['customdata']),
            # plotly_row(p1['name'], tri['name'], p1['x'], p1['y'], tri_color, tri['customdata']),
            # plotly_row(p2['name'], tri['name'], p2['x'], p2['y'], tri_color, tri['customdata']),
            # plotly_row(p3['name'], tri['name'], p3['x'], p3['y'], tri_color, tri['customdata']),
            plotly_none_row(tri['id'], tri['name'])
        ])

        plotly_tris_mid = pd.concat([
            plotly_tris_mid,
            plotly_row(mp['id'], tri['name'], mp['x'], mp['y'], tri_mid_color, tri['customdata'])
        ])

    plotly_tris = fix_type_plotly(plotly_tris)
    plotly_tris_mid = fix_type_plotly(plotly_tris_mid)
    return plotly_tris, plotly_tris_mid


def gen_plotly_hl(names, plotly_df, hl_color):
    """
    Assigns the highlight color ``hl_color`` in ``plotly_df``
    to rows corresponding to ``names``
        :param names:
        :param plotly_df:
        :param hl_color:
        :return:
    """
    # plotly_df = plotly_df.set_index("name", drop=False)
    plotly_df = _set_log_index(plotly_df, "name", "gen_plotly_hl : plotly_df")
    plotly_df = plotly_df.loc[names]
    if not plotly_df.empty:
        plotly_df.loc[:, "color"] = hl_color
    return plotly_df


def get_stats(snames: list):
    # TODO: get_stats : number of elements, area/length..
    # number of elements, area/length..
    pass

# -----------------------------------------------------------------------------
#           MAKE TRACES / PLOTS
# -----------------------------------------------------------------------------
def get_trace(trace_name) -> go.Scatter:
    global main_figure, main_data
    if trace_name not in SC_TRACES:
        raise ValueError("get_trace: unknown trace name", trace_name)

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
    plotly_points = plotly_points[plotly_points['mp_pointer'].isna()]

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

    # print("\n>>> plotly tris <<<")
    # print(f"{plotly_tris['id']}")
    # print("\n-------------------")

    main_figure.update_traces(dict(
        ids=plotly_tris['name'],
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

    # tr_tris_mid = get_trace(SC_TRIS_MID)
    # print(f"+++ upd tris mid +++ : {tr_tris_mid.mode} : {tr_tris_mid.x[0:3]}")
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
def Random_Cloud(n, **kwargs) -> go.Figure:
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


def Triangulate(**kwargs) -> go.Figure:
    """
    Generate a triangulation for existing cloud
    and draw it on the figure
        :param kwargs:
        :return: main_figure
    """
    global main_figure, main_data
    if 'data' not in main_data['points']:
        return get_main_figure()

    points_df = main_data['points']['data']
    points_df = points_df[points_df['mp_pointer'].isna()]

    tri = Delaunay(points_df[["x", "y"]])
    points_dict, edges_dict, tris_dict = gen_triangulation_data(tri.simplices, points_df, **kwargs)
    main_data['points'].update(points_dict)
    main_data['edges'].update(edges_dict)
    main_data['tris'].update(tris_dict)

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
    plotly_pts = pts_data['plotly']
    # plotly_pts = plotly_pts[plotly_pts['mp_pointer'].isna()].set_index('name')
    # plotly_pts = set_my_index(plotly_pts[plotly_pts['mp_pointer'].isna()], 'name', "highlight_pts : plotly_pts")
    plotly_pts = plotly_pts[plotly_pts['mp_pointer'].isna()]

    marker_size = pd.Series([pts_data['marker_size']] * pts_data['n'], index=plotly_pts['name'])
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
    boundary = get_tris_boundary(tri_names, tris_df)

    pos_boundary = boundary.loc[boundary.ort == 1]
    neg_boundary = boundary.loc[boundary.ort == -1]
    # print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    # print(f"tris= [{tri_names}]")
    # print("------------------------------------------")
    # print(f"### boundary ### \n {boundary}")
    # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

    highlight_edges(pos_boundary['ename'],
                    hl_color=pos_color, sc_name=SC_EDGES_HL)
    highlight_edges(neg_boundary['ename'],
                    hl_color=neg_color, sc_name=SC_EDGES_NEG_HL)

    return get_main_figure()


# ------- wish list ----
def show_edge_orientations():
    pass


def gen_edges_boundary(enames):
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
