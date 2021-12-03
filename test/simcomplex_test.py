import unittest

import numpy as np
import pandas as pd

import simcomplex as scx


class TestSimcomplex(unittest.TestCase):
    def test_temp(self):
        self.assertEqual(sum([1, 2, 3]), 6, "Should be 6")

    def test_as_range(self):
        lim = scx.as_range(3)
        self.assertEqual(lim, (0, 3), "single int input")

        lim = scx.as_range(3.5)
        self.assertEqual(lim, (0, 3.5), "single float input")

        lim = scx.as_range((3.3, 4.4))
        self.assertEqual(lim, (3.3, 4.4), "tuple of floats, a < b")

        lim = scx.as_range((4.4, 3.3))
        self.assertEqual(lim, (3.3, 4.4), "tuples of floats, a > b")

        lim = scx.as_range((3, 4))
        self.assertEqual(lim, (3, 4), "tuple of ints, a < b")

        lim = scx.as_range((4, 3))
        self.assertEqual(lim, (3, 4), "tuples of ints, a > b")

    def test_random_cloud(self):
        scx.setup_fig()
        N_POINTS = 10
        fig = scx.random_cloud(n=N_POINTS, xlim=11, ylim=(2, 5), color="red")
        self.assertIsNotNone(fig, "fig is none")
        self.assertEqual(fig.layout.xaxis.range, (0, 11))
        self.assertEqual(fig.layout.yaxis.range, (2, 5))

        tr_points = scx.get_trace(scx.SC_POINTS)
        self.assertIsNotNone(tr_points, "trace is None")
        self.assertEqual(N_POINTS, len(tr_points.x), "not all points are in trace")



    def test_gen_triangulation_data_run(self):
        x = [0, 2, 0, 3]
        y = [0, 0, 4, 1]
        pts_data = scx.gen_points_data((x, y))
        print("\n---- POINTS -----")
        print(pts_data['data'][["id", "name", "x", "y", "mid_point", "mid_type"]])
        print("")
        edges, tris, mids = scx.gen_triangulation_df(pts_data['data'])

        print("\n---- EDGES -----")
        print(edges[["id", "name", "pid_1", "pid_2",
                     "cof_pos", "cof_neg",
                     # "cof_pos_type", "cof_neg_type"
                     "mid_point", "length"
                     ]])
        print(list(edges["customdata"]))

        print("\n---- TRIS -----")
        print(tris[["id", "name",
                    "pid_1", "pid_2", "pid_3",
                    "eid_1", "eid_2", "eid_3",
                    "ort_1", "ort_2", "ort_3",
                    "mid_point", "area"]])
        print(tris[["customdata"]])

        print("\n---- MIDS -----")
        print(mids[["id", "name", "x", "y", "mid_point", "mid_type"]])



    def test_triangulation_data_edges(self):
        N_EDGES, N_TRIS, N_MIDS = 5, 2, 7
        x = [0, 2, 0, 3]
        y = [0, 0, 4, 1]

        pts_data = scx.gen_points_data((x, y))
        points_df = pts_data['data']
        edges_df, tris_df, mids_df = scx.gen_triangulation_df(points_df)
        self.assertEqual(len(edges_df.index), 5)
        self.assertEqual(len(tris_df.index), 2)
        self.assertEqual(len(mids_df.index), 7)

        points_df = pd.concat([points_df, mids_df], ignore_index=True)
        points_df = points_df.set_index("id", drop=False)

        for index, edge in edges_df.iterrows():
            print(f"idx={index}, edge_name={edge['name']}")
            edge_id = edge['id']
            edge_mp = edge['mid_point']
            mp = points_df.loc[edge_mp]
            mp_id = mp['id']
            mp_mp = mp['mid_point']
            print(f"edge: {edge[['id', 'mid_point']]}")
            print(f"mids: {mp[['id', 'mid_point', 'mid_type']]}")
            self.assertIsNotNone(mp_mp, "mid_point doesn't have a mid_point pointer")
            self.assertEqual(edge_mp, mp_id, "edge[mp] == mp[id]")
            self.assertEqual(edge_id, mp_mp, "edge[id] == mp[mp]")


    def test_triangulation_data_tris(self):
        N_EDGES, N_TRIS, N_MIDS = 5, 2, 7
        x = [0, 2, 0, 3]
        y = [0, 0, 4, 1]
        pts_data = scx.gen_points_data((x, y))
        points_df = pts_data['data']
        edges_df, tris_df, mids_df = scx.gen_triangulation_df(points_df)

        self.assertEqual(len(edges_df.index), N_EDGES, "number of edges in triangulation data is wrong")
        self.assertEqual(len(tris_df.index), N_TRIS, "number of triangles in triangulation data is wrong")
        self.assertEqual(len(mids_df.index), N_MIDS, "number of mid_points in triangulation data is wrong")

        points_df = pd.concat([points_df, mids_df], ignore_index=True)
        points_df = points_df.set_index("id", drop=False)

        for index, tri in tris_df.iterrows():
            print(f"idx={index}, tri_name={tri['name']}")
            tri_id = tri['id']
            tri_mp = tri['mid_point']
            mp = points_df.loc[tri_mp]
            mp_id = mp['id']
            mp_mp = mp['mid_point']
            print(f"tri: {tri[['id', 'mid_point']]}")
            print(f"mids: {mp[['id', 'mid_point', 'mid_type']]}")
            self.assertIsNotNone(mp_mp, "mid_point doesn't have a mid_point pointer")
            self.assertEqual(tri_mp, mp_id, "edge[mp] == mp[id]")
            self.assertEqual(tri_id, mp_mp, "edge[id] == mp[mp]")


    def test_plotly_edges(self):
        N_EDGES = 5
        x = [0, 2, 0, 3]
        y = [0, 0, 4, 1]
        pts_data = scx.gen_points_data((x, y))
        points_df = pts_data['data']
        edges_df, tris_df, mids_df = scx.gen_triangulation_df(points_df)
        points_df = pd.concat([points_df, mids_df], ignore_index=True)

        plotly_edges, plotly_edges_mid = scx.gen_plotly_edges(points_df, edges_df)
        self.assertEqual(N_EDGES * 3, len(plotly_edges.index), "not all edges are in the plotly data")
        self.assertEqual(N_EDGES, len(plotly_edges_mid.index), "not all edges are in the plotly data")
        self.assertTrue(
            (edges_df.reset_index()['customdata'] ==
             plotly_edges_mid.reset_index()['customdata']).all(),
            "edges' customdata is different from mid points' customdata"
        )
        self.assertCountEqual(
            plotly_edges_mid['customdata'],
            edges_df['customdata'],
            "edges' customdata is different from mid points' customdata"
        )


    def test_plotly_tris(self):
        N_TRIS = 2
        x = [0, 2, 0, 3]
        y = [0, 0, 4, 1]
        pts_data = scx.gen_points_data((x, y))
        points_df = pts_data['data']
        edges_df, tris_df, mids_df = scx.gen_triangulation_df(points_df)
        points_df = pd.concat([points_df, mids_df], ignore_index=True)

        plotly_tris, plotly_tris_mid = scx.gen_plotly_tris(points_df, tris_df)
        self.assertEqual(N_TRIS * 4, len(plotly_tris.index), "not all edges are in the plotly data")
        self.assertEqual(N_TRIS, len(plotly_tris_mid.index), "not all edges are in the plotly data")


    def test_plotly_hl_edges(self):
        N_EDGES = 5
        x = [0, 2, 0, 3]
        y = [0, 0, 4, 1]
        pts_data = scx.gen_points_data((x, y))
        points_df = pts_data['data']
        edges_df, tris_df, mids_df = scx.gen_triangulation_df(points_df)
        points_df = pd.concat([points_df, mids_df], ignore_index=True)
        plotly_edges, plotly_edges_mid = scx.gen_plotly_edges(points_df, edges_df)

        # regular call
        test_edges = ['e0002', 'e0004']
        hl_color = "red"
        hl_colors = [hl_color] * len(test_edges) * 3

        plotly_hl = scx.gen_plotly_hl(test_edges, plotly_edges, hl_color)
        self.assertIsNotNone(plotly_hl)
        self.assertEqual(len(test_edges) * 3, len(plotly_hl.index), "highlighting size is wrong")
        self.assertCountEqual(np.repeat(test_edges, 3), plotly_hl['name'])
        self.assertCountEqual(hl_colors, plotly_hl["color"], "colors are wrong")

        # empty call
        plotly_hl = scx.gen_plotly_hl([], plotly_edges, hl_color)
        self.assertTrue(plotly_hl.empty)


    def test_plotly_hl_tris(self):
        N_TRIS = 4
        x = [0, 2, 0, 3, 4]
        y = [0, 0, 4, 1, -2]
        pts_data = scx.gen_points_data((x, y))
        points_df = pts_data['data']
        edges_df, tris_df, mids_df = scx.gen_triangulation_df(points_df)
        points_df = pd.concat([points_df, mids_df], ignore_index=True)
        plotly_tris, plotly_tris_mid = scx.gen_plotly_tris(points_df, tris_df)

        # test --------------------------------------------
        test_tris = ['t0002', 't0004']
        hl_color = "red"
        hl_colors = [hl_color] * len(test_tris) * 4

        plotly_hl = scx.gen_plotly_hl(test_tris, plotly_tris, hl_color)
        self.assertIsNotNone(plotly_hl)
        self.assertEqual(len(test_tris) * 4, len(plotly_hl.index), "highlighting size is wrong")
        self.assertCountEqual(np.repeat(test_tris, 4), plotly_hl['name'])
        self.assertCountEqual(hl_colors, plotly_hl["color"], "colors are wrong")

    def test_triangulate(self):
        # points -------------------------------------------
        N_POINTS, N_EDGES, N_TRIS, N_MIDS = 4, 5, 2, 7
        x = [0, 2, 0, 3]
        y = [0, 0, 4, 1]

        # setup -------------------------------------------
        scx.setup_fig()
        pts_upd = scx.gen_points_data((x, y))
        scx.main_data['points'].update(pts_upd)

        points = scx.main_data['points']['data']
        l1 = len(points.index)

        # triangulate -------------------------------------
        fig = scx.triangulate()

        # test --------------------------------------------
        points = scx.main_data['points']['data']
        l2 = len(points.index)
        self.assertIsNotNone(fig, "fig is none")
        self.assertGreater(l2, l1, "mid points should be added")
        print(points.loc[:, points.columns != "customdata"])

        # test EDGES trace
        tr_edges = scx.get_trace(scx.SC_EDGES)
        self.assertIsNotNone(tr_edges)
        self.assertEqual(N_EDGES * 3, len(tr_edges.x), "not all edges are in the trace")
        self.assertIsNone(tr_edges.customdata, "customdata on edges")

        # test EDGES_MID trace
        tr_edges_mid = scx.get_trace(scx.SC_EDGES_MID)
        self.assertIsNotNone(tr_edges_mid)
        self.assertEqual(N_EDGES, len(tr_edges_mid.x), "not all edges_mid are in the trace")
        self.assertIsNotNone(tr_edges_mid.customdata, "no customdata on edge mids")

        # test TRIS trace
        tr_tris = scx.get_trace(scx.SC_TRIS)
        self.assertIsNotNone(tr_tris)
        self.assertEqual(N_TRIS * 4, len(tr_tris.x), "not all tris are in the trace")
        self.assertIsNone(tr_edges.customdata, "customdata on triangles")

        # test TRIS_MID trace
        tr_tris_mid = scx.get_trace(scx.SC_TRIS_MID)
        self.assertIsNotNone(tr_tris_mid)
        self.assertEqual(N_TRIS, len(tr_tris_mid.x), "not all tris_mid are in the trace")
        self.assertIsNotNone(tr_tris_mid.customdata, "no customdata on tris mids")

    def test_show_hide_points(self):
        # setup / constants -------------------------------
        msg_visible = "Test[{test}] : <{sc}> should visible"
        msg_hidden = "Test[{test}] : <{sc}> should visible"
        sc_show = {
            scx.ST_POINT: scx.SC_POINTS,
            scx.ST_EDGE: scx.SC_EDGES_MID,
            scx.ST_TRI: scx.SC_TRIS_MID,
        }

        scx.setup_fig()

        # test routines -----------------------------------
        def test_visible(test_st):
            sc_on = sc_show[test_st]
            tr_on = scx.get_trace(sc_on)
            self.assertIsNotNone(tr_on, "trace is None")
            self.assertTrue(tr_on.visible, msg_visible.format(test=test_st, sc=sc_on))

        def test_hidden_except_of(test_st):
            sc_off = [sc for st, sc in sc_show.items() if st != test_st]
            tr_off = scx.get_traces(sc_off)
            for tr, sc in zip(tr_off, sc_off):
                self.assertFalse(tr.visible, msg_hidden.format(test=test_st, sc=sc))

        # test: points
        scx.show_hide_points(scx.ST_POINT)
        test_visible(scx.ST_POINT)
        test_hidden_except_of(scx.ST_POINT)

        # test: edges
        scx.show_hide_points(scx.ST_EDGE)
        test_visible(scx.ST_EDGE)
        test_hidden_except_of(scx.ST_EDGE)

        # test: tris
        scx.show_hide_points(scx.ST_TRI)
        test_visible(scx.ST_TRI)
        test_hidden_except_of(scx.ST_TRI)

        # test: none
        scx.show_hide_points('none')
        test_hidden_except_of('none')


    def test_highlight_edges(self):
        # points -------------------------------------------
        N_POINTS, N_EDGES, N_TRIS, N_MIDS = 4, 5, 2, 7
        x = [0, 2, 0, 3]
        y = [0, 0, 4, 1]

        # setup /triangulate ------------------------------
        scx.setup_fig()
        pts_upd = scx.gen_points_data((x, y))
        scx.main_data['points'].update(pts_upd)
        scx.triangulate()

        # test: default sc --------------------------------
        test_edges = ["e0002", "e0004"]
        hl_color = "red"
        scx.highlight_edges(test_edges, hl_color=hl_color)

        tr_edges_hl = scx.get_trace(scx.SC_EDGES_HL)
        self.assertIsNotNone(tr_edges_hl, "no trace")
        self.assertEqual(len(test_edges) * 3, len(tr_edges_hl.x), "not all edges are highlighted")
        self.assertCountEqual(np.repeat(test_edges, 3), tr_edges_hl.ids, "not all edges are highlighted")
        self.assertEqual(hl_color, tr_edges_hl.line.color, "line color")

        tr_edges_neg_hl = scx.get_trace(scx.SC_EDGES_NEG_HL)
        self.assertIsNotNone(tr_edges_neg_hl, "no trace")
        self.assertFalse(tr_edges_neg_hl.x, "extra highlighting")

        tr_tris_hl = scx.get_trace(scx.SC_TRIS_HL)
        self.assertIsNotNone(tr_tris_hl, "no trace")
        self.assertFalse(tr_tris_hl.x, "extra highlighting")

        # test: different sc ------------------------------
        test_edges = ["e0001", "e0004", "e0003"]
        test_sc = scx.SC_EDGES_NEG_HL
        hl_color = "blue"
        scx.highlight_edges(test_edges, hl_color=hl_color, sc_name=test_sc)

        tr_edges_hl = scx.get_trace(scx.SC_EDGES_HL)
        self.assertIsNotNone(tr_edges_hl, "no trace")
        # self.assertFalse(tr_edges_hl.x, "extra highlighting")
        self.assertCountEqual([], tr_edges_hl.x, "extra highlighting")

        tr_edges_neg_hl = scx.get_trace(scx.SC_EDGES_NEG_HL)
        self.assertIsNotNone(tr_edges_neg_hl, "no trace")
        self.assertEqual(len(test_edges) * 3, len(tr_edges_neg_hl.x), "not all edges are highlighted")
        self.assertEqual(len(test_edges) * 3, len(tr_edges_neg_hl.x), "not all edges are highlighted")
        self.assertCountEqual(np.repeat(test_edges, 3), tr_edges_neg_hl.ids, "not all edges are highlighted")
        self.assertEqual(hl_color, tr_edges_neg_hl.line.color, "line color")

        tr_tris_hl = scx.get_trace(scx.SC_TRIS_HL)
        self.assertIsNotNone(tr_tris_hl, "no trace")
        self.assertFalse(tr_tris_hl.x, "extra highlighting")


    def test_clear_highlighting(self):
        # points -------------------------------------------
        N_POINTS, N_EDGES, N_TRIS, N_MIDS = 4, 5, 2, 7
        x = [0, 2, 0, 3]
        y = [0, 0, 4, 1]

        # setup /triangulate ------------------------------
        scx.setup_fig()
        pts_upd = scx.gen_points_data((x, y))
        scx.main_data['points'].update(pts_upd)
        scx.triangulate()

        # highlight edges ---------------------------------
        test_edges = ["e0002", "e0004"]
        scx.highlight_edges(test_edges)

        # test --------------------------------------------
        scx.clear_highlighting()

        tr_edges_hl = scx.get_trace(scx.SC_EDGES_HL)
        self.assertIsNotNone(tr_edges_hl, "no edges_hl")
        self.assertCountEqual([], tr_edges_hl.ids, "some edges are still highlighted")

        tr_edges_neg_hl = scx.get_trace(scx.SC_EDGES_NEG_HL)
        self.assertIsNotNone(tr_edges_neg_hl, "no edges_neg_hl")
        self.assertCountEqual([], tr_edges_neg_hl.ids, "some edges are still highlighted")

        tr_tris_hl = scx.get_trace(scx.SC_TRIS_HL)
        self.assertIsNotNone(tr_tris_hl, "no tris_hl")
        self.assertCountEqual([], tr_tris_hl.ids, "some edges are still highlighted")


if __name__ == '__main__':
    unittest.main()