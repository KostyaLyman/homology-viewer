import unittest

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay

import simcomplex as scx


class TestSimcomplex(unittest.TestCase):

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

    def test_parse_name(self):
        valid_names = ["p0012", "e0123", "t1234", "h4321"]
        expected_parse = [("p", 12), ("e", 123), ("t", 1234), ("h", 4321)]
        invalid_names = ["q0012", "e4ee"]

        # valid tests
        for sname, expected in zip(valid_names, expected_parse):
            exp_type, exp_id = expected
            stype, sid = scx.parse_sname(sname)
            self.assertEqual(exp_type, stype, "type doesn't match")
            self.assertEqual(exp_id, sid, "ids do not match")

        # invalid tests
        for sname in invalid_names:
            stype, sid = scx.parse_sname(sname)
            self.assertIsNone(stype, "type should be empty")
            self.assertEqual(0, sid, "id should be 0")



    def test_random_cloud(self):
        # test data ---------------------------------------
        N_POINTS = 10
        xlim = xmin, xmax = (0, 11)
        ylim = ymin, ymax = (2, 5)

        # test random cloud -------------------------------
        scx.setup_fig()
        fig = scx.Random_Cloud(n=N_POINTS, xlim=xmax, ylim=ylim, color="red")
        self.assertIsNotNone(fig, "fig is none")
        self.assertEqual(xlim, fig.layout.xaxis.range, "xlim is wrong")
        self.assertEqual(ylim, fig.layout.yaxis.range, "ylim is wrong")

        tr_points = scx.get_trace(scx.SC_POINTS)
        self.assertIsNotNone(tr_points, "trace is None")
        self.assertEqual(N_POINTS, len(tr_points.ids), "not all points are in trace")
        self.assertTrue((tr_points.x >= xmin).all(), "expecting: x >= xmin")
        self.assertTrue((tr_points.x <= xmax).all(), "expecting: x <= xmax")
        self.assertTrue((tr_points.y >= ymin).all(), "expecting: y >= ymin")
        self.assertTrue((tr_points.y <= ymax).all(), "expecting: y <= ymax")

        pts_data = scx.main_data['points']
        self.assertIn('data', pts_data)
        self.assertIn('plotly', pts_data)
        
        points = pts_data['data']
        self.assertIsNotNone(points)
        self.assertEqual(N_POINTS, len(points.index), "not all points are in points data")
        self.assertTrue((points['x'] >= xmin).all(), "expecting: x >= xmin")
        self.assertTrue((points['x'] <= xmax).all(), "expecting: x <= xmax")
        self.assertTrue((points['y'] >= ymin).all(), "expecting: y >= ymin")
        self.assertTrue((points['y'] <= ymax).all(), "expecting: y <= ymax")

        pass

    def test_triangulate(self):
        """
        Test whether ``scx.triangulate`` creates and fills corresponding traces
        """
        # points #1 -------------------------------------------------
        N_POINTS, N_EDGES, N_TRIS, N_MIDS = 4, 5, 2, 7
        x = [0, 2, 0, 3]
        y = [0, 0, 4, 1]

        # setup -----------------------------------------------------
        scx.setup_fig()
        pts_upd = scx.gen_points_data((x, y))
        scx.main_data['points'].update(pts_upd)

        points = scx.main_data['points']['data']
        l1 = len(points.index)

        # triangulate -----------------------------------------------
        fig = scx.Triangulate()

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



    def test_gen_triangulation_data_run(self):
        # points #1 -------------------------------------------------
        N_POINTS, N_EDGES, N_TRIS = 4, 5, 2
        x = [0, 2, 0, 3]
        y = [0, 0, 4, 1]

        pts_data = scx.gen_points_data((x, y))
        print("\n---- POINTS -----")
        print(pts_data['data'][["id", "name", "x", "y", "mp_pointer", "mp_type"]])
        print("")
        tri = Delaunay(pts_data['data'][["x", "y"]])
        edges, tris, mids = scx.gen_triangulation_df(tri.simplices, pts_data['data'])

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
        print(mids[["id", "name", "x", "y", "mp_pointer", "mp_type"]])


    def test_gen_triangulation_data_edges(self):
        """
        Tests whether ``scx.gen_triangulation_data()`` fills correctly
        the edge info:
            - number of edges
            - mid points
            - cofaces
            - orientation
        """
        # points #1 -------------------------------------------------
        N_POINTS, N_EDGES, N_TRIS, N_MIDS = 4, 5, 2, 7
        x = [0, 2, 0, 3]
        y = [0, 0, 4, 1]

        # setup -----------------------------------------------------
        pts_data = scx.gen_points_data((x, y))
        points_df = pts_data['data']

        # triangulate -----------------------------------------------
        tri = Delaunay(points_df[["x", "y"]])
        edges_df, tris_df, mids_df = scx.gen_triangulation_df(tri.simplices, points_df)
        self.assertEqual(N_EDGES, len(edges_df.index), "number of edges is wrong")
        self.assertEqual(N_TRIS, len(tris_df.index), "number of triangles is wrong")
        self.assertEqual(N_MIDS, len(mids_df.index), "number of mid points is wrong")

        points_df = pd.concat([points_df, mids_df], ignore_index=True)
        points_df = points_df.set_index("id", drop=False)
        tris_df = tris_df.set_index("name", drop=False)

        # test edges df ------------------------------------
        self.assertFalse((edges_df['cof_pos'].isnull()).any(), "all cof_pos should be assigned")
        self.assertTrue((edges_df['cof_neg'].isnull()).any(), "there are cof_neg == None")
        self.assertEqual(1, (edges_df['cof_neg'].notnull()).sum(), "there is  one cof_neg != None")

        for index, edge in edges_df.iterrows():
            # test mid_points
            eid, ename, edge_mp = edge['id'], edge['name'], edge['mid_point']
            mp = points_df.loc[edge_mp]
            mid, mp_pointer = mp['id'], mp['mp_pointer']
            self.assertIsNotNone(mp_pointer, "mid_point doesn't have a mid_point pointer")
            self.assertEqual(edge_mp, mid, "edge[mid_point] != mp[id]")
            self.assertEqual(ename, mp_pointer, "edge[name] != mp[mp_pointer]")

            # test cofaces / orientations
            cof_pos, cof_neg = edge['cof_pos'], edge['cof_neg']
            cof_pos_type, cof_neg_type = edge['cof_pos_type'], edge['cof_neg_type']
            self.assertIsNotNone(cof_pos, "positive coface always exists")
            # self.assertNotEqual(0, cof_pos, "positive coface always exists")
            self.assertIsNotNone(cof_pos_type, "positive coface always exists")
            self.assertEqual(scx.ST_TRI, cof_pos_type, "positive coface should be a triangle")

            tri_pos = tris_df.loc[cof_pos]
            tid_pos = tri_pos['id']
            tri_eids = tri_pos[['eid_1', 'eid_2', 'eid_3']]
            tri_orts = tri_pos[['ort_1', 'ort_2', 'ort_3']]
            self.assertIn(eid, tri_eids.values, f"triangle[{tid_pos}] doesn't contain edge[{eid}]")
            self.assertEqual(1, (tri_eids == eid).sum(), f"triangle[{tid_pos}] should have only one edge[{eid}]: {tri_eids}")
            eid_indicator = (tri_eids == eid).values
            self.assertEqual(1, tri_orts[eid_indicator][0], f"edge[{eid}] should be oriented positively in triangle[{tid_pos}]")


    def test_gen_triangulation_data_tris(self):
        """
        Tests whether ``scx.gen_triangulation_data()`` fills correctly
        the tris info:
            - number of edges
            - mid points
        """
        # points #1 -------------------------------------------------
        N_POINTS, N_EDGES, N_TRIS, N_MIDS = 4, 5, 2, 7
        x = [0, 2, 0, 3]
        y = [0, 0, 4, 1]

        # setup -----------------------------------------------------
        pts_data = scx.gen_points_data((x, y))
        points_df = pts_data['data']

        # triangulate -----------------------------------------------
        tri = Delaunay(points_df[["x", "y"]])
        edges_df, tris_df, mids_df = scx.gen_triangulation_df(tri.simplices, points_df)
        self.assertEqual(len(edges_df.index), N_EDGES, "number of edges in triangulation data is wrong")
        self.assertEqual(len(tris_df.index), N_TRIS, "number of triangles in triangulation data is wrong")
        self.assertEqual(len(mids_df.index), N_MIDS, "number of mid_points in triangulation data is wrong")

        points_df = pd.concat([points_df, mids_df], ignore_index=True)
        points_df = points_df.set_index("id", drop=False)

        # test tris df ------------------------------------
        for index, tri in tris_df.iterrows():
            tid, tname, tri_mp = tri['id'], tri['name'], tri['mid_point']
            mp = points_df.loc[tri_mp]
            mid, mp_pointer = mp['id'], mp['mp_pointer']
            self.assertIsNotNone(mp_pointer, "mid_point doesn't have a mid_point pointer")
            self.assertEqual(tri_mp, mid, "tri[mid_point] != mp[id]")
            self.assertEqual(tname, mp_pointer, "tri[name] != mp[mp_pointer]")


    def test_get_tris_boundary(self):
        # points #2 -------------------------------------------------
        N_POINTS, N_EDGES, N_TRIS = 5, 8, 4
        x = [0, 2, 0, 3, 4]
        y = [0, 0, 4, 1, -2]

        # setup / triangulate ---------------------------------------
        pts_data = scx.gen_points_data((x, y))
        points_df = pts_data['data']
        tri = Delaunay(points_df[["x", "y"]])
        edges_df, tris_df, mids_df = scx.gen_triangulation_df(tri.simplices, points_df)
        points_df = pd.concat([points_df, mids_df], ignore_index=True)

        # ===========================================================
        # test: no overlapping edges --------------------------------
        test_tris = ["t0001", "t0004"]
        expected_boundary = {
            "e0001": 1, "e0002": 1, "e0003": 1, "e0004": -1, "e0006": -1, "e0008": 1
        }
        boundary = scx.get_tris_boundary(test_tris, tris_df)
        self.assertEqual(len(expected_boundary), len(boundary.index),
                         "number of edges in the boundary[t1-t4] is wrong")
        self.assertFalse((boundary.ort == 0).any(),
                         "an edge with ort=0 is in the boundary[t1-t4]")
        self.assertFalse(set(expected_boundary.keys()) ^ set(boundary['ename']),
                         "there are some unexpected edges in the boundary[t1-t4]")
        self.assertTrue((abs(boundary.ort) == 1).all(),
                        "edges from boundary[t1-t4] should have coefs == ±1")
        self.assertCountEqual(expected_boundary.values(), boundary.ort,
                              "orientation of boundary edges is wrong")
        self.assertEqual("eid", boundary.index.name, "index name is wrong")

        # test: with overlapping edges ------------------------------
        test_tris = ["t0003", "t0001"]
        expected_boundary = {
            "e0002": 1, "e0003": 1, "e0006": 1, "e0007": 1
        }
        expected_snames = {
            "e0002": "t0001", "e0003": "t0001", "e0006": "t0003", "e0007": "t0003"
        }
        boundary = scx.get_tris_boundary(test_tris, tris_df)
        self.assertEqual(len(expected_boundary), len(boundary.index),
                         "number of edges in the boundary[t1-t3] is wrong")
        self.assertFalse((boundary['ort'] == 0).any(),
                         "an edge with ort=0 is in the boundary[t1-t3]")
        self.assertFalse(set(expected_boundary.keys()) ^ set(boundary['ename']),
                         "there are some unexpected edges in the boundary[t1-t3]")
        self.assertTrue((boundary['ort'] == 1).all(),
                        "edges from boundary[t1-t3] should have all coefs == +1")
        self.assertCountEqual(expected_boundary.values(), boundary['ort'],
                              "orientation of boundary edges is wrong")
        self.assertCountEqual(expected_snames.values(), boundary['sname'],
                              "orientation of boundary edges is wrong")
        self.assertEqual("eid", boundary.index.name, "index name is wrong")

        # TODO: test gen_tris_bd with repeating tnames
        pass


    def test_add_boundaries(self):
        # NO OVERLAP ================================================
        bd_1 = scx.boundary_row(
            eid=[12, 13, 14], ename=["e0012", "e0013", "e0014"],
            ort=[+1, -1, +1], sname=["t0012", "h0013", "t0014"]
        )
        bd_2 = scx.boundary_row(
            eid=[92, 93, 94], ename=["e0092", "e0093", "e0094"],
            ort=[-1, +1, -1], sname=["t0029", "h0099", "t0049"]
        )
        expected_bd = [12, 13, 14, 92, 93, 94]

        bd = scx.add_boundaries(bd_1, bd_2)
        self.assertIsNotNone(bd)
        self.assertEqual(len(expected_bd), bd.shape[0], "wrong size of bd")
        for eid in expected_bd:
            self.assertTrue(eid in bd.index, f"edge[{eid}] is not in the resulting boundary")
            self.assertNotEqual(0, bd.loc[eid, "ort"], f"edge[{eid}] has ort=0")

        self.assertEqual(3, bd_1.shape[0], "original boundary has changed")
        for eid in [12, 13, 14]:
            self.assertTrue(eid in bd_1.index)
        self.assertEqual(1, bd_1.loc[12, 'ort'], "original boundary has changed")
        self.assertEqual(-1, bd_1.loc[13, 'ort'], "original boundary has changed")
        self.assertEqual(1, bd_1.loc[14, 'ort'], "original boundary has changed")

        # WITH OVERLAP ================================================
        bd_1 = scx.boundary_row(
            eid=[12, 13, 14], ename=["e0012", "e0013", "e0014"],
            ort=[+1, -1, +1], sname=["t0012", "h0013", "t0014"]
        )
        bd_2 = scx.boundary_row(
            eid=[92, 13, 94], ename=["e0092", "e0013", "e0094"],
            ort=[-1, +1, -1], sname=["t0029", "h0099", "t0049"]
        )
        expected_bd = [12, 14, 92, 94]

        bd = scx.add_boundaries(bd_1, bd_2)
        self.assertIsNotNone(bd)
        self.assertEqual(len(expected_bd), bd.shape[0], "wrong size of bd")
        for eid in expected_bd:
            self.assertTrue(eid in bd.index, f"edge[{eid}] is not in the resulting boundary")
            self.assertNotEqual(0, bd.loc[eid, "ort"], f"edge[{eid}] has ort=0")

        self.assertEqual(3, bd_1.shape[0], "original boundary has changed")
        for eid in [12, 13, 14]:
            self.assertTrue(eid in bd_1.index)
        self.assertEqual(1, bd_1.loc[12, 'ort'], "original boundary has changed")
        self.assertEqual(-1, bd_1.loc[13, 'ort'], "original boundary has changed")
        self.assertEqual(1, bd_1.loc[14, 'ort'], "original boundary has changed")

        # with overlap and DROP_ZEROS -------------------------------
        expected_bd = [12, 13, 14, 92, 94]
        bd = scx.add_boundaries(bd_1, bd_2, drop_zeros=False)
        self.assertIsNotNone(bd)
        self.assertEqual(len(expected_bd), bd.shape[0], "wrong size of bd")
        for eid in expected_bd:
            self.assertTrue(eid in bd.index, f"edge[{eid}] is not in the resulting boundary")

        eid = 13
        self.assertEqual(0, bd.loc[eid, "ort"], f"edge[{eid}] should have ort=0")
        self.assertEqual("h0013h0099", bd.loc[eid, "sname"])

        # with overlap and MULTIPLICITIES > +1 ----------------------
        bd_1 = scx.boundary_row(
            eid=[12, 13, 14], ename=["e0012", "e0013", "e0014"],
            ort=[+1, -1, +2], sname=["t0012", "h0013", "t0014"]
        )
        bd_2 = scx.boundary_row(
            eid=[92, 13, 14], ename=["e0092", "e0013", "e0014"],
            ort=[-1, +2, -1], sname=["t0029", "h0099", "t0049"]
        )
        expected_bd = [12, 13, 14, 92]

        bd = scx.add_boundaries(bd_1, bd_2)
        self.assertIsNotNone(bd)
        self.assertEqual(len(expected_bd), bd.shape[0], "wrong size of bd")
        for eid in expected_bd:
            self.assertTrue(eid in bd.index, f"edge[{eid}] is not in the resulting boundary")

        eid = 13
        self.assertEqual(1, bd.loc[eid, "ort"], f"edge[{eid}] should have ort=1")
        self.assertEqual("h0099", bd.loc[eid, "sname"],
                         f"ort of edge[{eid}] has flipped, so sname should be from bd_2")

        eid = 14
        self.assertEqual(1, bd.loc[eid, "ort"], f"edge[{eid}] should have ort=1")
        self.assertEqual("t0014", bd.loc[eid, "sname"],
                         f"ort of edge[{eid}] has not flipped, so sname should be from bd_1")

        # with overlap and IGNORE_ZEROS=True ------------------------
        bd_1 = scx.boundary_row(
            eid=[12, 13, 14, 15], ename=["e0012", "e0013", "e0014", "e0015"],
            ort=[+1, -1, +1, 0], sname=["t0012", "h0013", "t0014", "t0015"]
        )
        bd_2 = scx.boundary_row(
            eid=[92, 13, 95, 94], ename=["e0092", "e0013", "e0095", "e0094"],
            ort=[-1, +1, 0, -1], sname=["t0029", "h0099", "t0095", "t0049"]
        )
        expected_bd = [12, 14, 92, 94]

        bd = scx.add_boundaries(bd_1, bd_2, ignore_zeros=True)
        self.assertIsNotNone(bd)
        self.assertEqual(len(expected_bd), bd.shape[0], "wrong size of bd")
        for eid in expected_bd:
            self.assertTrue(eid in bd.index, f"edge[{eid}] is not in the resulting boundary")
            self.assertNotEqual(0, bd.loc[eid, "ort"], f"edge[{eid}] has ort=0")

        self.assertEqual(4, bd_1.shape[0], "original boundary has changed")
        for eid in [12, 13, 14, 15]:
            self.assertTrue(eid in bd_1.index)
        self.assertEqual(1, bd_1.loc[12, 'ort'], "original boundary has changed")
        self.assertEqual(-1, bd_1.loc[13, 'ort'], "original boundary has changed")
        self.assertEqual(1, bd_1.loc[14, 'ort'], "original boundary has changed")
        self.assertEqual(0, bd_1.loc[15, 'ort'], "original boundary has changed")

        # with overlap and IGNORE_ZEROS=False ------------------------
        bd_1 = scx.boundary_row(
            eid=[12, 13, 14, 15], ename=["e0012", "e0013", "e0014", "e0015"],
            ort=[+1, -1, +1, 0], sname=["t0012", "h0013", "t0014", "t0015"]
        )
        bd_2 = scx.boundary_row(
            eid=[92, 13, 15, 94], ename=["e0092", "e0013", "e0015", "e0094"],
            ort=[-1, 0, 1, -1], sname=["t0029", "h0099", "t0088", "t0049"]
        )
        expected_bd = [12, 13, 14, 92, 94, 15]

        bd = scx.add_boundaries(bd_1, bd_2, ignore_zeros=False)
        self.assertIsNotNone(bd)
        self.assertEqual(len(expected_bd), bd.shape[0], "wrong size of bd")
        for eid in expected_bd:
            self.assertTrue(eid in bd.index, f"edge[{eid}] is not in the resulting boundary")
            self.assertNotEqual(0, bd.loc[eid, "ort"], f"edge[{eid}] has ort=0")

        self.assertEqual(4, bd_1.shape[0], "original boundary has changed")
        for eid in [12, 13, 14, 15]:
            self.assertTrue(eid in bd_1.index)
        self.assertEqual(1, bd_1.loc[12, 'ort'], "original boundary has changed")
        self.assertEqual(-1, bd_1.loc[13, 'ort'], "original boundary has changed")
        self.assertEqual(1, bd_1.loc[14, 'ort'], "original boundary has changed")
        self.assertEqual(0, bd_1.loc[15, 'ort'], "original boundary has changed")

        pass

    def test_gen_holes_from_tris(self):
        # points #3 -------------------------------------------------
        N_POINTS, N_EDGES, N_TRIS = 6, 10, 5
        x = [0, 2, 0, 3, 4, 2]
        y = [0, 0, 4, 1, -2, 4]

        # setup / triangulate ---------------------------------------
        # pts_data = scx.gen_points_data((x, y))
        points_df = scx.gen_points_data((x, y))['data']
        tri = Delaunay(points_df[["x", "y"]])
        edges_df, tris_df, mids_df = scx.gen_triangulation_df(tri.simplices, points_df)
        points_df = pd.concat([points_df, mids_df], ignore_index=True)
        data = dict(
            points=points_df, edges=edges_df, tris=tris_df, holes=None, holes_bd=None
        )

        # testing utility ---------------------------------------
        def assert_holes_df(self, test_tnames, expecations, holes, holes_bd, holes_mids, mids_df, to_remove):
            self.assertIsNotNone(holes, "holes are missing")
            self.assertIsNotNone(holes_bd, "holes_bd are missing")
            self.assertIsNotNone(holes_mids, "holes_mids are missing")
            self.assertIsNotNone(to_remove, "to_remove is missing")

            print(holes[holes.columns[:-1]])
            holes = holes.set_index("name", drop=False)
            holes_bd = holes_bd.set_index("ename", drop=False)
            holes_mids = holes_mids.set_index("id", drop=False)

            self.assertEqual(expecations["holes"]["len"], len(holes.index), "expecations[holes][len]")
            self.assertCountEqual(expecations["holes"]["area"], holes["area"].tolist(), "expecations[holes][area]")
            self.assertEqual(expecations["bd"]["len"], len(holes_bd.index), "expecations[bd][len]")
            for ename, ort in expecations["bd"]["ort"].items():
                self.assertEqual(ort, holes_bd.loc[ename, "ort"], f"expecations[bd][ort][{ename}]")
            self.assertEqual(expecations["mids"]["len"], len(holes_mids), "expecations[mids][len]")
            for i, (pid, hmid) in enumerate(holes_mids.iterrows()):
                pname = scx.get_pname(int(pid))
                self.assertNotIn(pname, mids_df["name"].tolist(), f"repeating mids name = [{i}: {pname}]")
                self.assertIn(pid, holes["mid_point"].tolist(), f"hmid [{i}: {pname}] not in holes")
                self.assertAlmostEqual(expecations["mids"]["x"][i], hmid["x"], 2, f"expecations[mids][x][{i}: {pname}]")
                self.assertAlmostEqual(expecations["mids"]["y"][i], hmid["y"], 2, f"expecations[mids][y][{i}: {pname}]")
            self.assertEqual(expecations["to_remove"]["points"], len(to_remove['points']),
                             "expecations[to_remove][points]")
            self.assertEqual(expecations["to_remove"]["mids"], len(to_remove['mids']), "expecations[to_remove][mids]")
            self.assertEqual(expecations["to_remove"]["edges"], len(to_remove['edges']),
                             "expecations[to_remove][edges]")
            self.assertEqual(expecations["to_remove"]["tris"], len(to_remove['tris']), "expecations[to_remove][tris]")
            self.assertEqual(expecations["to_remove"]["holes"], len(to_remove['holes']),
                             "expecations[to_remove][holes]")
            self.assertCountEqual(test_tnames, to_remove['tris'], f"to_remove[tris] ")

        # NO OVERLAP ================================================
        test_tnames = ["t0003", "t0002"]
        expecations = dict(
            holes={"len": 2, "area": [2, 4]},
            bd={"len": 6, "ort": {
                "e0001": -1, "e0006": +1, "e0007": +1, "e0004": +1, "e0005": +1, "e0002": -1
            }},
            mids={"len": 2,
                  "x": mids_df.set_index("mp_pointer").loc[test_tnames, "x"].tolist(),
                  "y": mids_df.set_index("mp_pointer").loc[test_tnames, "y"].tolist()
                  },
            to_remove={"points": 0, "mids": 2, "edges": 0, "tris": 2, "holes": 0}
        )
        # test:
        holes, holes_bd, holes_mids, to_remove = scx.gen_holes_df(test_tnames, **data)
        assert_holes_df(self, test_tnames, expecations, holes, holes_bd, holes_mids, mids_df, to_remove)

        # WITH OVERLAP ==============================================
        test_tnames = ["t0003", "t0005"]
        expecations = dict(
            holes={"len": 1, "area": [6]},
            bd={"len": 4, "ort": {
                "e0001": -1, "e0006": +1, "e0004": -1, "e0010": +1
            }},
            mids={"len": 1,
                  "x": mids_df.set_index("mp_pointer").loc[[test_tnames[1]], "x"].tolist(),
                  "y": mids_df.set_index("mp_pointer").loc[[test_tnames[1]], "y"].tolist()
                  },
            to_remove={"points": 0, "mids": 3, "edges": 1, "tris": 2, "holes": 0},
            to_remove_lists={"mids": ["p0014", "p0016", "p0020"], # t0003, e0007, t0005
                             "edges": ["e0007"], "tris": ["t0003", "t0005"]}
        )

        holes, holes_bd, holes_mids, to_remove = scx.gen_holes_df(test_tnames, **data)
        assert_holes_df(self, test_tnames, expecations, holes, holes_bd, holes_mids, mids_df, to_remove)
        self.assertCountEqual(expecations['to_remove_lists']['mids'], sorted(map(scx.get_pname, to_remove['mids'])),
                              "expecations[to_remove_lists][mids]")
        self.assertCountEqual(expecations['to_remove_lists']['edges'], to_remove['edges'],
                              "expecations[to_remove_lists][edges]")
        self.assertCountEqual(expecations['to_remove_lists']['tris'], to_remove['tris'],
                              "expecations[to_remove_lists][tris]")

        # WITH OVERLAP & POINT ======================================
        test_tnames = ["t0001", "t0002", "t0003", "t0005"]
        expecations = dict(
            holes={"len": 1, "area": [10]},
            bd={"len": 4, "ort": {
                "e0003": +1, "e0006": +1, "e0010": +1, "e0005": +1
            }},
            mids={"len": 1,
                  "x": mids_df.set_index("mp_pointer").loc[[test_tnames[0]], "x"].tolist(),
                  "y": mids_df.set_index("mp_pointer").loc[[test_tnames[0]], "y"].tolist()
                  },
            to_remove={"points": 1, "mids": 8, "edges": 4, "tris": 4, "holes": 0},
            to_remove_lists={
                "points": {"p0002"},
                "mids": {"p0008", "p0009", "p0012", "p0016", "p0007", "p0011", "p0014", "p0020"},
                "edges": {"e0001", "e0002", "e0004", "e0007"},
                "tris": {"t0001", "t0002", "t0003", "t0005"}
            }
        )

        holes, holes_bd, holes_mids, to_remove = scx.gen_holes_df(test_tnames, **data)
        assert_holes_df(self, test_tnames, expecations, holes, holes_bd, holes_mids, mids_df, to_remove)
        self.assertSetEqual(expecations['to_remove_lists']['points'], set(map(scx.get_pname, to_remove['points'])),
                              "expecations[to_remove_lists][points]")
        self.assertSetEqual(expecations['to_remove_lists']['mids'], set(map(scx.get_pname, to_remove['mids'])),
                              "expecations[to_remove_lists][mids]")
        self.assertSetEqual(expecations['to_remove_lists']['edges'], set(to_remove['edges']),
                              "expecations[to_remove_lists][edges]")
        self.assertCountEqual(expecations['to_remove_lists']['tris'], set(to_remove['tris']),
                              "expecations[to_remove_lists][tris]")

        pass



    def test_get_neighbours_of_tri_no_holes(self):
        # points #3 -------------------------------------------------
        N_POINTS, N_EDGES, N_TRIS = 6, 10, 5
        x = [0, 2, 0, 3, 4, 2]
        y = [0, 0, 4, 1, -2, 4]

        # setup / triangulate ---------------------------------------
        pts_data = scx.gen_points_data((x, y))
        points_df = pts_data['data']
        tri = Delaunay(points_df[["x", "y"]])
        edges_df, tris_df, mids_df = scx.gen_triangulation_df(tri.simplices, points_df)
        points_df = pd.concat([points_df, mids_df], ignore_index=True)

        # no holes --------------------------------------------------
        holes_df = scx.holes_row()
        holes_bd = scx.boundary_row()
        data = dict(
            edges=edges_df, tris=tris_df, holes=holes_df, holes_bd=holes_bd
        )

        # NO ``from_snames`` ========================================
        # test: inner triangle ----------------------------
        test_tri = "t0003"
        expected_nb = {
            "e0001": "t0001", "e0006": "t0004", "e0007": "t0005"
        }

        neighbors = scx.get_neighbours(test_tri, **data, from_snames=None)

        self.assertIsNotNone(neighbors, "neighbors are None")
        self.assertEqual(len(expected_nb), len(neighbors), "number of neighbors is wrong")
        self.assertSetEqual(set(expected_nb.values()), set(neighbors.values()), "some neighbors are missing/wrong")

        for ename, nb in expected_nb.items():
            self.assertTrue(ename in neighbors.keys(), f"{ename} is not in neighbors.keys()")
            self.assertEqual(nb, neighbors[ename], f"{ename} neighbor is wrong")
            # self.assertEqual()

        # test: outer triangle ----------------------------
        test_tri = "t0005"
        expected_nb = {
            "e0004": "t0002", "e0007": "t0003", "e0010": None
        }
        neighbors = scx.get_neighbours(test_tri, **data, from_snames=None)
        self.assertIsNotNone(neighbors, "neighbors are None")
        self.assertEquals(len(expected_nb), len(neighbors), "number of neighbors is wrong")
        self.assertSetEqual(set(expected_nb.values()), set(neighbors.values()), "some neighbors are missing/wrong")

        for ename, nb in expected_nb.items():
            self.assertTrue(ename in neighbors.keys(), f"{ename} is not in neighbors.keys()")
            self.assertEqual(nb, neighbors[ename], f"{ename} neighbor is wrong")


        # WITH ``from_snames`` ======================================
        # test: inner triangle ----------------------------
        test_tri = "t0003"
        test_list = ["t0004", "t0005"]
        expected_nb = {
            "e0006": "t0004", "e0007": "t0005"
        }

        neighbors = scx.get_neighbours(test_tri, **data, from_snames=test_list)

        self.assertIsNotNone(neighbors, "neighbors are None")
        self.assertEquals(len(expected_nb), len(neighbors), "number of neighbors is wrong")
        self.assertSetEqual(set(expected_nb.values()), set(neighbors.values()), "some neighbors are missing/wrong")

        for ename, nb in expected_nb.items():
            self.assertTrue(ename in neighbors.keys(), f"{ename} is not in neighbors.keys()")
            self.assertEqual(nb, neighbors[ename], f"{ename} neighbor is wrong")
            # self.assertEqual()

        # test: outer triangle ----------------------------
        test_tri = "t0005"
        test_list = ["t0005", "t0003"]
        expected_nb = {
            "e0007": "t0003", "e0010": None
        }

        neighbors = scx.get_neighbours(test_tri, **data, from_snames=test_list)

        self.assertIsNotNone(neighbors, "neighbors are None")
        self.assertEquals(len(expected_nb), len(neighbors), "number of neighbors is wrong")
        self.assertSetEqual(set(expected_nb.values()), set(neighbors.values()), "some neighbors are missing/wrong")

        for ename, nb in expected_nb.items():
            self.assertTrue(ename in neighbors.keys(), f"{ename} is not in neighbors.keys()")
            self.assertEqual(nb, neighbors[ename], f"{ename} neighbor is wrong")
            # self.assertEqual()

        pass

    def test_get_neighbouring_tris_of_hole(self):
        pass


    def test_get_points(self):
        # points #3 -------------------------------------------------
        N_POINTS, N_EDGES, N_TRIS = 6, 10, 5
        x = [0, 2, 0, 3, 4, 2]
        y = [0, 0, 4, 1, -2, 4]

        # setup / triangulate ---------------------------------------
        # pts_data = scx.gen_points_data((x, y))
        points_df = scx.gen_points_data((x, y))['data']
        tri = Delaunay(points_df[["x", "y"]])
        edges_df, tris_df, mids_df = scx.gen_triangulation_df(tri.simplices, points_df)
        points_df = pd.concat([points_df, mids_df], ignore_index=True)

        # test edges ------------------------------------------------
        test_edges = ["e0002", "e0004", "e0005"]
        expected_pids = [2, 4, 5]
        pids = sorted(scx.get_points(test_edges, edges_df))
        self.assertIsNotNone(pids)
        self.assertCountEqual(expected_pids, pids)

        # test tris -------------------------------------------------
        test_tris = ["t0004", "t0003", "t0005"]
        expected_pids = [1, 2, 3, 4, 6]
        pids = sorted(scx.get_points(test_tris, tris_df))
        self.assertIsNotNone(pids)
        self.assertCountEqual(expected_pids, pids)
        pass

    def test_get_centralish_point(self):
        x = [0, 2, -2, 1, -1]
        y = [0, 2, -2, 0, 0]

        points_df = scx.gen_points_data((x, y))['data']

        # test 1
        pids = [1, 2, 3, 4, 5]
        expected = {"x": 0, "y": 0}
        center = scx.get_centralish_point(pids, points_df)
        self.assertIsNotNone(center)
        self.assertTrue(type(center) is pd.Series, "center should be Series")
        self.assertEqual(expected["x"], center["x"], "x is wrong")
        self.assertEqual(expected["y"], center["y"], "y is wrong")

        # test 2
        pids = [2, 4, 5]
        expected = {"x": 1, "y": 0}
        center = scx.get_centralish_point(pids, points_df)
        self.assertIsNotNone(center)
        self.assertTrue(type(center) is pd.Series, "center should be Series")
        self.assertEqual(expected["x"], center["x"], "x is wrong")
        self.assertEqual(expected["y"], center["y"], "y is wrong")

        # test 3
        pids = [4, 5]
        expected = {"x": 1, "y": 0}
        center = scx.get_centralish_point(pids, points_df)
        self.assertIsNotNone(center)
        self.assertTrue(type(center) is pd.Series, "center should be Series")
        self.assertEqual(expected["x"], center["x"], "x is wrong")
        self.assertEqual(expected["y"], center["y"], "y is wrong")

        pass

    def test_hole_row(self):
        empty_hole_row = scx.holes_row()
        self.assertIsNotNone(empty_hole_row)
        self.assertEqual(0, len(empty_hole_row.index))

        hole_row = scx.holes_row(12, "h12", 33, 44, "hole-customdata")
        self.assertIsNotNone(hole_row)
        self.assertEqual(1, len(hole_row.index))

        hole_rows = scx.holes_row([12, 14], ["h12", "h14"], [33, 55], [44, 66], ["hole-customdata-1", "hole-customdata-2"])
        self.assertIsNotNone(hole_rows)
        self.assertEqual(2, len(hole_rows.index))

        hole_rows = scx.holes_row([12, 14], ["h12", "h14"], None, [44, 66], ["hole-customdata-1", "hole-customdata-2"])
        self.assertIsNotNone(hole_rows)
        self.assertEqual(2, len(hole_rows.index))
        pass

    def test_boundary_row(self):
        # empty boundary row
        empty_bd_row = scx.boundary_row()
        self.assertIsNotNone(empty_bd_row)
        self.assertEqual(0, len(empty_bd_row.index))

        # atomary inputs
        bd_row = scx.boundary_row(12, "e12", +1, "t33")
        self.assertIsNotNone(bd_row)
        self.assertEqual(1, len(bd_row.index))
        self.assertEqual(12, bd_row.loc[12, 'eid'], "eid is wrong")
        self.assertEqual("e12", bd_row.loc[12, 'ename'], "ename is wrong")
        self.assertEqual(1, bd_row.loc[12, 'ort'], "ort is wrong")
        self.assertEqual("t33", bd_row.loc[12, 'sname'], "sname is wrong")
        self.assertEqual("eid", bd_row.index.name, "index name is wrong")

        # list inputs
        bd_rows = scx.boundary_row(
            [12, 14], ["e12", "e14"], [+1, -1], ["t33", "t66"])
        self.assertIsNotNone(bd_rows)
        self.assertEqual(2, len(bd_rows.index))
        self.assertEqual(12, bd_rows.loc[12, 'eid'], "eid is wrong")
        self.assertEqual("e12", bd_rows.loc[12, "ename"], "ename is wrong")
        self.assertEqual(1, bd_rows.loc[12, "ort"], "ort is wrong")
        self.assertEqual("t33", bd_rows.loc[12, "sname"], "sname is wrong")
        self.assertEqual("eid", bd_rows.index.name, "index name is wrong")

        # list inputs with: sname <- None
        bd_rows = scx.boundary_row(
            [12, 14], ["e12", "e14"], [+1, -1], None)
        self.assertIsNotNone(bd_rows)
        self.assertEqual(2, len(bd_rows.index))
        self.assertEqual(12, bd_rows.loc[12, 'eid'], "eid is wrong")
        self.assertEqual("e12", bd_rows.loc[12, "ename"], "ename is wrong")
        self.assertEqual(1, bd_rows.loc[12, "ort"], "ort is wrong")
        self.assertIsNone(bd_rows.loc[12, "sname"], "sname is wrong")
        self.assertEqual("eid", bd_rows.index.name, "index name is wrong")

        # list inputs with: idx, idx_name <- sname
        # bd_rows = scx.boundary_row(
        #     [12, 14], ["e12", "e14"], [+1, -1], ["t33", "t66"],
        #     idx=["t33", "t66"], idx_name="sname"
        # )
        # self.assertIsNotNone(bd_rows)
        # self.assertEqual(2, len(bd_rows.index))
        # self.assertEqual(12, bd_rows.loc["t33", "eid"])
        # self.assertEqual("e12", bd_rows.loc["t33", "ename"])
        # self.assertEqual(1, bd_rows.loc["t33", "ort"], "ort is wrong")
        # self.assertEqual("t33", bd_rows.loc["t33", "sname"], "sname is wrong")
        # self.assertEqual("sname", bd_rows.index.name, "index name is wrong")
        #
        # # list inputs with: idx <- None, idx_name <- sname
        # bd_rows = scx.boundary_row(
        #     [12, 14], ["h12", "h14"], [+1, -1], ["t33", "t66"],
        #     idx=None, idx_name="sname"
        # )
        # self.assertIsNotNone(bd_rows)
        # self.assertEqual(2, len(bd_rows.index))
        # self.assertEqual("sname", bd_rows.index.name, "index name is wrong")
        pass

    def test_plotly_edges(self):
        # points #1 -------------------------------------------------
        N_POINTS, N_EDGES, N_TRIS = 4, 5, 2
        x = [0, 2, 0, 3]
        y = [0, 0, 4, 1]

        pts_data = scx.gen_points_data((x, y))
        points_df = pts_data['data']
        tri = Delaunay(points_df[["x", "y"]])
        edges_df, tris_df, mids_df = scx.gen_triangulation_df(tri.simplices, points_df)
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
        # points #1 -------------------------------------------------
        N_POINTS, N_EDGES, N_TRIS = 4, 5, 2
        x = [0, 2, 0, 3]
        y = [0, 0, 4, 1]

        pts_data = scx.gen_points_data((x, y))
        points_df = pts_data['data']
        tri = Delaunay(points_df[["x", "y"]])
        edges_df, tris_df, mids_df = scx.gen_triangulation_df(tri.simplices, points_df)
        points_df = pd.concat([points_df, mids_df], ignore_index=True)

        plotly_tris, plotly_tris_mid = scx.gen_plotly_tris(points_df, tris_df)
        self.assertEqual(N_TRIS * 4, len(plotly_tris.index), "not all edges are in the plotly data")
        self.assertEqual(N_TRIS, len(plotly_tris_mid.index), "not all edges are in the plotly data")


    def test_plotly_hl_edges(self):
        # points #1 -------------------------------------------------
        N_POINTS, N_EDGES, N_TRIS = 4, 5, 2
        x = [0, 2, 0, 3]
        y = [0, 0, 4, 1]

        # setup / triangulate ------------------------------
        pts_data = scx.gen_points_data((x, y))
        points_df = pts_data['data']
        tri = Delaunay(points_df[["x", "y"]])
        edges_df, tris_df, mids_df = scx.gen_triangulation_df(tri.simplices, points_df)
        points_df = pd.concat([points_df, mids_df], ignore_index=True)
        plotly_edges, plotly_edges_mid = scx.gen_plotly_edges(points_df, edges_df)

        # test: regular call
        test_edges = ['e0002', 'e0004']
        hl_color = "red"
        hl_colors = [hl_color] * len(test_edges) * 3

        plotly_hl = scx.gen_plotly_hl(test_edges, plotly_edges, hl_color)
        self.assertIsNotNone(plotly_hl)
        self.assertEqual(len(test_edges) * 3, len(plotly_hl.index), "highlighting size is wrong")
        self.assertCountEqual(np.repeat(test_edges, 3), plotly_hl['name'])
        self.assertCountEqual(hl_colors, plotly_hl["color"], "colors are wrong")

        # test: empty call
        plotly_hl = scx.gen_plotly_hl([], plotly_edges, hl_color)
        self.assertTrue(plotly_hl.empty)

    def test_plotly_hl_tris(self):
        # points #2 -------------------------------------------------
        N_POINTS, N_EDGES, N_TRIS = 5, 8, 4
        x = [0, 2, 0, 3, 4]
        y = [0, 0, 4, 1, -2]

        # setup / triangulate ------------------------------
        pts_data = scx.gen_points_data((x, y))
        points_df = pts_data['data']
        tri = Delaunay(points_df[["x", "y"]])
        edges_df, tris_df, mids_df = scx.gen_triangulation_df(tri.simplices, points_df)
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

    def test_show_hide_points_visibility(self):
        # setup / constants -------------------------------
        msg_visible = "Test[{test}] : <{sc}> should be visible"
        msg_hidden = "Test[{test}] : <{sc}> should be hidden"
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
        scx.show_hide_points(scx.ST_POINT, mode='visibility')
        test_visible(scx.ST_POINT)
        test_hidden_except_of(scx.ST_POINT)

        # test: edges
        scx.show_hide_points(scx.ST_EDGE, mode='visibility')
        test_visible(scx.ST_EDGE)
        test_hidden_except_of(scx.ST_EDGE)

        # test: tris
        scx.show_hide_points(scx.ST_TRI, mode='visibility')
        test_visible(scx.ST_TRI)
        test_hidden_except_of(scx.ST_TRI)

        # test: none
        scx.show_hide_points('none', mode='visibility')
        test_hidden_except_of('none')

    def test_show_hide_points_opacity(self):
        # setup / constants -------------------------------
        msg_visible = "Test[{test}] : <{sc}> should be visible"
        msg_hidden = "Test[{test}] : <{sc}> should be hidden"
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
            self.assertTrue(tr_on.marker.opacity, msg_visible.format(test=test_st, sc=sc_on))

        def test_hidden_except_of(test_st):
            sc_off = [sc for st, sc in sc_show.items() if st != test_st]
            tr_off = scx.get_traces(sc_off)
            for tr, sc in zip(tr_off, sc_off):
                self.assertFalse(tr.marker.opacity, msg_hidden.format(test=test_st, sc=sc))

        # test: points
        scx.show_hide_points(scx.ST_POINT, mode='opacity')
        test_visible(scx.ST_POINT)
        test_hidden_except_of(scx.ST_POINT)

        # test: edges
        scx.show_hide_points(scx.ST_EDGE, mode='o')
        test_visible(scx.ST_EDGE)
        test_hidden_except_of(scx.ST_EDGE)

        # test: tris
        scx.show_hide_points(scx.ST_TRI, mode='opacity')
        test_visible(scx.ST_TRI)
        test_hidden_except_of(scx.ST_TRI)

        # test: none
        scx.show_hide_points('none', mode='opacity')
        test_hidden_except_of('none')

    def test_highlight_points(self):
        # points #1 -------------------------------------------------
        N_POINTS, N_EDGES, N_TRIS, N_MIDS = 4, 5, 2, 7
        x = [0, 2, 0, 3]
        y = [0, 0, 4, 1]

        # setup /triangulate ------------------------------
        scx.setup_fig()
        pts_upd = scx.gen_points_data((x, y))
        scx.main_data['points'].update(pts_upd)
        scx.Triangulate()

        # test --------------------------------------------
        test_points = ["p0002", "p0004"]
        hl_mult = 3
        expected_marker_sizes = [scx.MARKER_SIZE, scx.MARKER_SIZE * hl_mult] * 2

        scx.highlight_points(test_points, hl_mult=hl_mult)

        tr_points = scx.get_trace(scx.SC_POINTS)
        self.assertIsNotNone(tr_points, "trace is None")
        self.assertCountEqual(expected_marker_sizes, tr_points.marker.size, "wrong sizes")
        pass

    def test_highlight_edges_std(self):
        # points #1 -------------------------------------------------
        N_POINTS, N_EDGES, N_TRIS, N_MIDS = 4, 5, 2, 7
        x = [0, 2, 0, 3]
        y = [0, 0, 4, 1]

        # setup /triangulate ------------------------------
        scx.setup_fig()
        pts_upd = scx.gen_points_data((x, y))
        scx.main_data['points'].update(pts_upd)
        scx.Triangulate()

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
        pass

    def test_highlight_edges_neg(self):
        # points #1 -------------------------------------------------
        N_POINTS, N_EDGES, N_TRIS, N_MIDS = 4, 5, 2, 7
        x = [0, 2, 0, 3]
        y = [0, 0, 4, 1]

        # setup /triangulate ----------------------------------------
        scx.setup_fig()
        pts_upd = scx.gen_points_data((x, y))
        scx.main_data['points'].update(pts_upd)
        scx.Triangulate()

        # test: neg_hl sc -------------------------------------------
        test_edges = ["e0001", "e0004", "e0003"]
        test_sc = scx.SC_EDGES_NEG_HL
        hl_color = "blue"
        scx.highlight_edges(test_edges, hl_color=hl_color, sc_name=test_sc)

        tr_edges_hl = scx.get_trace(scx.SC_EDGES_HL)
        self.assertIsNotNone(tr_edges_hl, "no trace")
        self.assertFalse(tr_edges_hl.x, "extra highlighting")
        # self.assertCountEqual([], tr_edges_hl.x, "extra highlighting")

        tr_edges_neg_hl = scx.get_trace(scx.SC_EDGES_NEG_HL)
        self.assertIsNotNone(tr_edges_neg_hl, "no trace")
        self.assertEqual(len(test_edges) * 3, len(tr_edges_neg_hl.x), "not all edges are highlighted")
        self.assertEqual(len(test_edges) * 3, len(tr_edges_neg_hl.x), "not all edges are highlighted")
        self.assertCountEqual(np.repeat(test_edges, 3), tr_edges_neg_hl.ids, "not all edges are highlighted")
        self.assertEqual(hl_color, tr_edges_neg_hl.line.color, "line color")

        tr_tris_hl = scx.get_trace(scx.SC_TRIS_HL)
        self.assertIsNotNone(tr_tris_hl, "no trace")
        self.assertFalse(tr_tris_hl.x, "extra highlighting")

        pass

    def test_highlight_triangles(self):
        # points #2 -------------------------------------------------
        N_POINTS, N_EDGES, N_TRIS = 5, 8, 4
        x = [0, 2, 0, 3, 4]
        y = [0, 0, 4, 1, -2]

        # setup / triangulate ---------------------------------------
        pts_data = scx.gen_points_data((x, y))
        points_df = pts_data['data']
        tri = Delaunay(points_df[["x", "y"]])
        edges_df, tris_df, mids_df = scx.gen_triangulation_df(tri.simplices, points_df)
        points_df = pd.concat([points_df, mids_df], ignore_index=True)
        # TODO: test_highlight_triangles
        pass

    def test_highlight_boundary_no_overlap(self):
        # points #2 -------------------------------------------------
        N_POINTS, N_EDGES, N_TRIS = 5, 8, 4
        x = [0, 2, 0, 3, 4]
        y = [0, 0, 4, 1, -2]

        # setup / triangulate ------------------------------
        scx.setup_fig()
        pts_upd = scx.gen_points_data((x, y))
        scx.main_data['points'].update(pts_upd)
        scx.Triangulate()

        # test: no overlapping edges ----------------------
        test_tris = ["t0001", "t0004"]
        expected_boundary = {
            "e0001": 1, "e0002": 1, "e0003": 1, "e0004": -1, "e0006": -1, "e0008": 1
        }
        scx.highlight_boundary(test_tris)

        tr_pos_hl = scx.get_trace(scx.SC_EDGES_HL)
        pos_bd = np.repeat([ename for ename, ort in expected_boundary.items() if ort == 1], 3)
        self.assertIsNotNone(tr_pos_hl, "no trace")
        self.assertEqual(len(pos_bd), len(tr_pos_hl.ids), "not all pos_boundary edges are highlighted")
        self.assertCountEqual(pos_bd, tr_pos_hl.ids, "not all pos_boundary edges are highlighted")

        tr_neg_hl = scx.get_trace(scx.SC_EDGES_NEG_HL)
        neg_bd = np.repeat([ename for ename, ort in expected_boundary.items() if ort != 1], 3)
        self.assertIsNotNone(tr_neg_hl, "no trace")
        self.assertEqual(len(neg_bd), len(tr_neg_hl.ids), "not all neg_boundary edges are highlighted")
        self.assertCountEqual(neg_bd, tr_neg_hl.ids, "not all neg_boundary edges are highlighted")
        pass

    def test_highlight_boundary_with_overlap(self):
        # points #2 -------------------------------------------------
        N_POINTS, N_EDGES, N_TRIS = 5, 8, 4
        x = [0, 2, 0, 3, 4]
        y = [0, 0, 4, 1, -2]

        # setup / triangulate ------------------------------
        scx.setup_fig()
        pts_upd = scx.gen_points_data((x, y))
        scx.main_data['points'].update(pts_upd)
        scx.Triangulate()

        # test: with overlapping edges --------------------
        test_tris = ["t0003", "t0001"]
        expected_boundary = {
            "e0002": 1, "e0003": 1, "e0006": 1, "e0007": 1
        }
        scx.highlight_boundary(test_tris)

        tr_pos_hl = scx.get_trace(scx.SC_EDGES_HL)
        pos_bd = np.repeat([ename for ename, ort in expected_boundary.items() if ort == 1], 3)
        self.assertIsNotNone(tr_pos_hl, "no trace")
        self.assertEqual(len(pos_bd), len(tr_pos_hl.ids), "not all pos_boundary edges are highlighted")
        self.assertCountEqual(pos_bd, tr_pos_hl.ids, "not all pos_boundary edges are highlighted")

        tr_neg_hl = scx.get_trace(scx.SC_EDGES_NEG_HL)
        # no negative edges in boundary
        self.assertIsNotNone(tr_neg_hl, "no trace")
        self.assertEqual(0, len(tr_neg_hl.ids), "some neg_boundary edges are highlighted")
        pass

    def test_clear_highlighting_after_edges(self):
        # points #1 -------------------------------------------------
        N_POINTS, N_EDGES, N_TRIS, N_MIDS = 4, 5, 2, 7
        x = [0, 2, 0, 3]
        y = [0, 0, 4, 1]

        # setup /triangulate ------------------------------
        scx.setup_fig()
        pts_upd = scx.gen_points_data((x, y))
        scx.main_data['points'].update(pts_upd)
        scx.Triangulate()

        # highlight edges: std ----------------------------
        test_edges = ["e0002", "e0004"]
        scx.highlight_edges(test_edges)

        # test
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

        # highlight edges: neg ----------------------------
        test_edges = ["e0001", "e0004", "e0003"]
        hl_color = "blue"
        scx.highlight_edges(test_edges, hl_color=hl_color, sc_name=scx.SC_EDGES_NEG_HL)

        # test
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
        pass

if __name__ == '__main__':
    unittest.main()