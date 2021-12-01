import unittest

import pandas as pd

import simcomplex as scx


class TestSimcomplex(unittest.TestCase):
    def test_temp(self):
        self.assertEqual(sum([1, 2, 3]), 6, "Should be 6")

    def test_random_cloud(self):
        scx.setup_fig()
        fig = scx.random_cloud(10, xlim=11, ylim=(2, 5), color="red")
        self.assertIsNotNone(fig, "fig is none")
        self.assertEqual(fig.layout.xaxis.range, (0, 11))
        self.assertEqual(fig.layout.yaxis.range, (2, 5))

        tr_points = scx.get_trace(scx.SC_POINTS)

        # scx.gen_triangulation_data(points_df)
        # (0, 0), (2, 0), (0, 4), (3, 1)


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
        x = [0, 2, 0, 3]
        y = [0, 0, 4, 1]
        pts_data = scx.gen_points_data((x, y))
        points_df = pts_data['data']
        edges_df, tris_df, mids_df = scx.gen_triangulation_df(points_df)

        # expected values
        N_EDGES, N_TRIS, N_MIDS = 5, 2, 7
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
        self.assertEqual(len(plotly_edges.index), N_EDGES * 3, "not all edges are in the plotly data")
        self.assertEqual(len(plotly_edges_mid.index), N_EDGES, "not all edges are in the plotly data")
        self.assertTrue(
            (edges_df.reset_index()['customdata'] ==
             plotly_edges_mid.reset_index()['customdata']).all(),
            "edges' customdata is different from mid points' customdata"
        )
        self.assertCountEqual(
            edges_df['customdata'], plotly_edges_mid['customdata'],
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
        self.assertEqual(len(plotly_tris.index), N_TRIS * 4, "not all edges are in the plotly data")
        self.assertEqual(len(plotly_tris_mid.index), N_TRIS, "not all edges are in the plotly data")



    def test_triangulate(self):
        scx.setup_fig()

        N_POINTS, N_EDGES, N_TRIS, N_MIDS = 4, 5, 2, 7
        x = [0, 2, 0, 3]
        y = [0, 0, 4, 1]
        pts_upd = scx.gen_points_data((x, y))
        scx.main_data['points'].update(pts_upd)

        points = scx.main_data['points']['data']
        l1 = len(points.index)

        # triangulate -------------------------------------
        fig = scx.triangulate()

        points = scx.main_data['points']['data']
        l2 = len(points.index)
        self.assertIsNotNone(fig, "fig is none")
        self.assertGreater(l2, l1, "mid points should be added")
        print(points.loc[:, points.columns != "customdata"])

        # test EDGES trace
        tr_edges = scx.get_trace(scx.SC_EDGES)
        self.assertIsNotNone(tr_edges)
        self.assertEqual(len(tr_edges.x), N_EDGES * 3, "not all edges are in the trace")
        self.assertIsNone(tr_edges.customdata, "customdata on edges")

        # test EDGES_MID trace
        tr_edges_mid = scx.get_trace(scx.SC_EDGES_MID)
        self.assertIsNotNone(tr_edges_mid)
        self.assertEqual(len(tr_edges_mid.x), N_EDGES, "not all edges_mid are in the trace")
        self.assertIsNotNone(tr_edges_mid.customdata, "no customdata on edge mids")

        # test TRIS trace
        tr_tris = scx.get_trace(scx.SC_TRIS)
        self.assertIsNotNone(tr_tris)
        self.assertEqual(len(tr_tris.x), N_TRIS * 4, "not all tris are in the trace")
        self.assertIsNone(tr_edges.customdata, "customdata on triangles")

        # test TRIS_MID trace
        tr_tris_mid = scx.get_trace(scx.SC_TRIS_MID)
        self.assertIsNotNone(tr_tris_mid)
        self.assertEqual(len(tr_tris_mid.x), N_TRIS, "not all tris_mid are in the trace")
        self.assertIsNotNone(tr_tris_mid.customdata, "no customdata on tris mids")



if __name__ == '__main__':
    unittest.main()