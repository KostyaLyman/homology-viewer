import unittest

import pandas as pd

import simcomplex as scx


class TestSum(unittest.TestCase):
    def test_temp(self):
        self.assertEqual(sum([1, 2, 3]), 6, "Should be 6")

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


    def test_gen_triangulation_data(self):
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



    def test_triangulate(self):
        scx.setup_fig()

        x = [0, 2, 0, 3]
        y = [0, 0, 4, 1]
        pts_upd = scx.gen_points_data((x, y))
        scx.main_data['points'].update(pts_upd)

        points = scx.main_data['points']['data']
        l1 = len(points.index)

        fig = scx.triangulate()
        points = scx.main_data['points']['data']
        l2 = len(points.index)
        self.assertIsNotNone(fig, "fig is none")
        self.assertGreater(l2, l1, "mid points should be added")

        tr_edges = scx.get_trace(scx.SC_EDGES)
        self.assertIsNotNone(tr_edges)
        self.assertEqual(len(tr_edges.x), 5*3, "not all edges are on the trace")

        print(points.loc[:, points.columns != "customdata"])


    def test_random_cloud(self):
        scx.setup_fig()
        fig = scx.random_cloud(10, xlim=11, ylim=(2, 5), color="red")
        self.assertIsNotNone(fig, "fig is none")
        self.assertEqual(fig.layout.xaxis.range, (0, 11))
        self.assertEqual(fig.layout.yaxis.range, (2, 5))
        # scx.gen_triangulation_data(points_df)
        # (0, 0), (2, 0), (0, 4), (3, 1)


if __name__ == '__main__':
    unittest.main()