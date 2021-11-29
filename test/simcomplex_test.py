import unittest
import simcomplex as scx


class TestSum(unittest.TestCase):
    def test_temp(self):
        self.assertEqual(sum([1, 2, 3]), 6, "Should be 6")

    def test_gen_triangulation_data(self):
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

    def test_triangulate(self):
        x, y = scx.gen_random_points(4)
        pts_upd = scx.gen_points_data((x, y))
        scx.main_data['points'].update(pts_upd)

        points = scx.main_data['points']['data']
        l1 = len(points.index)

        fig = scx.triangulate()
        points = scx.main_data['points']['data']
        l2 = len(points.index)
        self.assertIsNotNone(fig, "fig is none")
        self.assertGreater(l2, l1, "mid points should be added")
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