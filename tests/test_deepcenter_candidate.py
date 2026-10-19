import unittest

import numpy as np

from biohub_lab.deepcenter_candidate import (
    add_center_rescue_nodes,
    block_mean_xy,
    find_center_peaks,
    pooled_peak_to_original,
    rank_center_rescue_candidates,
)


class DeepCenterCandidateTests(unittest.TestCase):
    @staticmethod
    def _node(node_id, timepoint, x):
        return {
            "id": node_id,
            "dataset": "sample",
            "row_type": "node",
            "node_id": node_id,
            "t": timepoint,
            "z": 0,
            "y": 0,
            "x": x,
            "source_id": -1,
            "target_id": -1,
        }

    @staticmethod
    def _edge(row_id, source_id, target_id):
        return {
            "id": row_id,
            "dataset": "sample",
            "row_type": "edge",
            "node_id": -1,
            "t": -1,
            "z": -1,
            "y": -1,
            "x": -1,
            "source_id": source_id,
            "target_id": target_id,
        }

    def test_block_mean_xy_preserves_z_and_averages_xy_blocks(self):
        volume = np.arange(2 * 4 * 4, dtype=np.float32).reshape(2, 4, 4)

        pooled = block_mean_xy(volume, factor=2)

        self.assertEqual(pooled.shape, (2, 2, 2))
        np.testing.assert_allclose(
            pooled[0],
            np.asarray([[2.5, 4.5], [10.5, 12.5]], dtype=np.float32),
        )

    def test_pooled_peak_maps_to_original_voxel_coordinates(self):
        self.assertEqual(
            pooled_peak_to_original((3, 4, 5), pool_factor=4),
            (3.0, 17.5, 21.5),
        )

    def test_find_center_peaks_is_thresholded_and_non_max_suppressed(self):
        heatmap = np.zeros((3, 5, 5), dtype=np.float32)
        heatmap[1, 2, 2] = 0.9
        heatmap[1, 2, 3] = 0.8
        heatmap[2, 4, 4] = 0.7

        peaks, scores = find_center_peaks(heatmap, threshold=0.6, min_distance=1)

        np.testing.assert_array_equal(peaks, np.asarray([[1, 2, 2], [2, 4, 4]]))
        np.testing.assert_allclose(scores, np.asarray([0.9, 0.7], dtype=np.float32))

    def test_rank_rescue_candidates_requires_endpoint_proximity_and_nonduplicate(self):
        peaks = [
            {"t": 4, "z": 1.0, "y": 10.0, "x": 10.0, "score": 0.9},
            {"t": 4, "z": 1.0, "y": 30.0, "x": 30.0, "score": 0.95},
            {"t": 4, "z": 1.0, "y": 10.0, "x": 11.0, "score": 0.99},
        ]
        endpoints = [
            {"t": 4, "z": 1.0, "y": 10.0, "x": 10.0},
            {"t": 4, "z": 1.0, "y": 30.0, "x": 30.0},
        ]
        existing = [{"t": 4, "z": 1.0, "y": 10.0, "x": 10.5}]

        candidates = rank_center_rescue_candidates(
            peaks,
            endpoint_positions=endpoints,
            existing_nodes=existing,
            max_endpoint_distance_um=7.0,
            min_existing_distance_um=2.0,
        )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["x"], 30.0)

    def test_rejects_invalid_peak_gate_parameters(self):
        with self.assertRaises(ValueError):
            find_center_peaks(np.zeros((2, 2, 2), dtype=np.float32), threshold=float("nan"))
        with self.assertRaises(ValueError):
            rank_center_rescue_candidates([], max_endpoint_distance_um=-1.0)

    def test_add_center_rescue_uses_unique_adjacent_parent_and_keeps_topology(self):
        rows = [
            self._node(10, 0, 0),
            self._node(11, 1, 1),
            self._edge(2, 10, 11),
        ]
        peaks = [{"t": 2, "z": 0.0, "y": 0.0, "x": 2.0, "score": 0.9}]

        rescued, report = add_center_rescue_nodes(
            rows,
            peaks,
            frame_count=3,
            spatial_shape=(4, 8, 8),
            max_endpoint_distance_um=1.0,
            min_existing_distance_um=0.1,
        )

        nodes = [row for row in rescued if row["row_type"] == "node"]
        edges = [row for row in rescued if row["row_type"] == "edge"]
        self.assertEqual(len(nodes), 3)
        self.assertEqual([(row["source_id"], row["target_id"]) for row in edges], [(10, 11), (11, 12)])
        self.assertEqual(report["added_nodes"], 1)
        self.assertEqual(report["added_edges"], 1)
        self.assertEqual([row["id"] for row in rescued], list(range(len(rescued))))

    def test_add_center_rescue_rejects_peak_near_existing_target(self):
        rows = [
            self._node(10, 0, 0),
            self._node(11, 1, 1),
            self._node(12, 2, 2),
            self._edge(3, 10, 11),
            self._edge(4, 11, 12),
        ]
        peaks = [{"t": 2, "z": 0.0, "y": 0.0, "x": 2.0, "score": 0.99}]

        rescued, report = add_center_rescue_nodes(
            rows,
            peaks,
            frame_count=3,
            spatial_shape=(4, 8, 8),
            max_endpoint_distance_um=1.0,
            min_existing_distance_um=1.0,
        )

        self.assertEqual(
            [{key: value for key, value in row.items() if key != "id"} for row in rescued],
            [{key: value for key, value in row.items() if key != "id"} for row in rows],
        )
        self.assertEqual(report["added_nodes"], 0)


if __name__ == "__main__":
    unittest.main()
