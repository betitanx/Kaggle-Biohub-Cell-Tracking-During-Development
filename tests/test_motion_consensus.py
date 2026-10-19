import unittest

from biohub_lab.motion_consensus import add_motion_consensus_links


def _node(node_id, timepoint, x, dataset="sample"):
    return {
        "id": node_id,
        "dataset": dataset,
        "row_type": "node",
        "node_id": node_id,
        "t": timepoint,
        "z": 0,
        "y": 0,
        "x": x,
        "source_id": -1,
        "target_id": -1,
    }


def _edge(row_id, source_id, target_id, dataset="sample"):
    return {
        "id": row_id,
        "dataset": dataset,
        "row_type": "edge",
        "node_id": -1,
        "t": -1,
        "z": -1,
        "y": -1,
        "x": -1,
        "source_id": source_id,
        "target_id": target_id,
    }


class MotionConsensusTests(unittest.TestCase):
    def test_adds_link_only_when_nearest_and_velocity_agree(self):
        rows = [
            _node(1, 0, 0),
            _node(2, 1, 1),
            _node(3, 2, 2),
            _node(4, 2, 8),
            _edge(4, 1, 2),
        ]

        result = add_motion_consensus_links(
            rows,
            max_distance_um=2.0,
            max_prediction_error_um=2.0,
        )

        edges = [row for row in result if row["row_type"] == "edge"]
        self.assertEqual(
            [(row["source_id"], row["target_id"]) for row in edges],
            [(1, 2), (2, 3)],
        )
        self.assertEqual([row["id"] for row in result], list(range(len(result))))

    def test_rejects_disagreement_between_nearest_and_velocity_prediction(self):
        rows = [
            _node(1, 0, 0),
            _node(2, 1, 5),
            _node(3, 2, 6),
            _node(4, 2, 10),
            _edge(4, 1, 2),
        ]

        result = add_motion_consensus_links(
            rows,
            max_distance_um=10.0,
            max_prediction_error_um=10.0,
        )

        self.assertEqual(
            [(row["source_id"], row["target_id"]) for row in result if row["row_type"] == "edge"],
            [(1, 2)],
        )

    def test_respects_existing_parent_and_child_capacity(self):
        rows = [
            _node(1, 0, 0),
            _node(2, 1, 1),
            _node(3, 1, 2),
            _node(4, 2, 2),
            _node(5, 2, 3),
            _edge(5, 1, 2),
            _edge(6, 1, 3),
            _edge(7, 2, 4),
            _edge(8, 3, 5),
        ]

        result = add_motion_consensus_links(rows, max_distance_um=2.0)

        self.assertEqual(
            [(row["source_id"], row["target_id"]) for row in result if row["row_type"] == "edge"],
            [(1, 2), (1, 3), (2, 4), (3, 5)],
        )

    def test_requires_an_existing_parent_for_independent_velocity_signal(self):
        rows = [_node(1, 0, 0), _node(2, 1, 1)]

        result = add_motion_consensus_links(rows, max_distance_um=2.0)

        self.assertEqual(
            [row for row in result if row["row_type"] == "edge"],
            [],
        )
        fallback = add_motion_consensus_links(
            rows,
            max_distance_um=2.0,
            require_unique_parent=False,
        )
        self.assertEqual(
            [(row["source_id"], row["target_id"]) for row in fallback if row["row_type"] == "edge"],
            [(1, 2)],
        )

    def test_preserves_legal_gap_edge_and_uses_only_adjacent_parent_for_velocity(self):
        rows = [
            _node(1, 0, 0),
            _node(2, 1, 1),
            _node(3, 2, 2),
            _node(4, 3, 3),
            _edge(4, 1, 3),
            _edge(5, 3, 4),
        ]

        result = add_motion_consensus_links(
            rows,
            max_distance_um=2.0,
            max_prediction_error_um=2.0,
        )

        self.assertEqual(
            [(row["source_id"], row["target_id"]) for row in result if row["row_type"] == "edge"],
            [(1, 3), (3, 4)],
        )
        self.assertEqual([row["id"] for row in result], list(range(len(result))))

    def test_rejects_invalid_parameters(self):
        with self.assertRaises(ValueError):
            add_motion_consensus_links([], max_distance_um=-1.0)
        with self.assertRaises(ValueError):
            add_motion_consensus_links([], max_prediction_error_um=float("nan"))
        with self.assertRaises(ValueError):
            add_motion_consensus_links([], max_out_degree=3)


if __name__ == "__main__":
    unittest.main()
