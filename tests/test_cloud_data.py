import unittest

from biohub_lab.cloud_data import dataset_prefix, dataset_prefixes, parse_file_listing_page


class CloudDataTests(unittest.TestCase):
    def test_parser_ignores_pager_text_after_json(self):
        page = parse_file_listing_page(
            '[{"name":"test/embryo.zarr/zarr.json","size":12}]\nNext Page Token = opaque-token\n'
        )

        self.assertEqual(page[0]["name"], "test/embryo.zarr/zarr.json")

    def test_prefixes_require_root_metadata_and_keep_zarr_and_geff(self):
        entries = [
            {"name": "train/a.zarr/0/c/0"},
            {"name": "train/a.zarr/zarr.json"},
            {"name": "train/a.geff/zarr.json"},
            {"name": "test/b.zarr/zarr.json"},
        ]

        self.assertEqual(dataset_prefix("/train/a.zarr/zarr.json"), "train/a.zarr/")
        self.assertEqual(dataset_prefixes(entries, split="train"), ["train/a.geff/", "train/a.zarr/"])
        self.assertEqual(dataset_prefixes(entries, split="test"), ["test/b.zarr/"])


if __name__ == "__main__":
    unittest.main()
