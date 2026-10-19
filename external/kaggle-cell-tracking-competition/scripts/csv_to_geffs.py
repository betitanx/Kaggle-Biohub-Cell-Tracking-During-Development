from __future__ import annotations

import argparse
import re
from pathlib import Path

import polars as pl
import tracksdata as td

from tracking_cellmot.io import save_graph


def _reject_symlinks(path: Path) -> None:
    # Inspect lexical ancestors before resolving: resolve() would hide links,
    # including an ancestor in a path such as "linked/../outputs".
    absolute = path.absolute()
    for component in (*reversed(absolute.parents), absolute):
        if component.is_symlink():
            raise ValueError(f"Symlink output paths are not allowed: {component}")


def _check_output_path(path: Path, overwrite: bool) -> None:
    _reject_symlinks(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists (use --overwrite to replace it): {path}")


def build_graph_from_rows(
    node_rows: pl.DataFrame,
    edge_rows: pl.DataFrame,
) -> td.graph.InMemoryGraph:
    """Rebuild a tracksdata graph from one dataset's node and edge rows.

    tracksdata assigns fresh node ids, so CSV ``node_id``s are remapped when
    adding edges (matching is spatial, not by id).
    """
    graph = td.graph.InMemoryGraph()
    for key in ("z", "y", "x"):
        graph.add_node_attr_key(key, pl.Float64, -999999.0)

    assigned = graph.bulk_add_nodes(
        node_rows.select(
            pl.col("t").cast(pl.Int64),
            pl.col("z").cast(pl.Float64),
            pl.col("y").cast(pl.Float64),
            pl.col("x").cast(pl.Float64),
        ).to_dicts()
    )
    id_map = dict(zip(node_rows["node_id"].to_list(), assigned, strict=True))

    if edge_rows.height:
        graph.bulk_add_edges(
            [
                {"source_id": id_map[s], "target_id": id_map[t]}
                for s, t in zip(
                    edge_rows["source_id"].to_list(), edge_rows["target_id"].to_list(), strict=True
                )
            ]
        )

    return graph


def csv_to_geffs(
    csv_path: Path | str,
    out_dir: Path | str,
    overwrite: bool = False,
) -> list[Path]:
    """Convert a CSV to GEFFs after preflighting every dataset and output path.

    Existing outputs require explicit overwrite; symlinks are never allowed.
    The output tree must not be concurrently modified by untrusted processes:
    path-based checks cannot make the downstream writer race-free.
    """
    csv_path = Path(csv_path)
    out_dir = Path(out_dir)

    df = pl.read_csv(
        csv_path,
        columns=["dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"],
        schema_overrides={"dataset": pl.String},
    )

    _reject_symlinks(out_dir)
    if out_dir.exists() and not out_dir.is_dir():
        raise NotADirectoryError(f"Output directory is not a directory: {out_dir}")
    output_root = out_dir.resolve()
    outputs = []
    for (name,), group in df.group_by("dataset"):
        if not isinstance(name, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name) is None:
            raise ValueError(f"Invalid dataset name: {name!r}")
        out_path = out_dir / f"{name}.geff"
        _check_output_path(out_path, overwrite)
        if out_path.resolve().parent != output_root:
            raise ValueError(f"Output path escapes output directory: {out_path}")
        outputs.append((name, group, out_path))

    # Do not create outputs or invoke the destructive writer until all groups
    # have passed validation, regardless of Polars' group iteration order.
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, group, out_path in outputs:
        node_rows = group.filter(pl.col("row_type") == "node")
        edge_rows = group.filter(pl.col("row_type") == "edge")

        graph = build_graph_from_rows(node_rows, edge_rows)
        _check_output_path(out_path, overwrite)
        save_graph(graph, out_path, overwrite=overwrite)
        written.append(out_path)
        print(f"{name}: {node_rows.height} nodes, {edge_rows.height} edges -> {out_path}")

    print(f"\nWrote {len(written)} geffs to {out_dir}")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a submission CSV into one .geff per dataset.")
    parser.add_argument("--csv", type=Path, required=True, help="Submission CSV path.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for .geff files.")
    overwrite_flags = parser.add_mutually_exclusive_group()
    overwrite_flags.add_argument(
        "--overwrite", action="store_true", help="Replace existing non-symlink geffs."
    )
    overwrite_flags.add_argument(
        "--no-overwrite", dest="overwrite", action="store_false",
        help="Do not overwrite existing geffs (default).",
    )
    parser.set_defaults(overwrite=False)
    args = parser.parse_args()

    csv_to_geffs(args.csv, args.out_dir, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
