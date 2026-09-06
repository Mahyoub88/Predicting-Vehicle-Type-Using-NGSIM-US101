"""Reusable plumbing for the VTID2 data audit and leakage-controlled split.

The functions collected here are deliberately kept out of the analysis
notebook. They perform file-system inventory, cryptographic hashing, graph
construction and partitioning: mechanical steps whose correctness is easier to
argue in a tested module than in narrative notebook cells. The notebook then
carries only the convolutional-feature reasoning that the study is about.
"""

from __future__ import annotations

import hashlib
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFile

# Truncated files must raise rather than be silently padded, so that the
# integrity check in Section 3 detects them.
ImageFile.LOAD_TRUNCATED_IMAGES = False

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ==========================================================================
# 1. Reproducibility
# ==========================================================================

def set_seed(seed: int) -> None:
    """Seed every random source used by the audit."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# ==========================================================================
# 2. Inventory and integrity
# ==========================================================================

def sha256_of_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return the SHA-256 digest of the file's raw bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_inventory(data_root: Path) -> pd.DataFrame:
    """Walk the class sub-folders and record one row per image file.

    Every file is opened and fully decoded, so unreadable and truncated
    images are detected here rather than during training.
    """
    records = []

    for class_dir in sorted(p for p in Path(data_root).iterdir() if p.is_dir()):
        for path in sorted(class_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            record = {
                "image_id": f"{class_dir.name}/{path.name}",
                "path": str(path),
                "class_name": class_dir.name,
                "filename": path.name,
                "extension": path.suffix.lower(),
                "bytes": path.stat().st_size,
            }

            try:
                with Image.open(path) as image:
                    image.load()                 # forces full decode
                    record["width"], record["height"] = image.size
                    record["mode"] = image.mode
                    record["readable"] = True
            except Exception as error:           # noqa: BLE001 - reported, not raised
                record.update(
                    width=np.nan,
                    height=np.nan,
                    mode=None,
                    readable=False,
                    error=type(error).__name__,
                )

            record["sha256"] = sha256_of_file(path)
            records.append(record)

    inventory = pd.DataFrame(records)
    inventory["aspect_ratio"] = inventory["width"] / inventory["height"]
    inventory["n_channels"] = inventory["mode"].map(
        {"L": 1, "RGB": 3, "RGBA": 4, "P": 1, "CMYK": 4}
    )
    return inventory.reset_index(drop=True)


def integrity_report(inventory: pd.DataFrame) -> pd.DataFrame:
    """Summarise the structural properties that could bias a CNN."""
    readable = inventory[inventory["readable"]]
    rows = [
        ("Image files found", len(inventory)),
        ("Unreadable or truncated", int((~inventory["readable"]).sum())),
        ("Distinct classes", inventory["class_name"].nunique()),
        ("Distinct file extensions", inventory["extension"].nunique()),
        ("Non-RGB colour modes", int((readable["mode"] != "RGB").sum())),
        ("Distinct resolutions", readable.groupby(["width", "height"]).ngroups),
        ("Minimum width", int(readable["width"].min())),
        ("Maximum width", int(readable["width"].max())),
        ("Minimum height", int(readable["height"].min())),
        ("Maximum height", int(readable["height"].max())),
    ]
    return pd.DataFrame(rows, columns=["Property", "Value"])


# ==========================================================================
# 3. Exact duplicates
# ==========================================================================

def exact_duplicate_groups(inventory: pd.DataFrame) -> pd.DataFrame:
    """Group images that are byte-for-byte identical.

    SHA-256 answers only one question: is this the same file? Two visually
    identical images re-encoded at different JPEG qualities receive different
    digests, which is why this stage is a floor and not the whole analysis.
    """
    counts = inventory.groupby("sha256").size()
    duplicated = counts[counts > 1].index
    groups = (
        inventory[inventory["sha256"].isin(duplicated)]
        .sort_values(["sha256", "image_id"])
        .assign(group_size=lambda d: d.groupby("sha256")["sha256"].transform("size"))
    )
    return groups.reset_index(drop=True)


def choose_representative(frame: pd.DataFrame) -> str:
    """Pick the copy to retain from a set of duplicates.

    The highest-resolution copy is kept because it carries the most detail for
    a convolutional model; the filename breaks ties so the choice is
    deterministic and independent of directory-listing order.
    """
    ordered = frame.assign(pixels=frame["width"] * frame["height"]).sort_values(
        ["pixels", "image_id"], ascending=[False, True]
    )
    return str(ordered.iloc[0]["image_id"])


# ==========================================================================
# 4. Graphs over image pairs
# ==========================================================================

class UnionFind:
    """Disjoint-set forest used to turn accepted pairs into groups."""

    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, a: int, b: int) -> None:
        root_a, root_b = self.find(a), self.find(b)
        if root_a == root_b:
            return
        if self.rank[root_a] < self.rank[root_b]:
            root_a, root_b = root_b, root_a
        self.parent[root_b] = root_a
        if self.rank[root_a] == self.rank[root_b]:
            self.rank[root_a] += 1


def connected_components(n_nodes: int, edges: np.ndarray) -> np.ndarray:
    """Label each node with the identifier of its connected component."""
    forest = UnionFind(n_nodes)
    for a, b in edges:
        forest.union(int(a), int(b))
    roots = np.array([forest.find(i) for i in range(n_nodes)])
    _, labels = np.unique(roots, return_inverse=True)
    return labels


def split_components_by_consistency(
    labels: np.ndarray, edges: np.ndarray
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Separate fully pairwise-consistent components from ambiguous ones.

    A component is accepted as a duplicate group only when every one of its
    member pairs was independently accepted, i.e. the component is a complete
    graph. Components held together by a chain of individually accepted pairs
    are returned separately: transitivity is not guaranteed by a similarity
    threshold, so those components are not safe to delete from.
    """
    edge_set = {(int(min(a, b)), int(max(a, b))) for a, b in edges}
    members = defaultdict(list)
    for node, label in enumerate(labels):
        members[int(label)].append(node)

    consistent, ambiguous = [], []
    for nodes in members.values():
        if len(nodes) < 2:
            continue
        nodes_array = np.array(sorted(nodes))
        expected = len(nodes_array) * (len(nodes_array) - 1) // 2
        present = sum(
            (int(min(a, b)), int(max(a, b))) in edge_set
            for i, a in enumerate(nodes_array)
            for b in nodes_array[i + 1 :]
        )
        (consistent if present == expected else ambiguous).append(nodes_array)

    return consistent, ambiguous


def mutual_knn_edges(edges: np.ndarray, scores: np.ndarray, k: int) -> np.ndarray:
    """Keep an edge only when both endpoints rank the other in their top-k.

    Single-linkage grouping is vulnerable to chaining: one image that is
    moderately similar to many others merges unrelated components into a
    single giant group, which would leave too little material to build a
    group-disjoint partition. Requiring mutual rank membership removes those
    hub edges while retaining genuinely reciprocal relations.
    """
    if len(edges) == 0:
        return edges

    ranked = defaultdict(list)
    for index, (a, b) in enumerate(edges):
        ranked[int(a)].append((scores[index], int(b)))
        ranked[int(b)].append((scores[index], int(a)))

    top_k = {
        node: {neighbour for _, neighbour in sorted(pairs, reverse=True)[:k]}
        for node, pairs in ranked.items()
    }

    keep = [
        index
        for index, (a, b) in enumerate(edges)
        if int(b) in top_k[int(a)] and int(a) in top_k[int(b)]
    ]
    return edges[np.array(keep, dtype=int)] if keep else edges[:0]


# ==========================================================================
# 5. Group-stratified partitioning
# ==========================================================================

def group_stratified_split(
    frame: pd.DataFrame,
    ratios: dict[str, float],
    seed: int,
    group_column: str = "group_id",
    class_column: str = "class_name",
) -> pd.Series:
    """Assign whole groups to partitions while preserving class proportions.

    Groups are atomic: every image that might share a source with another is
    placed in the same partition, which is what makes the evaluation honest.
    Groups are visited largest first and assigned to whichever partition is
    furthest below its quota for the group's dominant class, so the discrete
    group sizes distort the target proportions as little as possible.
    """
    group_members = frame.groupby(group_column).indices
    group_classes = {
        group: Counter(frame.iloc[idx][class_column]) for group, idx in group_members.items()
    }

    class_totals = frame[class_column].value_counts().to_dict()
    quotas = {
        split: {cls: total * ratio for cls, total in class_totals.items()}
        for split, ratio in ratios.items()
    }
    assigned = {split: defaultdict(float) for split in ratios}

    # Equally sized groups are ordered by a seeded shuffle rather than by
    # directory-listing order, then the stable sort places the largest groups
    # first. Large groups are the ones that can overshoot a quota, so they are
    # placed while the most room remains.
    order = list(group_members)
    random.Random(seed).shuffle(order)
    order.sort(key=lambda g: -len(group_members[g]))

    assignment = pd.Series(index=frame.index, dtype=object)

    for group in order:
        counts = group_classes[group]
        dominant = max(counts.items(), key=lambda item: (item[1], item[0]))[0]

        deficits = {
            split: quotas[split][dominant] - assigned[split][dominant] for split in ratios
        }
        target = max(deficits.items(), key=lambda item: item[1])[0]

        for cls, count in counts.items():
            assigned[target][cls] += count
        assignment.iloc[group_members[group]] = target

    return assignment


def split_composition(frame: pd.DataFrame, display: dict[str, str]) -> pd.DataFrame:
    """Cross-tabulate class counts against partitions for the report."""
    table = pd.crosstab(frame["class_name"], frame["split"])
    for column in ("train", "val", "test"):
        if column not in table:
            table[column] = 0
    table = table[["train", "val", "test"]]
    table.index = [display.get(name, name) for name in table.index]
    table["Total"] = table.sum(axis=1)
    table.loc["Total"] = table.sum(axis=0)
    return table


# ==========================================================================
# 6. Verification
# ==========================================================================

def verify_partition(
    frame: pd.DataFrame,
    accepted_edges: np.ndarray,
    risk_edges: np.ndarray,
) -> pd.DataFrame:
    """Independently re-check the partition for every known leakage channel.

    The checks do not reuse the objects that produced the partition; they are
    recomputed from the final assignment so that a bug in the pipeline would
    show up here rather than be reproduced.
    """
    split_of = frame["split"].to_numpy()
    hash_of = frame["sha256"].to_numpy()
    group_of = frame["group_id"].to_numpy()

    def crossing(edges: np.ndarray) -> int:
        if len(edges) == 0:
            return 0
        return int((split_of[edges[:, 0]] != split_of[edges[:, 1]]).sum())

    hash_splits = pd.DataFrame({"sha256": hash_of, "split": split_of})
    shared_hashes = (
        hash_splits.groupby("sha256")["split"].nunique().gt(1).sum()
    )

    group_splits = pd.DataFrame({"group_id": group_of, "split": split_of})
    spanning_groups = group_splits.groupby("group_id")["split"].nunique().gt(1).sum()

    rows = [
        ("Identical files (SHA-256) shared between partitions", int(shared_hashes)),
        ("Accepted duplicate pairs crossing partitions", crossing(accepted_edges)),
        ("Leakage-risk relations crossing partitions", crossing(risk_edges)),
        ("Leakage-risk groups spanning more than one partition", int(spanning_groups)),
    ]
    return pd.DataFrame(rows, columns=["Verification check", "Count (must be 0)"])


def max_cross_split_similarity(
    embeddings: np.ndarray, splits: np.ndarray, batch_size: int = 512
) -> pd.DataFrame:
    """Report the single most similar image pair across each partition boundary.

    A partition can satisfy every discrete check above and still be optimistic
    if some near-duplicate escaped detection entirely. Reporting the worst
    remaining cross-partition similarity states that residual risk explicitly
    instead of leaving it unquantified.
    """
    names = ["train", "val", "test"]
    worst = {}

    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            left = embeddings[splits == first]
            right = embeddings[splits == second]
            if len(left) == 0 or len(right) == 0:
                worst[f"{first} vs {second}"] = np.nan
                continue
            best = -1.0
            for start in range(0, len(left), batch_size):
                block = left[start : start + batch_size] @ right.T
                best = max(best, float(block.max()))
            worst[f"{first} vs {second}"] = best

    return pd.DataFrame(
        {"Partition boundary": list(worst), "Maximum cosine similarity": list(worst.values())}
    )


# ==========================================================================
# 7. Reporting helpers
# ==========================================================================

def save_figure(figure, name: str, figure_dir: Path, dpi: int = 120) -> Path:
    """Write a figure to disk instead of embedding it in the notebook.

    Storing evidence as referenced files rather than base64 payloads is what
    keeps the notebook small enough to review, version and submit.
    """
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    destination = figure_dir / f"{name}.png"
    figure.savefig(destination, dpi=dpi, bbox_inches="tight")
    return destination


def contact_sheet(pairs, inventory, scores, title, max_pairs=8, thumb=160):
    """Render a compact strip of image pairs as a single figure.

    One reviewed figure of the hardest cases is stronger evidence than many
    grids of easy ones, and costs a fraction of the notebook size.
    """
    import matplotlib.pyplot as plt

    pairs = list(pairs)[:max_pairs]
    figure, axes = plt.subplots(2, len(pairs), figsize=(1.6 * len(pairs), 3.6))
    axes = np.atleast_2d(axes)

    for column, (a, b) in enumerate(pairs):
        for row, index in enumerate((a, b)):
            path = inventory.iloc[index]["path"]
            with Image.open(path) as image:
                axes[row, column].imshow(image.convert("RGB").resize((thumb, thumb)))
            axes[row, column].axis("off")
        axes[0, column].set_title(f"{scores[column]:.3f}", fontsize=8)

    figure.suptitle(title, fontsize=10)
    figure.tight_layout()
    return figure
