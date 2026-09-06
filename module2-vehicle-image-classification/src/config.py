"""Project-wide configuration for the VTID2 vehicle-type image study.

Every path, constant and reference value used by the notebooks is declared
here so that the analysis has a single, auditable source of truth.
"""

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = PROJECT_ROOT / "data" / "VTID2"      # class sub-folders of images
CACHE_DIR = PROJECT_ROOT / "data" / "cache"      # cached CNN representations
LABEL_DIR = PROJECT_ROOT / "data" / "labels"     # manually labelled image pairs
FIGURE_DIR = PROJECT_ROOT / "figures"
RESULT_DIR = PROJECT_ROOT / "results"

PAIR_LABEL_FILE = LABEL_DIR / "pair_labels.csv"
SPLIT_MANIFEST = RESULT_DIR / "split_manifest.csv"
AUDIT_SUMMARY = RESULT_DIR / "audit_summary.csv"

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------

RANDOM_SEED = 42

# --------------------------------------------------------------------------
# Classes
#
# The published VTID2 directory names contain the misspelling "Seden".
# The on-disk name is preserved so that the inventory remains a faithful
# record of the source data; the corrected spelling is used for reporting.
# --------------------------------------------------------------------------

CLASS_DISPLAY = {
    "Hatchback": "Hatchback",
    "Other": "Other",
    "Pickup": "Pickup",
    "SUV": "SUV",
    "Seden": "Sedan",
}

CLASS_ORDER = ["Hatchback", "Other", "Pickup", "SUV", "Seden"]

# --------------------------------------------------------------------------
# Representation settings
# --------------------------------------------------------------------------

IMAGE_SIZE = 224                 # ResNet-18 native input resolution
SPATIAL_GRID = 7                 # feature maps are pooled to 7 x 7 locations
FEATURE_LAYERS = ("layer2", "layer3", "layer4")
EXTRACTION_BATCH_SIZE = 32       # sized for a 4 GB Quadro M3000M
PAIR_BATCH_SIZE = 2048           # pair descriptors are computed in blocks of this size

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# --------------------------------------------------------------------------
# Candidate generation (blocking)
#
# Deliberately permissive: this stage may not discard a true near-duplicate,
# because no later stage can recover a pair that blocking has removed.
# --------------------------------------------------------------------------

BLOCKING_COSINE = 0.90

# --------------------------------------------------------------------------
# Fallback decision rule
#
# These are conservative starting values only. They are NOT the calibrated
# operating point: Section 7 of notebook 01 estimates the rule from the
# manually labelled pairs and overrides them. They exist so that the notebook
# remains runnable when the label file has not yet been produced.
# --------------------------------------------------------------------------

FALLBACK_RULE = {
    "l2_agree_090": 0.55,   # fraction of the 49 layer2 locations agreeing >= 0.90
    "l3_mean": 0.88,        # mean layer3 spatial cosine
    "l2_concentration": 0.45,  # share of total difference held by 5 locations
}

# Ambiguous relations are not deleted, only constrained during partitioning.
AMBIGUOUS_RULE = {
    "l2_agree_090": 0.30,
    "l3_mean": 0.80,
}
MUTUAL_KNN_K = 3

# --------------------------------------------------------------------------
# Partitioning
# --------------------------------------------------------------------------

SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}

# --------------------------------------------------------------------------
# Reference run
#
# Values obtained by the original, exploratory version of this analysis.
# They are recorded so that the rewritten pipeline can be reconciled against
# them; they are never used as inputs to any computation.
# --------------------------------------------------------------------------

REFERENCE_RUN = {
    "raw_images": 4793,
    "exact_duplicate_redundant": 7,
    "after_exact_dedup": 4786,
    "high_confidence_duplicate_groups": 299,
    "near_duplicate_redundant": 437,
    "final_images": 4349,
    "train_images": 3036,
    "val_images": 655,
    "test_images": 658,
    "groups_spanning_splits": 0,
}

REFERENCE_CLASS_SPLIT = {
    # class : (train, validation, test)
    "Hatchback": (396, 80, 96),
    "Other": (414, 81, 86),
    "Pickup": (1128, 245, 233),
    "SUV": (308, 79, 72),
    "Seden": (790, 170, 171),
}

# A leakage-risk group larger than this fraction of the dataset cannot be
# accommodated by any group-disjoint partition, and indicates that relations
# have chained rather than that duplication is widespread.
MAX_GROUP_FRACTION = 0.05
