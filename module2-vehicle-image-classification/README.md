# Vehicle Type Recognition from Images (VTID2)

Convolutional classification of five vehicle types, with a documented data audit that
establishes whether the reported accuracy can be believed.

**Author:** Mohammed Mahyoub Ali Ayedh Mohammed — MSc Artificial Intelligence, UWE Bristol

---

## What this project argues

Applying a CNN to a labelled image folder is routine. The part that decides whether
the resulting number means anything is what happens before training: whether the test
partition contains images that share a source with the training partition. Barz and
Denzler (2020) found that 3.3% of the CIFAR-100 test set duplicates training images,
and that removing those duplicates changes the measured ranking of published models.

VTID2 is more exposed to this than CIFAR, because it contains images of the same
vehicle captured moments apart and, in some cases, both an original and an anonymised
copy of the same photograph. Detecting that reliably is the technical content of this
project.

### The design decision the project rests on

An earlier version of this analysis searched for a similarity threshold that would
separate "same photograph" from "different photograph", and the search did not
terminate. Every candidate threshold produced counter-examples in both directions,
each one was investigated by inspecting grids of image pairs, and the notebook grew
past the point of being reviewable without converging on an answer.

The problem was not the threshold. It was that one number was being asked to make two
decisions with opposite cost structures:

| | Deletion | Partition constraint |
|---|---|---|
| Action | Remove an image permanently | Force two images into the same partition |
| Cost of being wrong | Training data destroyed, unrecoverable | A few images sit in one partition rather than another |
| Evidence needed | Near-certainty | Reasonable suspicion |
| Correct target | **Precision** | **Recall** |

No single threshold can be both precision-optimal and recall-optimal. Separating the
two decisions makes each one tractable, and it is why the rewritten analysis fits in
one reviewable notebook.

---

## Method

Four levels, each applied only to what the previous level could not resolve.

| Level | Question | Instrument |
|---|---|---|
| 1. File integrity | Can every file be decoded, and do format or resolution leak the class? | Full decode and metadata |
| 2. Exact duplication | Is this literally the same file? | SHA-256 of the raw bytes |
| 3. Candidate generation | *Could* these two images share a source? | ResNet-18 global embedding, cosine similarity |
| 4. Identity decision | *Do* they share a source? | `layer2` / `layer3` feature-map agreement and difference localisation |

### Why Levels 3 and 4 read the same network at different depths

After global average pooling, the final residual block encodes *what* an image
contains and has discarded all spatial arrangement. Two photographs of two different
silver pickups in similar poses therefore receive nearly identical vectors. That
representation answers "same kind of thing?", which is not the question.

The intermediate blocks keep a 7 x 7 spatial grid, where each location describes the
mid-level structure present at that position. Images that share a source agree
*location by location*; different photographs of similar vehicles do not, because
their content is not registered to the same grid.

Three descriptors are computed per candidate pair, per layer:

- **mean** — average similarity across the 49 locations;
- **agreement** — the fraction of locations agreeing above 0.90;
- **concentration** — the share of the total difference held by the 5 most-different
  locations.

The third is what matches this dataset. An anonymised variant differs from its
original in a small region (a plate, a face) and is identical elsewhere: high
agreement, high concentration. Two different vehicles disagree moderately across many
locations: lower agreement, low concentration. A mean cannot distinguish these,
because both produce the same average — which is why averaging the layers together
was the step at which the earlier analysis lost the signal it needed.

### The rule is estimated, not chosen by eye

The deletion rule is a conjunction of three thresholds, selected against a manually
labelled set of image pairs by a stated objective:

> maximise recall subject to precision = 1.00 on the labelled reference set

The full precision–recall trade-off is reported, not just the selected point; a
depth-2 decision tree under leave-one-out cross-validation checks that the
hand-specified conjunction is not leaving signal unused; and perturbing every
threshold tests that the rule is not fitted to individual labels.

### Deletion is conservative; leakage control is not

Deletion applies only to components in which **every** internal pair was
independently accepted. Similarity is not transitive, so a chain A~B~C is not
evidence that A~C, and components held together by such chains are never deleted
from.

Everything the deletion rule rejected is then reused for the weaker purpose it is
adequate for: leakage-risk grouping. Ambiguous relations pass a mutual
nearest-neighbour filter first, without which a single generic image acts as a hub
and chains large parts of the dataset into one component. The notebook enforces a
hard limit on the largest group and fails loudly if chaining occurs, because the
alternative is a partition that is silently unusable.

---

## Repository layout

```
module2-vehicle-image-classification/
├── notebooks/
│   ├── 01_data_audit_and_leakage_control.ipynb   Levels 1–4, grouping, partitioning
│   └── 02_cnn_modelling.ipynb                    Scratch CNN, transfer learning, evaluation
├── src/
│   ├── config.py        Every path, constant and reference value
│   └── vtid_audit.py    Inventory, hashing, graphs, partitioning, verification
├── data/
│   ├── VTID2/           The dataset (not committed) — one folder per class
│   ├── labels/          pair_labels.csv, the manual reference set
│   └── cache/           Cached CNN representations (not committed)
├── figures/             Evidence, written as files rather than embedded
└── results/             Manifests and summary tables
```

Mechanical operations live in `src/`; the notebooks carry only the reasoning specific
to the study. This is also what keeps them small enough to review — the earlier
version's size came from base64-encoded image grids stored inline.

---

## Reproducing

```bash
pip install -r requirements.txt
# place the dataset at data/VTID2/<ClassName>/*.jpg
jupyter lab notebooks/01_data_audit_and_leakage_control.ipynb
```

Run `01` first: it writes `results/split_manifest.csv`, which `02` consumes. On the
first run, `01` also writes a stratified labelling template and its contact sheets;
complete it as `data/labels/pair_labels.csv` and re-run so the deletion rule is
calibrated rather than falling back to the uncalibrated defaults in `config.py`.

Sampling for that template is stratified across the descriptor range rather than
sequential, which concentrates labelling effort near the decision boundary. Roughly
eighty labelled pairs constrain the rule better than several hundred labelled in the
order they happen to appear.

---

## Verification

The partition is re-checked from the final assignment, without reusing the objects
that produced it:

| Check | Required |
|---|---|
| Identical files (SHA-256) shared between partitions | 0 |
| Accepted duplicate pairs crossing partitions | 0 |
| Leakage-risk relations crossing partitions | 0 |
| Leakage-risk groups spanning more than one partition | 0 |

Alongside these, the maximum cosine similarity still crossing each partition boundary
is reported. No audit can prove every near-duplicate was found, and one number
bounding the worst residual case is a more honest claim than silence.

`02` then measures what the audit was worth, by training the identical architecture on
a random split of the same images that ignores the groups. The difference between the
two test accuracies is the amount an unaudited evaluation of this dataset would have
overstated.

---

## Status

The notebooks have been executed end to end against a synthetic fixture with planted
duplicates of known type, which verifies the code paths, the grouping logic and the
partition guarantees. **The final figures must be produced by running `01` and `02`
once on the real VTID2 dataset**; the reference values in `config.py` come from the
earlier exploratory analysis and are used only for reconciliation, never as inputs.

Before submission, replace the dataset citation placeholder in both notebooks and in
the references below with the full citation for the copy of VTID2 you obtained.

---

## Key references

Barz, B. and Denzler, J. (2020) 'Do we train on test data? Purging CIFAR of
near-duplicates', *Journal of Imaging*, 6(6), 41.

Christen, P. (2012) *Data Matching: Concepts and Techniques for Record Linkage,
Entity Resolution, and Duplicate Detection*. Berlin: Springer.

He, K., Zhang, X., Ren, S. and Sun, J. (2016) 'Deep residual learning for image
recognition', *IEEE Conference on Computer Vision and Pattern Recognition*,
pp. 770-778.

Kaufman, S., Rosset, S., Perlich, C. and Stitelman, O. (2012) 'Leakage in data
mining: formulation, detection, and avoidance', *ACM Transactions on Knowledge
Discovery from Data*, 6(4), pp. 1-21.

Zeiler, M.D. and Fergus, R. (2014) 'Visualizing and understanding convolutional
networks', *European Conference on Computer Vision*, pp. 818-833.

Full reference lists are given in each notebook.

---

## License

Educational and academic use.
