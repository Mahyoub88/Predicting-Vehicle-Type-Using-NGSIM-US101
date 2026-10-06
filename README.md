# Predicting Vehicle Type Using NGSIM US-101

## Implementation at a glance

An implemented machine-learning pipeline applied to real NGSIM US-101 trajectory data, with reproducible notebook code and retained evaluation figures.

| Responsibility | Documented implementation |
|---|---|
| Observation unit | Aggregate trajectories to vehicle level to avoid repeated trajectory rows acting as independent vehicles. |
| Features | Vehicle length and width, mean speed/acceleration, dominant lane and mean space headway. |
| Models | Logistic Regression, Decision Tree and Random Forest. |
| Evaluation | Compare accuracy, precision, recall, F1 and confusion matrices at the documented observation level. |

### Results from the executed source notebook

![Random Forest confusion matrix. Inspect error counts for motorcycles, passenger cars and trucks. Source: published notebook output 11.](docs/overview/notebook-figure-11.png)

*Random Forest confusion matrix. Inspect error counts for motorcycles, passenger cars and trucks. Source: published notebook output 11.*

![Model-score comparison. Read the reported metrics together; the displayed vertical scale starts at 0.85. Source: published notebook output 12.](docs/overview/notebook-figure-12.png)

*Model-score comparison. Read the reported metrics together; the displayed vertical scale starts at 0.85. Source: published notebook output 12.*

![Decision Tree and Random Forest feature importance. These values describe how the fitted models use the available features. Source: published notebook output 13.](docs/overview/notebook-figure-13.png)

*Decision Tree and Random Forest feature importance. These values describe how the fitted models use the available features. Source: published notebook output 13.*

![Vehicle length versus width by class. This view shows physical-feature separation and helps contextualise the classification task. Source: published notebook output 14.](docs/overview/notebook-figure-14.png)

*Vehicle length versus width by class. This view shows physical-feature separation and helps contextualise the classification task. Source: published notebook output 14.*

### Architecture and implementation workflow

![Explanatory functional architecture](docs/overview/architecture.svg)

![Explanatory engineering workflow](docs/overview/workflow.svg)

*Documentation diagrams based on the project scope; original source images and results are captioned separately.*

[Full engineering guide](docs/engineering-guide.md) · [Illustrated case study](https://mahyoub88.github.io/projects/proj-ngsim/)

---



**Project author and sole implementer:** Mohammed Mahyoub.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23033235.svg)](https://doi.org/10.5281/zenodo.23033235)

## Overview

This project applies supervised machine learning techniques to classify vehicle types using the NGSIM US-101 trajectory dataset.

The objective is to compare the performance of three classification algorithms:

- Logistic Regression
- Decision Tree
- Random Forest

Vehicle trajectories are aggregated to vehicle level to prevent data leakage and provide independent observations for model training and evaluation.

---

## Dataset

- Dataset: NGSIM US-101 Vehicle Trajectories
- Source: Federal Highway Administration (FHWA)
- Classification task:
  - Motorcycle
  - Passenger Car
  - Truck

---

## Features

The models use the following traffic and vehicle characteristics:

- Vehicle Length
- Vehicle Width
- Mean Speed
- Mean Acceleration
- Dominant Lane
- Mean Space Headway

---

## Machine Learning Models

- Logistic Regression
- Decision Tree
- Random Forest

Model performance is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

---

## Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

---

## Repository Contents

- Predicting_Vehicle_Type_Using_NGSIM_US101.ipynb

---

## Author

**Mohammed Mahyoub Ali Ayedh Mohammed**

MSc Artificial Intelligence

University of the West of England (UWE Bristol)

---

## License

This repository is intended for educational and academic purposes.

## Illustrated project pages

Project-specific diagrams, source media and implementation context:

- [Vehicle Type Classification Using NGSIM US-101](https://mahyoub88.github.io/projects/proj-ngsim/)

[Browse all engineering case studies](https://mahyoub88.github.io/projects/)

