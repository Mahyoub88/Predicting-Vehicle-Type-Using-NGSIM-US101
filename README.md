# Predicting Vehicle Type Using NGSIM US-101

## Illustrated engineering guide

[Read the full engineering guide](docs/engineering-guide.md) for architecture, workflow, design rationale, evidence notes and the source gallery.

![Engineering overview](docs/overview/architecture.svg)

*Explanatory diagram added for this write-up.*

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
