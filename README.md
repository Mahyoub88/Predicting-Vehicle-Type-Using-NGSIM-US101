# Vehicle Type Classification Using NGSIM US-101

[Read case study](https://mahyoub88.github.io/projects/proj-ngsim/) · [Project index](docs/PROJECTS.md) · [Engineering guide](docs/engineering-guide.md)

Archived publication title: **Predicting Vehicle Type Using NGSIM US-101**. The [Zenodo record](https://doi.org/10.5281/zenodo.23033235) retains that citation title.


**Project author and sole implementer:** Mohammed Mahyoub.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23033235.svg)](https://doi.org/10.5281/zenodo.23033235)

## Executed notebook preview

![Output from the executed public source notebook](docs/overview/notebook-figure-12.png)

The complete source notebook and case-study gallery provide the evaluation context and remaining figures.

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
