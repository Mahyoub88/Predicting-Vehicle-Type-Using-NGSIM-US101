# Vehicle Type Classification Using NGSIM US-101 — Engineering Guide

Developed and evaluated supervised machine-learning models for classifying motorcycles, passenger cars, and trucks using the NGSIM US-101 traffic trajectory dataset. Models compared: Logistic Regression, Decision Tree, and Random Forest.

## Visual overview

![Functional overview](overview/architecture.svg)

*New explanatory diagram; grouped responsibilities, not an as-built schematic or test result.*

![Engineering workflow](overview/workflow.svg)

*New explanatory workflow; a documentation aid, not evidence that every proposed check was performed.*

## Why vehicle-level aggregation matters

A vehicle appears in many trajectory frames. Randomly splitting those frames can put the same vehicle into both training and testing. The notebook aggregates records into vehicle observations before modelling so the evaluation unit matches the classification task.

## Features and model choices

Vehicle length and width capture physical dimensions; mean speed, mean acceleration, dominant lane and mean space headway describe traffic behaviour. Logistic Regression, Decision Tree and Random Forest provide contrasting decision boundaries and interpretability.

## Evaluation beyond headline accuracy

Motorcycles, passenger cars and trucks form an imbalanced classification problem. Macro-F1 and classwise recall are needed alongside accuracy. The dimensions-only versus full-feature comparison asks how much behaviour adds beyond vehicle size. Numerical results belong to the notebook outputs and should retain their experimental context.

## Reproducibility

The repository contains the Jupyter notebook and requirements.txt. Run it with the original dataset paths reviewed first; preserve vehicle identity boundaries during train/test preparation and tuning. The Zenodo DOI identifies the archived research output, while GitHub contains the current documentation.

## Evidence to review or collect

The following are suggested review checks. A checklist entry is not a claimed pass result.

- Vehicle identities separated across splits.
- Feature definitions and missing-value handling.
- Training-only model selection.
- Classwise confusion matrix and ablation results.

## Source gallery

![Vehicle length — distribution and boxplot. Explore the spread and extreme values before fitting a classifier. Source: published notebook output 1.](overview/notebook-figure-01.png)

*Vehicle length — distribution and boxplot. Explore the spread and extreme values before fitting a classifier. Source: published notebook output 1..*

![Vehicle width — distribution and boxplot. Inspect this physical feature alongside length when interpreting class separation. Source: published notebook output 2.](overview/notebook-figure-02.png)

*Vehicle width — distribution and boxplot. Inspect this physical feature alongside length when interpreting class separation. Source: published notebook output 2..*

![Vehicle speed — distribution and boxplot. Summarised vehicle-level speeds describe the traffic conditions represented in the dataset. Source: published notebook output 3.](overview/notebook-figure-03.png)

*Vehicle speed — distribution and boxplot. Summarised vehicle-level speeds describe the traffic conditions represented in the dataset. Source: published notebook output 3..*

![Vehicle acceleration — distribution and boxplot. Review the range of aggregated acceleration values and their outliers. Source: published notebook output 4.](overview/notebook-figure-04.png)

*Vehicle acceleration — distribution and boxplot. Review the range of aggregated acceleration values and their outliers. Source: published notebook output 4..*

![Space headway — distribution and boxplot. Examine spacing between vehicles as a traffic-context variable. Source: published notebook output 5.](overview/notebook-figure-05.png)

*Space headway — distribution and boxplot. Examine spacing between vehicles as a traffic-context variable. Source: published notebook output 5..*

![Vehicle-class distribution. Counts and proportions reveal class imbalance, which matters when interpreting overall accuracy. Source: published notebook output 6.](overview/notebook-figure-06.png)

*Vehicle-class distribution. Counts and proportions reveal class imbalance, which matters when interpreting overall accuracy. Source: published notebook output 6..*

![Vehicle-level correlation matrix. Compare relationships between predictors; correlation alone does not establish causation. Source: published notebook output 7.](overview/notebook-figure-07.png)

*Vehicle-level correlation matrix. Compare relationships between predictors; correlation alone does not establish causation. Source: published notebook output 7..*

![Decision tree — top three levels. The figure explains early decision paths; the trained model extends beyond the depth shown. Source: published notebook output 8.](overview/notebook-figure-08.png)

*Decision tree — top three levels. The figure explains early decision paths; the trained model extends beyond the depth shown. Source: published notebook output 8..*

![Logistic Regression confusion matrix. Read the classwise errors rather than relying on a single aggregate score. Source: published notebook output 9.](overview/notebook-figure-09.png)

*Logistic Regression confusion matrix. Read the classwise errors rather than relying on a single aggregate score. Source: published notebook output 9..*

![Decision Tree confusion matrix. Compare which classes are confused against the other classifiers on the same evaluation split. Source: published notebook output 10.](overview/notebook-figure-10.png)

*Decision Tree confusion matrix. Compare which classes are confused against the other classifiers on the same evaluation split. Source: published notebook output 10..*

![Random Forest confusion matrix. Inspect error counts for motorcycles, passenger cars and trucks. Source: published notebook output 11.](overview/notebook-figure-11.png)

*Random Forest confusion matrix. Inspect error counts for motorcycles, passenger cars and trucks. Source: published notebook output 11..*

![Model-score comparison. Read the reported metrics together; the displayed vertical scale starts at 0.85. Source: published notebook output 12.](overview/notebook-figure-12.png)

*Model-score comparison. Read the reported metrics together; the displayed vertical scale starts at 0.85. Source: published notebook output 12..*

![Decision Tree and Random Forest feature importance. These values describe how the fitted models use the available features. Source: published notebook output 13.](overview/notebook-figure-13.png)

*Decision Tree and Random Forest feature importance. These values describe how the fitted models use the available features. Source: published notebook output 13..*

![Vehicle length versus width by class. This view shows physical-feature separation and helps contextualise the classification task. Source: published notebook output 14.](overview/notebook-figure-14.png)

*Vehicle length versus width by class. This view shows physical-feature separation and helps contextualise the classification task. Source: published notebook output 14..*


## Sources and provenance

- [Published portfolio description](https://mahyoub88.github.io/projects/proj-ngsim/).
- [Project README](../README.md) and existing repository files.
- [LinkedIn projects](https://www.linkedin.com/in/mohammed-mahyoub/details/projects/): supplementary descriptions and project media.
- New SVG figures and explanatory text were authored for this documentation update; they are not original photographs or new measured results.
