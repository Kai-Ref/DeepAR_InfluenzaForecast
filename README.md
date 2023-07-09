# DeepAR Influenza Forecast
This repository contains code and resources for training and evaluating a DeepAR model for probabilistic influenza forecasting.

## Table of Contents
- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Modules and Code Segments](#modules-and-code-segments)

## Overview <a name="overview"></a>

The DeepAR Influenza Forecast project aims to produce probabilistic forecasts for influenza activity in German districts based on historical data and covariates. The project uses the DeepAR algorithm, which is a popular deep learning model for time series forecasting introduced by <a href = "https://www.sciencedirect.com/science/article/pii/S0169207019301888" target = "_self">Salinas et al. (2020)</a>. In contrast to the Amazon Sagemaker implementation, we incorporate the DeepAR model through the <a href = "https://ts.gluon.ai/stable/index.html" target = "_self">GluonTS</a> library. Lastly, we compare the DeepAR model to the Simple Feedforward Neural Network from GluonTS as well as to the hhh4 model from the <a href = "https://www.jstatsoft.org/article/view/v070i10" target = "_self">surveillance</a> R-package.

## Repository Structure <a name="repository-structure"></a>

The repository is organized as follows:

- [`HyperparameterFiles/`](#hyperparameter-files): Contains the results of hyperparameter runs. 
- [`Notebooks/`](#notebooks): Contains all jupyter notebooks.
  - [`DataProcessing/`](#data-processing):
  - [`FigureCreation/`](#figure-creation):
  - [`Modeltuning/`](#modeltuning):
  - [`OldNotebooks/`](#old-notebooks):
- [`PythonFiles/`](#python-files): Python files that define important functionalities and are accessed continuosly by notebooks.
- [`R/`](#R): R files of to train and produce forecasts of the hhh4 model.


### HyperparameterFiles <a name="hyperparameter-files"></a>
### Notebooks <a name="notebooks"></a>
### DataProcessing <a name="data-processing"></a>
### FigureCreation <a name="figure-creation"></a>
### Modeltuning <a name="modeltuning"></a>
### OldNotebooks <a name="old-notebooks"></a>
### PythonFiles <a name="python-files"></a>
### R <a name="R"></a>

## Modules and Code Segments

The repository includes the following important modules and code segments:

- `data_preprocessing.ipynb`: This notebook demonstrates how to preprocess the influenza dataset, including handling missing values, scaling the data, and splitting it into training and validation sets.

- `model_training.ipynb`: This notebook showcases the process of training a DeepAR model on the preprocessed dataset using Amazon SageMaker. It covers configuring the model, specifying hyperparameters, and launching the training job.

- `model_evaluation.ipynb`: This notebook provides an evaluation of the trained DeepAR model. It includes loading the trained model, making predictions on the test set, and evaluating the model's performance using various metrics.

