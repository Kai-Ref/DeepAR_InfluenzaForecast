# DeepAR Influenza Forecast
This repository contains code and resources for training and evaluating a DeepAR model for probabilistic influenza forecasting.

## Table of Contents
- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Important Modules and Code Segments](#important-modules-and-code-segments)

## Overview <a name="overview"></a>

The DeepAR Influenza Forecast project aims to produce probabilistic forecasts for influenza activity in German districts based on historical data and covariates. The project uses the DeepAR algorithm, which is a popular deep learning model for time series forecasting introduced by <a href = "https://www.sciencedirect.com/science/article/pii/S0169207019301888" target = "_self">Salinas et al. (2020)</a>. In contrast to the Amazon Sagemaker implementation, we incorporate the DeepAR model through the <a href = "https://ts.gluon.ai/stable/index.html" target = "_self">GluonTS</a> library. Lastly, we compare the DeepAR model to the Simple Feedforward Neural Network from GluonTS as well as to the hhh4 model from the <a href = "https://www.jstatsoft.org/article/view/v070i10" target = "_self">surveillance</a> R-package.

## Repository Structure <a name="repository-structure"></a>

This repository is made up of the following directories, which contain:

- <b>HyperparameterFiles</b>: the result-files of hyperparameter runs. 
- <b>Notebooks</b>: all jupyter notebooks and in particular notebooks that:
  - [`DataProcessing/`](#data-processing): showcase and include the data.
  - [`EarlyNotebooks/`](#early-notebooks): display the general workflow and early stage developments.
  - [`FigureCreation/`](#figure-creation): are used to create figures and visualizations for the thesis/model evaluation.
  - [`FurtherResearch/`](#further-research): are outside of the scope of the thesis.
  - [`Modeltuning/`](#modeltuning): were used to produce and evaluate the hyperparameter training.
  - [`OldNotebooks/`](#old-notebooks):  are outdated and don't showcase relevant information.
- [`PythonFiles/`](#python-files): Python files that define important functionalities, which are accessed continuosly by notebooks.
- [`R/`](#r): R files, used to implement the hhh4 model from the surveillance package. 

## Important Modules and Code Segments<a name ="important-modules-and-code-segments">

A good first start to grasp our implementation is with the 
<a href = "https://github.com/Kai-Ref/DeepAR_InfluenzaForecast/blob/main/Notebooks/Early%20Notebooks/StepByStepGuide.ipynb" target = "_self">StepByStepGuide.ipynb</a> notebook. However, our final implementation looks a bit different from this. 

### DataProcessing <a name="data-processing"></a>
### EarlyNotebooks <a name="early-notebooks"></a>
### FigureCreation <a name="figure-creation"></a>
### FurtherResearch <a name="further-research"></a>
### Modeltuning <a name="modeltuning"></a>
### OldNotebooks <a name="old-notebooks"></a>
### PythonFiles <a name="python-files"></a>
### R <a name="r"></a>

