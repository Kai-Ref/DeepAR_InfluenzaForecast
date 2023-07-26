import ray
from ray import tune, air
from ray.air import session
from ray.tune import ResultGrid
from PythonFiles.model import model, preprocessing, split_forecasts_by_week, forecast_by_week, train_test_split
from PythonFiles.Configuration import Configuration
import pandas as pd
import numpy as np
from datetime import datetime
from gluonts.mx import Trainer, DeepAREstimator
from gluonts.mx.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.evaluation import make_evaluation_predictions, Evaluator
import os
import itertools


def get_data(truncate=False, with_features=True, config=None):
    """
    Load and process the influenza dataset along with additional population and neighborhood data.
    Split the data into training and testing sets with optional features.

    Parameters:
        truncate (bool): If True, truncate the data to start from 2010; otherwise, start from 1999.
        with_features (bool): If True, include additional features (population and neighborhood data) in the dataset.
        config (Configuration): Configuration object with settings for the forecasting process.

    Returns:
        dict: A dictionary containing data splits as per the configuration and the processed DataFrame.

    """
    # Change the working directory to the desired location
    os.chdir('/home/reffert/DeepAR_InfluenzaForecast')
    
    # If the configuration is not provided, create a new Configuration object
    if config is None:
        config = Configuration()
    
    # Import the data: influenza data, population data, and neighborhood data
    influenza_df = pd.read_csv("/home/reffert/DeepAR_InfluenzaForecast/Notebooks/DataProcessing/influenza.csv", sep=',')
    population_df = pd.read_csv("/home/reffert/DeepAR_InfluenzaForecast/Notebooks/DataProcessing/PopulationVector.csv", sep=',')
    neighbourhood_df = pd.read_csv("/home/reffert/DeepAR_InfluenzaForecast/Notebooks/DataProcessing/AdjacentMatrix.csv", sep=',', index_col=0)
    
    data_splits_dict = {}
    locations = list(influenza_df.location.unique())
    
    # Process the influenza DataFrame into a uniformly spaced DataFrame
    df = influenza_df.loc[influenza_df.location.isin(locations), ['value', 'location', 'date', 'week']]
    df = preprocessing(config, df, check_count=False, output_type="corrected_df")
    
    # Add population and neighborhood data to the DataFrame if with_features is True
    if with_features:
        for location in locations:
            df.loc[df.location == location, "population"] = int(population_df.loc[population_df.Location == location, "2011"].values[0])
            df.loc[df.location == location, locations] = neighbourhood_df.loc[neighbourhood_df.index == location, locations].values[0].astype(int)
    
    # Set the correct date to truncate or not truncate the available data
    if truncate:
        config.train_start_time = datetime(2010, 1, 1, 0, 0, 0)
        year = "2010"
    else:
        config.train_start_time = datetime(1999, 1, 1, 0, 0, 0)
        year = "2001"
    
    config.train_end_time = datetime(2016, 9, 30, 23, 0, 0)
    config.test_end_time = datetime(2018, 9, 30, 23, 0, 0)
    
    # Split the data into training and testing sets with or without features based on the 'with_features' parameter
    if with_features:
        data_splits_dict[f"with_features_{year}"] = list(train_test_split(config, df, True))
    else:
        data_splits_dict[f"without_features_{year}"] = list(train_test_split(config, df, False))
    
    return data_splits_dict, df


def update_deepAR_parameters(config, new_parameters):
    ''' 
    Create a new DeepAR-Estimator with adjusted hyperparameters. Hyperparameters that are not provided will be set to the value in the Configuration.py file.

    Parameters:
        config (Configuration): The existing configuration object.
        new_parameters (dict): A dictionary containing the new parameter values to be updated.

    Returns:
        DeepAREstimator: The updated DeepAR estimator with the new parameters.

    Note:
        new_parameters must be a dict containing the exact keys used in config.parameters.
    '''
    # Create a copy of the existing parameters
    parameters = config.parameters.copy()
    
    # Update the parameters with the new values from new_parameters
    for key in new_parameters.keys():
        if key in parameters.keys():
            parameters[key] = new_parameters[key]
        else:
            print(f"This key {key} isn't available in config.parameters! The default config will be maintained.")
    
    # Create the updated DeepAREstimator with the new parameters
    deeparestimator = DeepAREstimator(freq=parameters["freq"],
                                      context_length=parameters["context_length"],
                                      prediction_length=parameters["prediction_length"],
                                      num_layers=parameters["num_layers"],
                                      num_cells=parameters["num_cells"],
                                      cell_type=parameters["cell_type"],
                                      dropout_rate=parameters["dropout_rate"],              
                                      trainer=Trainer(epochs=parameters["epochs"],
                                                      learning_rate=parameters["learning_rate"]),
                                      batch_size=parameters["batch_size"],
                                      distr_output=parameters["distr_output"],
                                      use_feat_static_real=parameters["use_feat_static_real"],
                                      use_feat_dynamic_real=parameters["use_feat_dynamic_real"],
                                      use_feat_static_cat=parameters["use_feat_static_cat"],
                                      cardinality=parameters["cardinality"],
                                     )
    
    return deeparestimator


def update_FNN_parameters(config, new_parameters):
    ''' 
    Create a new FNN-Estimator with adjusted hyperparameters. Hyperparameters that are not provided will be set to the value in the Configuration.py file.

    Parameters:
        config (Configuration): The existing configuration object.
        new_parameters (dict): A dictionary containing the new parameter values to be updated.

    Returns:
        SimpleFeedForwardEstimator: The updated SimpleFeedForward estimator with the new parameters.

    Note:
        new_parameters must be a dict containing the exact keys used in config.fnnparameters.
    '''
    # Create a copy of the existing parameters
    parameters = config.fnnparameters.copy()
    
    # Update the parameters with the new values from new_parameters
    for key in new_parameters.keys():
        if key in parameters.keys():
            parameters[key] = new_parameters[key]
        else:
            print(f"This key {key} isn't available in config.parameters! The default config will be maintained.")
    
    # Create the updated SimpleFeedForwardEstimator with the new parameters
    fnnestimator = SimpleFeedForwardEstimator(num_hidden_dimensions=parameters["num_hidden_dimensions"],
                                              prediction_length=parameters["prediction_length"],
                                              context_length=parameters["context_length"],
                                              distr_output=parameters["distr_output"],
                                              batch_size=parameters["batch_size"],
                                              batch_normalization=parameters["batch_normalization"],
                                              trainer=Trainer(epochs=parameters["epochs"],
                                                              num_batches_per_epoch=parameters["num_batches_per_epoch"]),
                                             )
    
    return fnnestimator



def fitDeepAR(config, train, test, configuration):
    """
    Fits a DeepAR model with the given configuration and evaluates its performance using the mean absolute Quantile Loss (MAE of Quantile Loss) normalized by the number of predictions per series.

    Parameters:
        config (dict): Configuration parameters for the DeepAR model. -> This contains the parameters of the DeepAR model amd not!! the Configuration class object!
        train (gluonts.dataset.common.Dataset): Training data in the GluonTS Dataset format.
        test (gluonts.dataset.common.Dataset): Test data in the GluonTS Dataset format.
        configuration (Configuration): Configuration object containing additional settings. -> This is an instance of the Configuration class implemented in Configuration.py!!

    Returns:
        float: The mean absolute WIS (Weighted Interval Score) for the DeepAR model.

    Note:
        The configuration should contain the parameters required for the DeepAR model.
        The "prediction_length" parameter can be provided in either the "config" or "configuration" objects.
        The mean WIS is calculated as the mean absolute Quantile Loss divided by the number of predictions per series (411 for each district).
    """
    # Update the DeepAR configuration with the given parameters
    deeparestimator = update_deepAR_parameters(configuration, config)
    
    # Fit the DeepAR model on the training data and make predictions on the test data
    forecasts, tss = model(train, test, deeparestimator)
    
    # Evaluation with the quantiles of the configuration and calculation of the mean_WIS
    evaluator = Evaluator(quantiles=configuration.quantiles)    
    agg_metrics = evaluator(tss, forecasts)[0]
    
    # Determine the prediction_length from the configuration or provided in the config dictionary
    if "prediction_length" in config.keys():
        prediction_length = config["prediction_length"]
    else:
        prediction_length = configuration.parameters["prediction_length"]
    
    # Calculate the mean WIS as the mean absolute Quantile Loss divided by the number of predictions per series (411 for each district)
    mean_WIS = agg_metrics["mean_absolute_QuantileLoss"] / (prediction_length * 411)
    
    return mean_WIS


def objectiveDeepAR(config, train, test, configuration):
    """
    The objective function used for optimization in the DeepAR model.
    
    Parameters:
        config (dict): Configuration parameters for the DeepAR model. -> This contains the parameters of the DeepAR model amd not!! the Configuration class object!
        train (gluonts.dataset.common.Dataset): Training data in the GluonTS Dataset format.
        test (gluonts.dataset.common.Dataset): Test data in the GluonTS Dataset format.
        configuration (Configuration): Configuration object containing additional settings. -> This is an instance of the Configuration class implemented in Configuration.py!!

    Note:
        The configuration should contain the parameters required for the DeepAR model.
        The "prediction_length" parameter can be provided in either the "config" or "configuration" objects.
        The objective is to minimize the mean_WIS (mean Weighted Interval Score) obtained from the DeepAR model.
    """
    # Calculate the mean_WIS score by fitting the DeepAR model and evaluating its performance
    score = fitDeepAR(config, train, test, configuration)

    # Report the mean_WIS score to the optimization session
    session.report({"mean_WIS": score})



def fitFNN(config, train, test, configuration):
    """
    Fits the Feed-Forward Neural Network (FNN) model with the given configuration.

    Parameters:
        config (dict): Configuration parameters for the FNN model. -> This contains the parameters of the DeepAR model amd not!! the Configuration class object!
        train (gluonts.dataset.common.Dataset): Training data in the GluonTS Dataset format.
        test (gluonts.dataset.common.Dataset): Test data in the GluonTS Dataset format.
        configuration (Configuration): Configuration object containing additional settings. -> This is an instance of the Configuration class implemented in Configuration.py!!

    Returns:
        float: The mean Weighted Interval Score (mean_WIS) obtained from the FNN model.

    Note:
        The configuration should contain the parameters required for the FNN model.
        The "prediction_length" parameter can be provided in either the "config" or "configuration" objects.
        The objective is to calculate the mean_WIS score obtained from the FNN model.
    """
    # Update the FNN model parameters based on the given configuration
    fnnestimator = update_FNN_parameters(configuration, config)

    # Train the FNN model on the provided training dataset
    predictor = fnnestimator.train(train)

    # Make predictions on the test dataset using the trained FNN model
    forecast_it, ts_it = make_evaluation_predictions(dataset=test, predictor=predictor, num_samples=100)
    forecasts = list(forecast_it)
    tss = list(ts_it)

    # Evaluate the FNN model performance using the quantiles specified in the configuration
    evaluator = Evaluator(quantiles=configuration.quantiles)
    agg_metrics = evaluator(tss, forecasts)[0]

    # Calculate the mean_WIS score as mean absolute Quantile Loss divided by the number of involved locations per prediction
    if "prediction_length" in config.keys():
        prediction_length = config["prediction_length"]
    else:
        prediction_length = configuration.fnnparameters["prediction_length"]
    mean_WIS = agg_metrics["mean_absolute_QuantileLoss"] / (prediction_length * 411)

    return mean_WIS


def objectiveFNN(config, train, test, configuration):
    """
    Optimization objective function for FNN hyperparameter tuning.

    Parameters:
        config (dict): Configuration parameters for the FNN model. -> This contains the parameters of the DeepAR model amd not!! the Configuration class object!
        train (gluonts.dataset.common.Dataset): Training data in the GluonTS Dataset format.
        test (gluonts.dataset.common.Dataset): Test data in the GluonTS Dataset format.
        configuration (Configuration): Configuration object containing additional settings. -> This is an instance of the Configuration class implemented in Configuration.py!!

    Note:
        The configuration should contain the parameters required for the FNN model.

    Returns:
        None: The mean_WIS score is reported to the optimization session.
    """
    # Calculate the mean_WIS score using the fitFNN function
    score = fitFNN(config, train, test, configuration)

    # Report the mean_WIS score to the optimization session
    session.report({"mean_WIS": score})


# Functions needed after the hp_run

def restore_HP_results(experiment_path, objective, train, test, configuration):
    """
    Restore hyperparameter tuning results from a previous (failed) experiment.

    Parameters:
        experiment_path (str): Path to the directory containing the hyperparameter tuning experiment results.
        objective (function): The optimization objective function used to produce the restored results.
        train (gluonts.dataset.common.Dataset): Training data in the GluonTS Dataset format.
        test (gluonts.dataset.common.Dataset): Test data in the GluonTS Dataset format.
        configuration (Configuration): Configuration object containing additional settings.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the restored hyperparameter tuning results.
    """
    print(f"Loading results from {experiment_path}...")
    # Initialize the Ray Tune framework
    ray.init()

    # Restore the tuner object using the experiment path and the objective function
    restored_tuner = tune.Tuner.restore(experiment_path, trainable=tune.with_parameters(objective, train=train, test=test, configuration=configuration))

    # Get the results grid containing information about different hyperparameter configurations
    result_grid = restored_tuner.get_results()

    # Convert the results grid to a pandas DataFrame
    results_df = result_grid.get_dataframe()

    # Shutdown the Ray Tune framework
    ray.shutdown()

    return results_df



def generate_model_results_by_hp_dict(df, hp_search_space):
    """
    Filter out each possible combination in the hp_search_space and corresponding modelRun results.
    Then concatenate them again and check if the modelRuns are matching.

    Parameters:
        df (pd.DataFrame): DataFrame containing the model run results.
        hp_search_space (dict): Dictionary representing the hyperparameter search space.

    Returns:
        dict: A dictionary containing the model run results grouped by hyperparameter combinations.
        pd.DataFrame: A concatenated DataFrame containing the overall model run results.
    """
    model_results_by_hp = {}

    # Save the relevant hyperparameters for configurations (exclude dependent parameters)
    hyperparameters = [hyperparameter for hyperparameter in hp_search_space.keys() if not "cardinality" in hyperparameter]

    # Determine all possible combinations within the grid set up by combinations of unique values
    hp_grid_combinations = list(itertools.product(*[list(df["config/"+hp].unique()) for hp in hyperparameters]))

    # Build up an index out of the combination that is true for every value
    for hp_grid_combination in hp_grid_combinations:
        # Determine the index that combines the hp configuration
        index_list = [df["config/"+k] == v for k, v in zip(hyperparameters, hp_grid_combination)]
        combined_index = np.logical_and.reduce(index_list)

        # Filter and combine the results for the combination
        df.loc[combined_index, "shape"] = df.loc[combined_index].shape[0]
        df.loc[combined_index, "model_WIS_variance"] = df.loc[combined_index, "mean_WIS"].var()
        df.loc[combined_index, "model_WIS_sd"] = np.sqrt(df.loc[combined_index, "mean_WIS"].var())
        df.loc[combined_index, "model_WIS_mean"] = df.loc[combined_index, "mean_WIS"].mean()
        df.loc[combined_index, "model_WIS_median"] = df.loc[combined_index, "mean_WIS"].median()
        df.loc[combined_index, "model_time_variance"] = df.loc[combined_index, "time_total_s"].var()
        df.loc[combined_index, "model_time_sd"] = np.sqrt(df.loc[combined_index, "time_total_s"].var())
        df.loc[combined_index, "model_time_mean"] = df.loc[combined_index, "time_total_s"].mean()
        df.loc[combined_index, "model_time_median"] = df.loc[combined_index, "time_total_s"].median()

        # Store the filtered results for the combination in the dictionary
        model_results_by_hp[str(hp_grid_combination)] = df[combined_index]

    # Concatenate the filtered results to create an overall DataFrame
    overall_df = pd.DataFrame()
    for key in list(model_results_by_hp.keys())[:]:
        overall_df = pd.concat([overall_df, model_results_by_hp[key]])

    # Calculate the number of model runs per combination and store it in a DataFrame
    modelruns_per_combination = pd.DataFrame(overall_df["shape"].value_counts())
    modelruns_per_combination.index.names = ["modelruns_per_combination"]
    modelruns_per_combination.rename(columns={'shape': 'total_modelruns'}, inplace=True)
    modelruns_per_combination['independent_combinations'] = modelruns_per_combination['total_modelruns'] / modelruns_per_combination.index

    if len(modelruns_per_combination) > 1:
        print("There are combinations with fewer modelRuns!!")

    # Print and return the number of model runs per combination and the model results dictionary
    print(modelruns_per_combination)
    return model_results_by_hp, overall_df



