import ray
from ray import tune, air
from ray.air import session
from ray.tune import ResultGrid
from PythonFiles.model import model, preprocessing, split_forecasts_by_week, forecast_by_week, train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from gluonts.mx import Trainer, DeepAREstimator
from gluonts.mx.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.evaluation import make_evaluation_predictions, Evaluator

def get_data(truncate=False, with_features=True):
    import os
    os.chdir('/home/reffert/DeepAR_InfluenzaForecast')
    from PythonFiles.Configuration import Configuration
    from PythonFiles.model import train_test_split, preprocessing
    config = Configuration()
    # import the data
    influenza_df = pd.read_csv("/home/reffert/DeepAR_InfluenzaForecast/Notebooks/DataProcessing/influenza.csv", sep=',')
    population_df = pd.read_csv("/home/reffert/DeepAR_InfluenzaForecast/Notebooks/DataProcessing/PopulationVector.csv", sep=',')
    neighbourhood_df = pd.read_csv("/home/reffert/DeepAR_InfluenzaForecast/Notebooks/DataProcessing/AdjacentMatrix.csv", sep=',', index_col=0)
    
    data_splits_dict = {}
    locations = list(influenza_df.location.unique())
    #Process the df into a uniformly spaced df
    df = influenza_df.loc[influenza_df.location.isin(locations), ['value', 'location', 'date','week']]
    df = preprocessing(config, df, check_count=False, output_type="corrected_df")
    for location in locations:
        df.loc[df.location == location, "population"] = int(population_df.loc[population_df.Location == location, "2011"].values[0])
        df.loc[df.location == location, locations] = neighbourhood_df.loc[neighbourhood_df.index==location,locations].values[0].astype(int)
    # set the correct date to truncate or not truncate the available data
    if truncate:
        config.train_start_time = datetime(2010,1,1,0,0,0)
        year = "2010"
    else:
        config.train_start_time = datetime(1999,1,1,0,0,0)
        year = "2001"
    config.train_end_time = datetime(2016,9,30,23,0,0)
    config.test_end_time = datetime(2018,9,30,23,0,0)
    
    if with_features:
        data_splits_dict[f"with_features_{year}"] = list(train_test_split(config, df, True))
    else:
        data_splits_dict[f"without_features_{year}"] = list(train_test_split(config, df, False))
    return data_splits_dict, df


def update_deepAR_parameters(config, new_parameters):
    ''' 
    This function updates the DeepAR-configuration in the Configuration.py file. 
    Note that new_parameters must be a dict containing the exact keys used in config.parameters.
    '''
    parameters = config.parameters.copy()
    for key in new_parameters.keys():
        if key in parameters.keys():
            parameters[key] = new_parameters[key]
        else:
            print(f"This key {key} isn't available in config.parameters! Thus the default config will maintain.")
    #update the deeparestimator in config
    deeparestimator = DeepAREstimator(freq=parameters["freq"],
                    context_length=parameters["context_length"],
                    prediction_length=parameters["prediction_length"],
                    num_layers=parameters["num_layers"],
                    num_cells=parameters["num_cells"],
                    cell_type=parameters["cell_type"],
                    dropout_rate = parameters["dropout_rate"],              
                    trainer=Trainer(epochs=parameters["epochs"],
                                    learning_rate=parameters["learning_rate"],),
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
    This function updates the DeepAR-configuration in the Configuration.py file. 
    Note that new_parameters must be a dict containing the exact keys used in config.parameters.
    '''
    parameters = config.fnnparameters.copy()
    for key in new_parameters.keys():
        if key in parameters.keys():
            parameters[key] = new_parameters[key]
        else:
            print(f"This key {key} isn't available in config.parameters! Thus the default config will maintain.")

    fnnestimator = SimpleFeedForwardEstimator(num_hidden_dimensions=parameters["num_hidden_dimensions"],
                                              prediction_length=parameters["prediction_length"],
                                              context_length=parameters["context_length"],
                                              distr_output=parameters["distr_output"],
                                              batch_size=parameters["batch_size"],
                                              batch_normalization=parameters["batch_normalization"],
                                              trainer=Trainer(epochs=parameters["epochs"],
                                                              num_batches_per_epoch=parameters["num_batches_per_epoch"],
                                                             ),
                                              )
    return fnnestimator


def fitDeepAR(config, train, test, configuration):
    deeparestimator = update_deepAR_parameters(configuration, config)
    forecasts, tss = model(train, test, deeparestimator)
    
    # Evaluation with the quantiles of the configuration and calculation of the mean_WIS
    evaluator = Evaluator(quantiles=configuration.quantiles)    
    agg_metrics = evaluator(tss, forecasts)[0]
    if "prediction_length" in config.keys():
        prediction_length = config["prediction_length"]
    else:
        prediction_length = configuration.parameters["prediction_length"]
    mean_WIS = agg_metrics["mean_absolute_QuantileLoss"]/(prediction_length*411)
    return mean_WIS

def objectiveDeepAR(config, train, test, configuration):
    score = fitDeepAR(config, train, test, configuration)
    session.report({"mean_WIS":score})


def fitFNN(config, train, test, configuration):
    
    fnnestimator = update_FNN_parameters(configuration, config)
    
    predictor = fnnestimator.train(train)
    
    forecast_it, ts_it = make_evaluation_predictions(dataset=test, predictor=predictor,num_samples=100)
    forecasts = list(forecast_it)
    tss = list(ts_it)
    
    # Evaluation with the quantiles of the configuration and calculation of the mean_WIS
    evaluator = Evaluator(quantiles=configuration.quantiles)    
    agg_metrics = evaluator(tss, forecasts)[0]
    if "prediction_length" in config.keys():
        prediction_length = config["prediction_length"]
    else:
        prediction_length = configuration.fnnparameters["prediction_length"]
    mean_WIS = agg_metrics["mean_absolute_QuantileLoss"]/(prediction_length*411)
    return mean_WIS

def objectiveFNN(config, train, test, configuration):
    score = fitFNN(config, train, test, configuration)
    session.report({"mean_WIS":score})

# Functions needed after the hp_run

def restore_HP_results(experiment_path, objective, train, test, configuration):
    print(f"Loading results from {experiment_path}...")
    ray.init()
    restored_tuner = tune.Tuner.restore(experiment_path, trainable=tune.with_parameters(objective, train=train, test=test, configuration=configuration))
    result_grid = restored_tuner.get_results()
    results_df = result_grid.get_dataframe()
    return results_df


def generate_model_results_by_hp_dict(df, hp_search_space): 
    """
    Filter out each possible combination in the hp_search_space and correpsonding modelRun results. 
    Then concatenate them again and check if the modelRuns are matching.
    """
    model_results_by_hp = {}
    
    # save the relevant hyperaparameters for configurations (exclude dependent parameters)
    hyperparameters = [hyperparameter for hyperparameter in hp_search_space.keys()\
                       if not "cardinality" in hyperparameter]

    # determine all possible combinations within the grid set up by combinatios of unique values
    hp_grid_combinations = list(itertools.product(*[list(df["config/"+hp].unique()) for hp in hyperparameters]))

    # build up an index out of the combination that is true for every value
    for hp_grid_combination in hp_grid_combinations:
        # determine the index that combines the hp configuration
        index_list = [df["config/"+k] ==v for k,v in zip(hyperparameters, hp_grid_combination)]
        combined_index = np.logical_and.reduce(index_list)
        # filter and combine the results for the combination
        df.loc[combined_index,"shape"] = df.loc[combined_index,].shape[0]
        df.loc[combined_index,"model_WIS_variance"] = df.loc[combined_index,"mean_WIS"].var()
        df.loc[combined_index,"model_WIS_sd"] = np.sqrt(df.loc[combined_index,"mean_WIS"].var())
        df.loc[combined_index,"model_WIS_mean"] = df.loc[combined_index,"mean_WIS"].mean()
        df.loc[combined_index,"model_WIS_median"] = df.loc[combined_index,"mean_WIS"].median()
        df.loc[combined_index,"model_time_variance"] = df.loc[combined_index,"time_total_s"].var()
        df.loc[combined_index,"model_time_sd"] = np.sqrt(df.loc[combined_index,"time_total_s"].var())
        df.loc[combined_index,"model_time_mean"] = df.loc[combined_index,"time_total_s"].mean()
        df.loc[combined_index,"model_time_median"] = df.loc[combined_index,"time_total_s"].median()
        model_results_by_hp[str(hp_grid_combination)] = df[combined_index]
        
    overall_df = pd.DataFrame()
    for key in list(model_results_by_hp.keys())[:]:
        overall_df = pd.concat([overall_df, model_results_by_hp[key]])

    modelruns_per_combination = pd.DataFrame(overall_df["shape"].value_counts())
    modelruns_per_combination.index.names = ["modelruns_per_combination"]
    modelruns_per_combination.rename(columns = {'shape':'total_modelruns'}, inplace = True)
    modelruns_per_combination['independent_combinations'] = modelruns_per_combination['total_modelruns'] / modelruns_per_combination.index
    if len(modelruns_per_combination)>1:
        print("There are combinations with fewer modelRuns!!")
    print(modelruns_per_combination)
    return model_results_by_hp, overall_df


