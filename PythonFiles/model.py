import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import itertools
import gluonts
from gluonts.mx import Trainer, DeepAREstimator
from gluonts.dataset.common import ListDataset
from gluonts.dataset.pandas import PandasDataset
from gluonts.evaluation import make_evaluation_predictions, Evaluator
from gluonts.dataset.split import split, TestData
from gluonts.dataset.util import to_pandas
from gluonts.dataset.rolling_dataset import generate_rolling_dataset,StepStrategy

def preprocessing(config, df, check_count=False, output_type="PD"):
    """
    This function processes the data into either a correctly spaced pd.DataFrame, a PandasDataset, a ListDataset or
    a pd.Dataframe where only the index has been set.
    We also have the option to receive an output of the count of each location, with fewer observations than the maximum
    observations within the training and testing period.
    """
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    if check_count:
        count_dict = {}
        for location in df.location.unique():
            # save the amount of values within the train & test time period into the count_dict
            location_df = df.loc[(df['location'] == location) & (df.index > config.train_start_time) & (df.index <= config.test_end_time),:]
            count_dict[location] = location_df.shape[0]
        # print out the distribution of each region with missing values
        print('LK mit weniger als' + str(max(count_dict.values())))
        missing_values_dict = {k : v for k, v in count_dict.items() if v < max(count_dict.values())}
        print(missing_values_dict)
        return df, missing_values_dict
    
    if output_type in ['PD', 'LD', 'corrected_df']:
        #Create a DataFrame Blueprint
        start, end = min(df.index), max(df.index)
        correctly_spaced_index = pd.date_range(start=start, end=end,freq=config.parameters["freq"])
        correctly_spaced_location_df = pd.DataFrame(index=correctly_spaced_index)
        correctly_spaced_df = pd.DataFrame()
        location_list = df.loc[:, 'location'].unique()
        for location in location_list:
            temporary_df = correctly_spaced_location_df.join(df.loc[df.location == location])
            temporary_df['location'] = temporary_df['location'].fillna(location)
            correctly_spaced_df = pd.concat([correctly_spaced_df, temporary_df])
        if output_type == "PD":
            df = PandasDataset.from_long_dataframe(dataframe=correctly_spaced_df,item_id='location', target="value",freq=config.parameters["freq"])
        if output_type == "LD":
            df = ListDataset([{"start": min(correctly_spaced_index), "target": correctly_spaced_df.loc[correctly_spaced_df.location == location, 'value']}
                              for location in location_list], freq=config.parameters["freq"])
        if output_type == "corrected_df":
            return correctly_spaced_df
    return df

def train_test_split(config, df, with_features=False):
    locations = list(df.location.unique())
    # Split with the usual time and the 
    train_set = df.loc[(df.index <= config.train_end_time) & (df.index >= config.train_start_time), :]
    test_set = df.loc[(df.index >= config.train_start_time) & (df.index <= config.test_end_time), :]
    start_time = min(test_set.index.difference(train_set.index))
    end_time = max(test_set.index.difference(train_set.index))
    if with_features:
        # Format the train and test_set into a PandasDataset with features
        train_set = PandasDataset.from_long_dataframe(dataframe=train_set, item_id='location', target="value", freq=config.parameters["freq"],
                                                      feat_static_real=["population"], feat_static_cat=list(locations),
                                                      feat_dynamic_real=["week"])
        test_set = PandasDataset.from_long_dataframe(dataframe=test_set, item_id='location', target="value", freq=config.parameters["freq"],
                                                     feat_static_real=["population"], feat_static_cat=list(locations),
                                                     feat_dynamic_real=["week"])
    else:
        # Format the train and test_set into a PandasDataset without features
        train_set = PandasDataset.from_long_dataframe(dataframe=train_set, item_id='location', target="value", freq=config.parameters["freq"])
        test_set = PandasDataset.from_long_dataframe(dataframe=test_set, item_id='location', target="value", freq=config.parameters["freq"])
    # Create the rolling version of the test set with windows of length config.prediction_length and following windows of 1 timestep
    test_set = generate_rolling_dataset(dataset=test_set,
                                        strategy=StepStrategy(prediction_length=config.parameters["prediction_length"], step_size=1),
                                        start_time=pd.Period(start_time,config.parameters["freq"]),
                                        end_time=pd.Period(end_time,config.parameters["freq"])
                                        )
    return train_set, test_set


def model(training_data, test_data, estimator):
    """
    This function fits a given estimator.
    Then this estimator is fit with the given training_data and the forecasts,
    aswell as the true values for the test_data are calculated via the make_evaluation_predictions function from gluonts. 
    """
    # Train and Test the predictor
    predictor = estimator.train(training_data=training_data)
    forecast_it, ts_it = make_evaluation_predictions(
                         dataset=test_data,  
                         predictor=predictor,  
                         num_samples=100,  
                         )
    print(f"Ende make_evaluation_prediction: {datetime.now()}")
    # unpack Iterator-Objects into lists (NOTE: this may take longer than the actual fitting process!) -> Some options for speeding up are: 1. lowering num_samples, 2. use DF's instead of lists, 3. parallelisation...
    forecasts = list(forecast_it)
    tss = list(ts_it)
    print(f"Ende umformen in Listen: {datetime.now()}")
    
    return forecasts, tss

def forecast_by_week(config, train_set, test_set, locations, models_dict):
    #define the dicts that are going to be output later on
    evaluator_df_dict = {}
    forecasts_dict = {}
    #iterate through the given models and fit them
    for key in models_dict.keys():
        forecasts, tss = model(train_set, test_set, models_dict[key])
        # Splitting the forecasts into their weekly contribution
        split_tss = split_forecasts_by_week(config, forecasts, tss, locations, 4, equal_time_frame=True)[1]
        forecast_dict ={1 : split_forecasts_by_week(config, forecasts, tss, locations, 1, equal_time_frame=True)[0],
                        2 : split_forecasts_by_week(config, forecasts, tss, locations, 2, equal_time_frame=True)[0],
                        3 : split_forecasts_by_week(config, forecasts, tss, locations, 3, equal_time_frame=True)[0],
                        4 : split_forecasts_by_week(config, forecasts, tss, locations, 4, equal_time_frame=True)[0]}
        # Evaluation with the quantiles of the configuration
        evaluator = Evaluator(quantiles=config.quantiles)
        evaluator_df = pd.DataFrame()         
        # iterate over the 4 different week-aheads
        for forecast in forecast_dict.values():
            agg_metrics, item_metrics = evaluator(split_tss, forecast)
            d = {key for key in forecast_dict if forecast_dict[key] == forecast}
            for location in locations[:]:
                #rename location id to differentiate between the weekahead predictions and concat
                item_metrics.loc[item_metrics.item_id == f"{location}", "item_id"] = f"{location} {d}"
                evaluator_df = pd.concat([evaluator_df, item_metrics[item_metrics.item_id == f"{location} {d}"]])
            agg_metrics["item_id"] = f"aggregated {d}"
            evaluator_df = pd.concat([evaluator_df, pd.DataFrame(agg_metrics, index=[0])])
        # produce the average Quantile Loss metric by dividing the mean absolute QL through the number of involved locations per weekahead, which is usually 411 (each district)
        included_locations = [item_id for item_id in evaluator_df.item_id.unique() if "aggregated" not in item_id if "1" in item_id]
        evaluator_df.loc[evaluator_df.item_id.isin([item_id for item_id in evaluator_df.item_id if "aggregate" in item_id]), "mean_WIS"] = evaluator_df.loc[evaluator_df.item_id.isin([item_id for item_id in evaluator_df.item_id if "aggregate" in item_id]),"mean_absolute_QuantileLoss"]/len(included_locations)
        evaluator_df_dict[key] = evaluator_df
        forecasts_dict[key] = forecast_dict
    return forecasts_dict, evaluator_df_dict


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


def split_forecasts_by_week(config, forecasts, tss, locations, week, equal_time_frame=False):
    """
    Splits up a Forecast-List into the forecasts by a given [week]-week ahead.
    Test values and forecasts are results of calling model(), whilst before a rolling window is applied.
    """   
    # First determine the amount of forecasted windows per location, so we can iterate through and specify the correct index later
    windows_per_location = int(len(forecasts) / len(locations))
    week_ahead_forecasts = []
    split_tss = []
    for location in locations:
        if equal_time_frame:
            # choose an index that sets the starting date of different week ahead within 1 and 4 week ahead to equal each other
            first_time_point_of_location = windows_per_location + windows_per_location*locations.index(location) - ((-week) % 4) - 1
            for_loop_end = first_time_point_of_location - windows_per_location + ((-week) % 4) + week
        else:
            # define the index of the time wise first forecast point
            first_time_point_of_location = windows_per_location + windows_per_location*locations.index(location)-1
            for_loop_end = first_time_point_of_location - windows_per_location
        start_date_list = []
        # Append the True underlying values (of the complete test window) to split_tss for the current location
        split_tss.append(tss[windows_per_location*locations.index(location)])
        # Add the timewise first (and index-wise last) [num_samples]-arrays of the corresponding week to [weekly_samples_array]
        weekly_samples_array = forecasts[first_time_point_of_location].samples[:, (week-1):week]
        for k in range(first_time_point_of_location - 1,
                       for_loop_end, -1):
            # Reverse iterate through the windows of each location, as the time-wise first forecasts are last in the forecast-list
            # and concatenate the array with the corresponding values of each [week]-week ahead forecast
            weekly_samples_array = np.concatenate((weekly_samples_array, forecasts[k].samples[:, (week-1):week]), axis=1)
        
        # Save the correct starting time, determined by the first [start_date] of the location and the [week] parameter
        start_date = pd.date_range(start=forecasts[first_time_point_of_location].start_date.to_timestamp(), periods=week, freq=config.parameters["freq"])[-1]
        # append the filtered [weekly_samples_array]-array and the correct [start_date] as a SampleForecast-Object for each location
        week_ahead_forecasts.append(
                                    gluonts.model.forecast.SampleForecast(info=forecasts[first_time_point_of_location].info,
                                                                          item_id=forecasts[first_time_point_of_location].item_id,
                                                                          samples=weekly_samples_array,
                                                                          start_date=pd.Period(start_date,freq=config.parameters["freq"]),
                                                 )
        )
    return week_ahead_forecasts, split_tss
    

    
def make_one_ts_prediction(config, df, location="LK Bad Dürkheim"):
    """
    This function makes a model for the univariat time series of the given [location].  It also plots the resulting prediction(s).
    """
    #Process the df into a uniformly spaced df
    one_ts_df = df.loc[df.location == location, ["value", 'location', 'date']]
    one_ts_df = preprocessing(config, one_ts_df, check_count=False, output_type="corrected_df")
    #seperate the intervals for training and testing
    train_set = one_ts_df.loc[(one_ts_df.index <= config.train_end_time) & (one_ts_df.index >= config.train_start_time),:]
    test_set = one_ts_df.loc[(one_ts_df.index >= config.train_start_time) & (one_ts_df.index <= config.test_end_time),:]
    #select the correct dates for splitting within the test data for each window
    window_dates = []
    for window in range(1, config.windows):
        unique_weeks = test_set.index.unique()
        selected_split_week = unique_weeks[-window*config.parameters["prediction_length"] : -window*config.parameters["prediction_length"] + 1]
        window_dates.append(datetime(selected_split_week.year[0], selected_split_week.month[0], selected_split_week.day[0]))
    #also add the last date available
    window_dates.append(config.test_end_time)
    window_dates.sort()
    #define the list of dfs of each testing window
    test_windows = [test_set.loc[test_set.index < window_date,:] for window_date in window_dates]
    #Format the train and test_set into a PandasDataset
    train_set = PandasDataset.from_long_dataframe(dataframe=train_set, item_id='location', target="value", freq=config.parameters["freq"])
    test_set = PandasDataset(test_windows, target="value", freq=config.parameters["freq"])
    #train and evaluate the model
    forecasts,tss = model(train_set, test_set, config.deeparestimator)
    #plot the forecasts
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    plt.title(f'{location}')
    #first plot the time series as a whole (x-axis: Date, y-axis: influenza-values)
    plt.plot((one_ts_df.loc[(one_ts_df['location'] == location) &
                            (one_ts_df.index <= config.test_end_time) &
                            (one_ts_df.index >= config.train_start_time)].index),
             one_ts_df.loc[(one_ts_df['location'] == location) &
                           (one_ts_df.index <= config.test_end_time) &
                           (one_ts_df.index >= config.train_start_time), 'value'])
    plt.grid(which="both")
    #define the colors to use for each different window
    color = ["g", "r", "purple", "black", "yellow", "grey"] * config.windows
    for k in range(0, config.windows):
        forecast_entry = forecasts[k]
        prediction_intervals = (50.0, 90.0)
        legend = ["train_set observations", "median prediction"] +\
                 [f"{k}% prediction interval" for k in prediction_intervals][::-1]
        forecast_entry.plot(prediction_intervals=prediction_intervals, color=color[k])
    plt.grid(which="both")
    plt.show()
    return forecasts, tss


# Evaluation Plots


def print_forecasts_by_week(config, corrected_df, forecast_dict, locations, week_ahead_list, plot_begin_at_trainstart=False):
    '''
    Prints out plots for the given week-Ahead forecasts of given locations. It needs the initial corrected dataframe, as well as the forecast_dict
    that contains the different week-ahead forecasts.
    The start of the plot time axis, can be set to the training start time (TRUE) or the testing start time (FALSE).
    '''
    for location in locations:
        for week_ahead in week_ahead_list:
            #plot the forecasts
            fig, ax = plt.subplots(1, 1, figsize=(10, 7))
            plt.title(f'{location} - WA{week_ahead}')
            # determine the beginning of the time series
            if plot_begin_at_trainstart == True:
                plot_start_time = config.train_start_time
            else:
                plot_start_time = config.train_end_time
            #first plot the time series as a whole (x-axis: Date, y-axis: influenza-values)
            plt.plot((corrected_df.loc[(corrected_df['location'] == location) &
                                    (corrected_df.index <= config.test_end_time) &
                                    (corrected_df.index >= plot_start_time)].index),
                     corrected_df.loc[(corrected_df['location'] == location) &
                                   (corrected_df.index <= config.test_end_time) &
                                   (corrected_df.index >= plot_start_time),'value'])
            plt.grid(which="both")
            # select the right week-ahead forecast entry for a set location
            forecast_entry = forecast_dict[list(forecast_dict.keys())[week_ahead-1]][locations.index(location)]
            prediction_intervals = (50.0, 90.0)
            forecast_entry.plot(prediction_intervals=prediction_intervals, color="g")
            plt.grid(which="both")
            plt.show()


def plot_coverage(config, evaluator_df_dict, locations=None):
    """
    Given a dictionary, where the values consist of evaluation_df's, this function is going to create plots of the 4 different week-ahead coverages.  
    However, the weekly performances have to be under the "item_id" with f.e. "aggregated {1}" for the 1 week-ahead metrics.
    """
    week_coverage_dict = {}
    coverage_columns = [col for col in evaluator_df_dict[list(evaluator_df_dict.keys())[0]].columns if "Coverage" in col]
    coverage_columns.remove("MAE_Coverage")
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(16, 9))
    for week in range(1,5):
        if week == 1:
            plotnumber = (0, 0)
        if week == 2:
            plotnumber = (1, 0)
        if week == 3:
            plotnumber = (0, 1)
        if week == 4:
            plotnumber = (1, 1)
        for key in evaluator_df_dict.keys():
            week_coverage_dict[week] = evaluator_df_dict[key].loc[evaluator_df_dict[key].item_id.isin(["aggregated {"+ f"{week}" + "}"]), coverage_columns]
            axs[plotnumber].plot([0.0, 1.0], [0.0, 1.0])
            axs[plotnumber].scatter(config.quantiles, evaluator_df_dict[key].loc[evaluator_df_dict[key].item_id.isin(["aggregated {" + f"{week}" + "}"]), coverage_columns])
            axs[plotnumber].plot(config.quantiles, evaluator_df_dict[key].loc[evaluator_df_dict[key].item_id.isin(["aggregated {" + f"{week}" + "}"]), coverage_columns].T, label=f"{key}")
            axs[plotnumber].title.set_text(f"{week}-Week Ahead Coverage")
            axs[plotnumber].legend()
            

def generate_model_results_by_hp_dict(df, hp_search_space): 
    model_results_by_hp = {}
    
    # save the relevant hyperaparameters for configurations (exclude dependent parameters)
    hyperparameters = [hyperparameter for hyperparameter in hp_search_space.keys()\
                       if not "cardinality" in hyperparameter]

    # determine all possible combinations within the grid set up by combinatios of unique values
    hp_grid_combinations = list(itertools.product(*[list(df["config/"+hp].unique()) for hp in hyperparameters]))

    # build up an index out of the combination that is true for every value
    for hp_grid_combination in hp_grid_combinations:
        index_list = [df["config/"+k] ==v for k,v in zip(hyperparameters, hp_grid_combination)]
        combined_index = np.logical_and.reduce(index_list)
        df.loc[combined_index,"model_WIS_variance"] = df.loc[combined_index,"mean_WIS"].var()
        df.loc[combined_index,"model_WIS_sd"] = np.sqrt(df.loc[combined_index,"mean_WIS"].var())
        df.loc[combined_index,"model_WIS_mean"] = df.loc[combined_index,"mean_WIS"].mean()
        df.loc[combined_index,"model_time_variance"] = df.loc[combined_index,"time_total_s"].var()
        df.loc[combined_index,"model_time_sd"] = np.sqrt(df.loc[combined_index,"time_total_s"].var())
        df.loc[combined_index,"model_time_mean"] = df.loc[combined_index,"time_total_s"].mean()
        model_results_by_hp[str(hp_grid_combination)] = df[combined_index]
    return model_results_by_hp

def plot_model_results_by_hp(model_results_by_hp, number_of_plots=30, col="mean_WIS",figsize=(16, 9)):
    dfs, labels = [], [] 
    for key in list(model_results_by_hp.keys())[:number_of_plots]:
        dfs.append(model_results_by_hp[key])
        labels.append(key)
    # Create a figure and axis object
    fig, ax = plt.subplots(figsize=figsize)

    # Create a list to hold the positions for each boxplot
    pos = range(1, len(dfs) * 2, 2)

    # Loop through each dataframe and plot a boxplot
    for i, model_df in enumerate(dfs):
        ax.boxplot(model_df[col], positions=[pos[i]])

    # Set the x-axis ticks and tick labels
    ax.set_xticks(pos)
    ax.set_xticklabels(labels)

    # Set the y-axis label
    ax.set_ylabel(f'{col}')

    # Set the title
    ax.set_title(f'Boxplots of {number_of_plots} different models')

    fig.autofmt_xdate(rotation=60, ha='right')
    # Show the plot
    plt.show()    
            
def hyperparameter_boxplots(results_df, hp_search_space, col="mean_WIS"):
    """
    Plot the hyperparameters as boxplots.
    """
    # Create a dict of filtered dfs and x_tick- renamings
    hp_plots = dict()
    for key in hp_search_space.keys():
        if type(hp_search_space[key]) == type(dict()):
            search_grid = hp_search_space[key][list(hp_search_space[key].keys())[0]]
            hp_plots[key] = {"cols" : [f"{i} {key}" for i in search_grid], "df": [results_df.loc[results_df[f'config/{key}']==i][col] for i in search_grid]}
    
    # plot the boxplots
    nrows = int(len(hp_plots.keys())/2) + int(len(hp_plots.keys())%2)
    fig, axs = plt.subplots(nrows=nrows, ncols=2, figsize=(16, 9), sharey=True)
    fig.tight_layout(pad=1.2)
    plotnumber = [0, 0]
    for key in hp_plots.keys():
        if list(hp_plots.keys()).index(key)%2 == 1:
            plotnumber[1] = 1
        else:
            if list(hp_plots.keys()).index(key) > 1:
                plotnumber[0] += 1
            plotnumber[1] = 0
        axs[tuple(plotnumber)].boxplot(hp_plots[key]["df"])
        axs[tuple(plotnumber)].set_title(key)
        axs[tuple(plotnumber)].set_xticks([i for i in range(1, len(hp_plots[key]["df"])+1)], hp_plots[key]["cols"])
        axs[tuple(plotnumber)].set_ylabel(col)
    plt.show()

    