import pandas as pd
import numpy as np
import mxnet as mx
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
import os
os.chdir('/home/reffert/DeepAR_InfluenzaForecast')
from PythonFiles.rolling_dataset import generate_rolling_dataset,StepStrategy

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
                                                      static_feature_columns =["population"] + list(locations),
                                                      feat_dynamic_real=["week"])
        test_set = PandasDataset.from_long_dataframe(dataframe=test_set, item_id='location', target="value", freq=config.parameters["freq"],
                                                     static_feature_columns =["population"] + list(locations),
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
    #print(f"Ende make_evaluation_prediction: {datetime.now()}")
    # unpack Iterator-Objects into lists (NOTE: this may take longer than the actual fitting process!) -> Some options for speeding up are: 1. lowering num_samples, 2. use DF's instead of lists, 3. parallelisation...
    forecasts = list(forecast_it)
    tss = list(ts_it)
    #print(f"Ende umformen in Listen: {datetime.now()}")
    
    return forecasts, tss

def forecast_by_week(config, train_set, test_set, locations, models_dict, seed=None, results_dict=None):
    if seed !=None:
        mx.random.seed(seed)
        np.random.seed(seed)
    #define the dicts that are going to be output later on
    evaluator_df_dict = {}
    forecasts_dict = {}
    #iterate through the given models and fit them
    for key in models_dict.keys():
        if results_dict == None:
            forecasts, tss = model(train_set, test_set, models_dict[key])
        else:
            forecasts = results_dict[f"{key}_forecasts"]
            tss = results_dict[f"{key}_tss"]
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
    colors = config.colors * config.windows
    for k in range(0, config.windows):
        forecast_entry = forecasts[k]
        prediction_intervals = (50.0, 90.0)
        legend = ["train_set observations", "median prediction"] +\
                 [f"{k}% prediction interval" for k in prediction_intervals][::-1]
        forecast_entry.plot(prediction_intervals=prediction_intervals, color=colors[k])
    plt.grid(which="both")
    plt.show()
    return forecasts, tss

# Functions for the R_Forecasts

def process_R_results(config, results_df, influenza_df):
    '''
    Process the results_df by adding a date column and the corresponding true values.
    '''
    df = results_df.copy()
    # Split the values in the Time column saved as 822.1, 823.2,... as Time : 822, 823, ... and WeekAhead: 1, 2, ...
    df['Time'] = df['Time'].astype(str)
    df[['Time', 'WeekAhead']] = df['Time'].str.split('.', 1, expand=True)

    # Convert columns to appropriate types i.e., integer
    df['Time'] = df['Time'].astype(int)
    df['WeekAhead'] = df['WeekAhead'].astype(int)
    
    # The dates in the R representation of the influenza df are shifted by 1 position, example for SK München:
    # 821          0 2016-09-25
    # 822          0 2016-10-02
    # ...
    # 929          2 2018-10-21
    R_start = datetime(2016, 10, 2)
    R_end = datetime(2018, 10, 21)
    # determine the daterange of the forecast period
    daterange = pd.date_range(start=R_start, periods=len(results_df.Time.unique()),freq=config.parameters["freq"])
    influenza_df["date"] = pd.to_datetime(influenza_df["date"])
    # Iterate over the zipped pairs of time and date e.g., [(821,2016-09-25), ..., (929,2018-10-21)]  
    for i in zip(df.Time.unique(), daterange): 
        df.loc[df.Time == i[0], "date"] = i[1]
        
    df["date"] = pd.to_datetime(df["date"])
    
    # rename the location column so it matches with the influenza location columnname
    df.rename(columns = {'Location':'location'}, inplace = True)
    df = df.merge(influenza_df[["date", "value", "location"]], on = ["date", "location"]) 
    df.rename(columns = {'value':'true_value'}, inplace = True)
    
    # truncate the predictions to lie within the same date range
    start_date = df.loc[df["WeekAhead"]==4,"date"].min()
    end_date = df.loc[df["WeekAhead"]==1,"date"].max()
    # the gluonts models only have a one week ahead forecast up to 98 data points
    # therefore set the start and the length to equal the gluonts model period
    end_date = pd.date_range(start=start_date, freq=config.parameters["freq"], periods=98)[-1:][0]
    df = df.loc[(df["date"]>=start_date) & (df["date"]<=end_date)]
    return df

def quantile_loss(target: np.ndarray, forecast: np.ndarray, q: float) -> float:
    r"""
    .. math::

        quantile\_loss = 2 * sum(|(Y - \hat{Y}) * (Y <= \hat{Y}) - q|)
    """
    return 2 * np.sum(np.abs((forecast - target) * ((target <= forecast) - q)))

def coverage(target: np.ndarray, forecast: np.ndarray) -> float:
    r"""
    .. math::

        coverage = mean(Y <= \hat{Y})
    """
    return np.mean(target <= forecast)

def abs_target_sum(target) -> float:
    r"""
    .. math::

        abs\_target\_sum = sum(|Y|)
    """
    return np.sum(np.abs(target))

def evaluate_R_forecasts(config, df_dict, locations, processed_df):
    evaluator_df = pd.DataFrame({"item_id":["aggregated {" + f"{week}" + "}" for week in [1, 2, 3, 4]]+[f"{location} "+ "{" + f"{week}" + "}" for location in locations for week in [1,2,3,4]]})
    # determine the metrics for individual series
    for week_ahead in [1, 2, 3, 4]:
        print(f"Evaluating {week_ahead}/4 -- {datetime.now()}")
        df = df_dict[week_ahead].copy()
        for quantile in config.quantiles:
            for location in locations:
                # calculate the Quantile Loss and the Coverage for each region
                evaluator_df.loc[evaluator_df.item_id == str(f"{location} "+ "{" + f"{week_ahead}" + "}"),f"QuantileLoss[{quantile}]"] =\
                    quantile_loss(df.loc[df["location"]==location, "true_value"], df.loc[df["location"]==location, f"{quantile}"], quantile)
                
                evaluator_df.loc[evaluator_df.item_id == str(f"{location} "+ "{" + f"{week_ahead}" + "}"),f"Coverage[{quantile}]"] =\
                    coverage(df.loc[df["location"]==location, "true_value"], df.loc[df["location"]==location, f"{quantile}"])

            evaluator_df.loc[evaluator_df.item_id == str("aggregated {" + f"{week_ahead}" + "}"),f"QuantileLoss[{quantile}]"] =\
                quantile_loss(df["true_value"], df[f"{quantile}"], quantile)
            
            evaluator_df.loc[evaluator_df.item_id == str("aggregated {" + f"{week_ahead}" + "}"),f"Coverage[{quantile}]"] =\
                coverage(df["true_value"], df[f"{quantile}"])
            
    # add the aggregate metrics
    evaluator_df["abs_target_sum"] = abs_target_sum(processed_df["true_value"])
    #for week_ahead in [1,2,3,4]:
    for quantile in config.quantiles:
        evaluator_df[f"wQuantileLoss[{quantile}]"] = (evaluator_df[f"QuantileLoss[{quantile}]"] / evaluator_df["abs_target_sum"])
    for item_id in evaluator_df.item_id.unique():   
        df = evaluator_df[evaluator_df.item_id == item_id].copy()
        df["mean_absolute_QuantileLoss"] = np.array([df[f"QuantileLoss[{quantile}]"] for quantile in config.quantiles]).mean()
        df["mean_wQuantileLoss"] = np.array([df[f"wQuantileLoss[{quantile}]"]for quantile in config.quantiles]).mean()
        df["MAE_Coverage"] = np.mean([np.abs(df[f"Coverage[{quantile}]"] - np.array([q]))for q in config.quantiles])    
        evaluator_df.loc[evaluator_df.item_id == item_id, ["mean_absolute_QuantileLoss", "mean_wQuantileLoss", "MAE_Coverage"]] = df[["mean_absolute_QuantileLoss", "mean_wQuantileLoss", "MAE_Coverage"]]
    # produce the average Quantile Loss metric by dividing the mean absolute QL through the number of involved locations per weekahead, which is usually 411 (each district)
    included_locations = [item_id for item_id in evaluator_df.item_id.unique() if "aggregated" not in item_id if "1" in item_id]
    evaluator_df.loc[evaluator_df.item_id.isin([item_id for item_id in evaluator_df.item_id if "aggregate" in item_id]), "mean_WIS"] = evaluator_df.loc[evaluator_df.item_id.isin([item_id for item_id in evaluator_df.item_id if "aggregate" in item_id]),"mean_absolute_QuantileLoss"]/len(included_locations)        
    return evaluator_df