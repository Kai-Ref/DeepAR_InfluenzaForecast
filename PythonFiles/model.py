import pandas as pd
import numpy as np
import mxnet as mx
import matplotlib.pyplot as plt
from datetime import datetime
import gluonts
from gluonts.mx import Trainer, DeepAREstimator
from gluonts.dataset.common import ListDataset
from gluonts.dataset.pandas import PandasDataset
from gluonts.evaluation import make_evaluation_predictions
from gluonts.dataset.split import split, TestData
from gluonts.dataset.util import to_pandas
import os
os.chdir('/home/reffert/DeepAR_InfluenzaForecast')
from PythonFiles.rolling_dataset import generate_rolling_dataset, StepStrategy
from PythonFiles.OverwrittenEvaluator import Evaluator, strict_coverage

def preprocessing(config, df, check_count=False, output_type="PD"):
    """
    This function processes the data into different formats based on the specified output type.
    
    Parameters:
        config (object): Configuration object.
        df (pandas.DataFrame): Input data with a 'date' column and other relevant information.
        check_count (bool, optional): If True, it returns a dictionary with counts of observations for each location.
                                      Defaults to False.
        output_type (str, optional): The type of output format. 
                                     Options are "PD" (PandasDataset), "LD" (ListDataset), or "corrected_df" (pandas DataFrame).
                                     Defaults to "PD".
    
    Returns:
        pandas.DataFrame or gluonts.dataset.pandas.PandasDataset or gluonts.dataset.common.ListDataset:
            The processed data in the specified output format.
        dict or None: If check_count is True, it returns a dictionary with counts of observations for each location.

    """

    # Convert the 'date' column to datetime and set it as the DataFrame index
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    if check_count:
        count_dict = {}
        for location in df.location.unique():
            # Save the number of values within the training & testing time period into the count_dict
            location_df = df.loc[(df['location'] == location) & (df.index > config.train_start_time) & (df.index <= config.test_end_time), :]
            count_dict[location] = location_df.shape[0]
        
        # Print out the distribution of each region with missing values
        print('LK mit weniger als' + str(max(count_dict.values())))
        missing_values_dict = {k: v for k, v in count_dict.items() if v < max(count_dict.values())}
        print(missing_values_dict)
        
        # Return the DataFrame and missing_values_dict if check_count is True
        return df, missing_values_dict
    
    if output_type in ['PD', 'LD', 'corrected_df']:
        # Create a DataFrame Blueprint with evenly spaced time index
        start, end = min(df.index), max(df.index)
        correctly_spaced_index = pd.date_range(start=start, end=end, freq=config.parameters["freq"])
        correctly_spaced_location_df = pd.DataFrame(index=correctly_spaced_index)
        correctly_spaced_df = pd.DataFrame()
        location_list = df.loc[:, 'location'].unique()
        
        for location in location_list:
            # Fill in the missing time steps and locations
            temporary_df = correctly_spaced_location_df.join(df.loc[df.location == location])
            temporary_df['location'] = temporary_df['location'].fillna(location)
            correctly_spaced_df = pd.concat([correctly_spaced_df, temporary_df])
        
        if output_type == "PD":
            # Convert to PandasDataset format
            df = PandasDataset.from_long_dataframe(dataframe=correctly_spaced_df, item_id='location', target="value", freq=config.parameters["freq"])
        elif output_type == "LD":
            # Convert to ListDataset format
            df = ListDataset([{"start": min(correctly_spaced_index), "target": correctly_spaced_df.loc[correctly_spaced_df.location == location, 'value']}
                              for location in location_list], freq=config.parameters["freq"])
        elif output_type == "corrected_df":
            # Return the corrected DataFrame
            return correctly_spaced_df
    
    # Return the processed data
    return df

def train_test_split(config, df, with_features=False):
    """
    This function splits the input DataFrame into train and test sets and formats them into either a PandasDataset 
    or a ListDataset with or without features based on the specified options.
    
    Parameters:
        config (object): Configuration object containing various parameters for preprocessing.
        df (pandas.DataFrame): Input data with a 'date', 'location', 'value', and other relevant columns.
        with_features (bool, optional): If True, the output datasets will include additional features.
                                        Defaults to False.
    
    Returns:
        gluonts.dataset.pandas.PandasDataset or gluonts.dataset.common.ListDataset:
            The train and test datasets in the specified output format.

    """

    locations = list(df.location.unique())
    # Split the DataFrame into train and test sets based on the specified time periods
    train_set = df.loc[(df.index <= config.train_end_time) & (df.index >= config.train_start_time), :]
    test_set = df.loc[(df.index >= config.train_start_time) & (df.index <= config.test_end_time), :]

    # Determine the starting and ending time points for the test set
    start_time = min(test_set.index.difference(train_set.index))
    end_time = max(test_set.index.difference(train_set.index))
    
    if with_features:
        # Format the train and test_set into a PandasDataset with features
        train_set = PandasDataset.from_long_dataframe(dataframe=train_set, item_id='location', target="value", freq=config.parameters["freq"],
                                                      static_feature_columns=["population"] + list(locations),
                                                      feat_dynamic_real=["week"])
        
        test_set = PandasDataset.from_long_dataframe(dataframe=test_set, item_id='location', target="value", freq=config.parameters["freq"],
                                                     static_feature_columns=["population"] + list(locations),
                                                     feat_dynamic_real=["week"])
    else:
        # Format the train and test_set into a PandasDataset without features
        train_set = PandasDataset.from_long_dataframe(dataframe=train_set, item_id='location', target="value", freq=config.parameters["freq"])
        test_set = PandasDataset.from_long_dataframe(dataframe=test_set, item_id='location', target="value", freq=config.parameters["freq"])
    
    # Create the rolling version of the test set with windows of length config.prediction_length and following windows of 1 timestep
    test_set = generate_rolling_dataset(dataset=test_set,
                                        strategy=StepStrategy(prediction_length=config.parameters["prediction_length"], step_size=1),
                                        start_time=pd.Period(start_time, config.parameters["freq"]),
                                        end_time=pd.Period(end_time, config.parameters["freq"])
                                        )
    return train_set, test_set


def model(training_data, test_data, estimator):
    """
    This function fits a given estimator to the training data.
    It then generates forecasts and calculates true values for the test data using the trained estimator.
    
    Parameters:
        training_data (gluonts.dataset.common.Dataset): The training dataset containing the time series data for model training.
        test_data (gluonts.dataset.common.Dataset): The test dataset containing the time series data for evaluation.
        estimator (gluonts.model.estimator.Estimator): The estimator used for training and forecasting.
    
    Returns:
        Tuple[List[gluonts.model.forecast.SampleForecast], List[gluonts.dataset.common.TimeSeriesItem]]:
            A tuple containing lists of forecasts and true values for the test data.

    """
    # Train the estimator on the training data
    predictor = estimator.train(training_data=training_data)
    
    # Generate forecasts and calculate true values for the test data using the trained estimator
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_data,  
        predictor=predictor,  
        num_samples=100,  # Number of samples for probabilistic forecasts
    )
    
    # Unpack Iterator-Objects into lists
    forecasts = list(forecast_it)
    tss = list(ts_it)
    
    return forecasts, tss


def forecast_by_week(config, train_set, test_set, locations, models_dict, seed=None, results_dict=None):
    """
    This function generates forecasts for each model in the models_dict, splits the forecasts by week-ahead,
    evaluates the forecasts using the provided test_set, and computes evaluation metrics for each week-ahead.
    
    Parameters:
        config (Configuration): Configuration object with settings for the forecasting process.
        train_set (gluonts.dataset.common.Dataset): The training dataset containing the time series data.
        test_set (gluonts.dataset.common.Dataset): The test dataset containing the time series data.
        locations (list): List of location IDs used in the datasets.
        models_dict (dict): A dictionary containing the models to be used for forecasting. 
                            The keys represent model names, and the values are GluonTS Estimator instances.
        seed (int or None, optional): Random seed for reproducibility. Defaults to None.
        results_dict (dict or None, optional): A dictionary containing precomputed forecasts and true values 
                                               (forecasts_dict and evaluator_df_dict) from previous runs. 
                                               Defaults to None.
    
    Returns:
        Tuple[dict, dict]:
            A tuple containing two dictionaries:
            - forecasts_dict: A dictionary containing the forecasts split by week-ahead for each model.
            - evaluator_df_dict: A dictionary containing evaluation metrics for each week-ahead for each model.

    """
    if seed is not None:
        mx.random.seed(seed)
        np.random.seed(seed)
    
    # Define dictionaries to store the evaluation results and forecasts
    evaluator_df_dict = {}
    forecasts_dict = {}
    
    # Iterate through the given models and fit them
    for key in models_dict.keys():
        if results_dict is None:
            # If no precomputed results are provided, train the model and generate new forecasts
            forecasts, tss = model(train_set, test_set, models_dict[key])
        else:
            # Otherwise, use the precomputed results for the model
            forecasts = results_dict[f"{key}_forecasts"]
            tss = results_dict[f"{key}_tss"]
        
        # Split the forecasts into their weekly contribution
        split_tss = split_forecasts_by_week(config, forecasts, tss, locations, 4, equal_time_frame=True)[1]
        forecast_dict ={1 : split_forecasts_by_week(config, forecasts, tss, locations, 1, equal_time_frame=True)[0],
                        2 : split_forecasts_by_week(config, forecasts, tss, locations, 2, equal_time_frame=True)[0],
                        3 : split_forecasts_by_week(config, forecasts, tss, locations, 3, equal_time_frame=True)[0],
                        4 : split_forecasts_by_week(config, forecasts, tss, locations, 4, equal_time_frame=True)[0]}
        
        # Evaluation with the quantiles of the configuration
        evaluator = Evaluator(quantiles=config.quantiles)
        evaluator_df = pd.DataFrame()
        
        # Iterate over the 4 different week-aheads
        for forecast in forecast_dict.values():
            agg_metrics, item_metrics = evaluator(split_tss, forecast)
            d = {key for key in forecast_dict if forecast_dict[key] == forecast}
            for location in locations[:]:
                # Rename location id to differentiate between the week-ahead predictions and concatenate
                item_metrics.loc[item_metrics.item_id == f"{location}", "item_id"] = f"{location} {d}"
                evaluator_df = pd.concat([evaluator_df, item_metrics[item_metrics.item_id == f"{location} {d}"]])
            agg_metrics["item_id"] = f"aggregated {d}"
            evaluator_df = pd.concat([evaluator_df, pd.DataFrame(agg_metrics, index=[0])])
        
        # Compute the average Quantile Loss metric by dividing the mean absolute QL by the number of involved locations per week-ahead
        included_locations = [item_id for item_id in evaluator_df.item_id.unique() if "aggregated" not in item_id if "1" in item_id]
        evaluator_df.loc[evaluator_df.item_id.isin([item_id for item_id in evaluator_df.item_id if "aggregate" in item_id]), "mean_WIS"] = evaluator_df.loc[evaluator_df.item_id.isin([item_id for item_id in evaluator_df.item_id if "aggregate" in item_id]),"mean_absolute_QuantileLoss"]/len(included_locations)
        
        evaluator_df_dict[key] = evaluator_df
        forecasts_dict[key] = forecast_dict
    
    return forecasts_dict, evaluator_df_dict


def split_forecasts_by_week(config, forecasts, tss, locations, week, equal_time_frame=False):
    """
    Splits up a list of forecasts into forecasts for a given [week]-week ahead.
    
    Parameters:
        config (Configuration): Configuration object with settings for the forecasting process.
        forecasts (List[gluonts.model.forecast.SampleForecast]): List of forecasts generated by the model.
        tss (List[gluonts.model.forecast.SampleForecast]): List of true values for each time series.
        locations (list): List of location IDs used in the forecasts and true values.
        week (int): The number of weeks ahead for which to split the forecasts.
        equal_time_frame (bool, optional): If True, ensure equal time frames for each week ahead by aligning the starting date.
                                          If False, use the actual starting date for each week ahead.
                                          Defaults to False.

    Returns:
        Tuple[List[gluonts.model.forecast.SampleForecast], List[gluonts.model.forecast.SampleForecast]]:
            A tuple containing two lists:
            - week_ahead_forecasts: List of SampleForecast objects for the given week ahead for each location.
            - split_tss: List of SampleForecast objects containing true values for each location.

    """
    # First determine the amount of forecasted windows per location, so we can iterate through and specify the correct index later
    windows_per_location = int(len(forecasts) / len(locations))
    week_ahead_forecasts = []
    split_tss = []
    
    for location in locations:
        if equal_time_frame:
            # Choose an index that sets the starting date of different week ahead within 1 and 4 week ahead to equal each other
            first_time_point_of_location = windows_per_location + windows_per_location * locations.index(location) - ((-week) % 4) - 1
            for_loop_end = first_time_point_of_location - windows_per_location + ((-week) % 4) + week
        else:
            # Define the index of the time-wise first forecast point
            first_time_point_of_location = windows_per_location + windows_per_location * locations.index(location) - 1
            for_loop_end = first_time_point_of_location - windows_per_location
        
        start_date_list = []
        # Append the True underlying values (of the complete test window) to split_tss for the current location
        split_tss.append(tss[windows_per_location * locations.index(location)])
        
        # Add the time-wise first (and index-wise last) [num_samples]-arrays of the corresponding week to [weekly_samples_array]
        weekly_samples_array = forecasts[first_time_point_of_location].samples[:, (week - 1):week]
        
        for k in range(first_time_point_of_location - 1, for_loop_end, -1):
            # Reverse iterate through the windows of each location, as the time-wise first forecasts are last in the forecast-list
            # and concatenate the array with the corresponding values of each [week]-week ahead forecast
            weekly_samples_array = np.concatenate((weekly_samples_array, forecasts[k].samples[:, (week - 1):week]), axis=1)
        
        # Save the correct starting time, determined by the first [start_date] of the location and the [week] parameter
        start_date = pd.date_range(
            start=forecasts[first_time_point_of_location].start_date.to_timestamp(),
            periods=week,
            freq=config.parameters["freq"],
        )[-1]
        
        # Append the filtered [weekly_samples_array]-array and the correct [start_date] as a SampleForecast-Object for each location
        week_ahead_forecasts.append(
            gluonts.model.forecast.SampleForecast(
                info=forecasts[first_time_point_of_location].info,
                item_id=forecasts[first_time_point_of_location].item_id,
                samples=weekly_samples_array,
                start_date=pd.Period(start_date, freq=config.parameters["freq"]),
            )
        )
    
    return week_ahead_forecasts, split_tss
    

    
def make_one_ts_prediction(config, df, location="LK Bad Dürkheim"):
    """
    Trains a model for the univariate time series of the given [location] and plots the resulting prediction(s).

    Parameters:
        config (Configuration): Configuration object with settings for the forecasting process.
        df (pd.DataFrame): The input DataFrame containing the time series data.
        location (str, optional): The location for which to make the prediction. Defaults to "LK Bad Dürkheim".

    Returns:
        Tuple[List[gluonts.model.forecast.SampleForecast], List[gluonts.model.forecast.SampleForecast]]:
            A tuple containing two lists:
            - forecasts: List of SampleForecast objects representing the model's predictions.
            - tss: List of SampleForecast objects containing the true values for the given location.

    """
    # Process the df into a uniformly spaced df
    one_ts_df = df.loc[df.location == location, ["value", 'location', 'date']]
    one_ts_df = preprocessing(config, one_ts_df, check_count=False, output_type="corrected_df")
    
    # Separate the intervals for training and testing
    train_set = one_ts_df.loc[(one_ts_df.index <= config.train_end_time) & (one_ts_df.index >= config.train_start_time), :]
    test_set = one_ts_df.loc[(one_ts_df.index >= config.train_start_time) & (one_ts_df.index <= config.test_end_time), :]
    
    # Select the correct dates for splitting within the test data for each window
    window_dates = []
    for window in range(1, config.windows):
        unique_weeks = test_set.index.unique()
        selected_split_week = unique_weeks[-window * config.parameters["prediction_length"]: -window * config.parameters["prediction_length"] + 1]
        window_dates.append(datetime(selected_split_week.year[0], selected_split_week.month[0], selected_split_week.day[0]))
    # Also add the last date available
    window_dates.append(config.test_end_time)
    window_dates.sort()
    
    # Define the list of DataFrames for each testing window
    test_windows = [test_set.loc[test_set.index < window_date, :] for window_date in window_dates]
    
    # Format the train and test_set into a PandasDataset
    train_set = PandasDataset.from_long_dataframe(dataframe=train_set, item_id='location', target="value", freq=config.parameters["freq"])
    test_set = PandasDataset(test_windows, target="value", freq=config.parameters["freq"])
    
    # Train and evaluate the model
    forecasts, tss = model(train_set, test_set, config.deeparestimator)
    
    # Plot the forecasts
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    plt.title(f'{location}')
    
    # First plot the time series as a whole (x-axis: Date, y-axis: influenza-values)
    plt.plot(
        one_ts_df.loc[(one_ts_df['location'] == location) &
                      (one_ts_df.index <= config.test_end_time) &
                      (one_ts_df.index >= config.train_start_time)].index,
        one_ts_df.loc[(one_ts_df['location'] == location) &
                      (one_ts_df.index <= config.test_end_time) &
                      (one_ts_df.index >= config.train_start_time), 'value']
    )
    plt.grid(which="both")
    
    # Define the colors to use for each different window
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

def process_R_results(config, results_df, influenza_df, validation=False):
    '''
    Process the results_df by adding a date column and the corresponding true values.

    Parameters:
        config (Configuration): Configuration object with settings for the forecasting process.
        results_df (pd.DataFrame): DataFrame containing the forecast results from R.
        influenza_df (pd.DataFrame): DataFrame containing the true influenza values.
        validation (bool, optional): Whether the processing is for validation or not. Defaults to False.

    Returns:
        pd.DataFrame: Processed DataFrame with date column and corresponding true values.

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
    if validation:
        R_start = datetime(2018, 10, 7)
        window_length = 97
    else:
        R_start = datetime(2016, 10, 2) # first forecasted point
        window_length = 98
    R_end = datetime(2018, 10, 21)
    # determine the daterange of the forecast period
    daterange = pd.date_range(start=R_start, periods=len(results_df.Time.unique()), freq=config.parameters["freq"])
    influenza_df["date"] = pd.to_datetime(influenza_df["date"])
    
    # Iterate over the zipped pairs of time and date e.g., [(821,2016-09-25), ..., (929,2018-10-21)]  
    for i in zip(df.Time.unique(), daterange): 
        df.loc[df.Time == i[0], "date"] = i[1]
        
    df["date"] = pd.to_datetime(df["date"])
    
    # rename the location column so it matches with the influenza location columnname
    df.rename(columns={'Location': 'location'}, inplace=True)
    df = df.merge(influenza_df[["date", "value", "location"]], on=["date", "location"]) 
    df.rename(columns={'value': 'true_value'}, inplace=True)
    
    # truncate the predictions to lie within the same date range
    start_date = df.loc[df["WeekAhead"] == 4, "date"].min()
    end_date = df.loc[df["WeekAhead"] == 1, "date"].max()
    # the gluonts models only have a one week ahead forecast up to 98 data points
    # therefore set the start and the length to equal the gluonts model period
    end_date = pd.date_range(start=start_date, freq=config.parameters["freq"], periods=window_length)[-1:][0]
    df = df.loc[(df["date"] >= start_date) & (df["date"] <= end_date)]
    
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
    """
    Evaluate the forecasts generated by R models.

    Parameters:
        config (Configuration): Configuration object with settings for the forecasting process.
        df_dict (dict): A dictionary containing forecast DataFrames for each week ahead.
        locations (list): List of locations (districts or regions) in the dataset.
        processed_df (pd.DataFrame): DataFrame containing processed R results with true values.

    Returns:
        pd.DataFrame: DataFrame containing the evaluation metrics for the forecasts.

    """
    evaluator_df = pd.DataFrame({"item_id": ["aggregated {" + f"{week}" + "}" for week in [1, 2, 3, 4]] + 
                                 [f"{location} " + "{" + f"{week}" + "}" for location in locations for week in [1,2,3,4]]})
    
    # determine the metrics for individual series
    for week_ahead in [1, 2, 3, 4]:
        print(f"Evaluating {week_ahead}/4 -- {datetime.now()}")
        df = df_dict[week_ahead].copy()
        for quantile in config.quantiles:
            for location in locations:
                # calculate the Quantile Loss and the Coverage for each region
                evaluator_df.loc[evaluator_df.item_id == str(f"{location} " + "{" + f"{week_ahead}" + "}"),
                                 f"QuantileLoss[{quantile}]"] = quantile_loss(df.loc[df["location"]==location, "true_value"],
                                                                               df.loc[df["location"]==location, f"{quantile}"],
                                                                               quantile)

                evaluator_df.loc[evaluator_df.item_id == str(f"{location} " + "{" + f"{week_ahead}" + "}"),
                                 f"Coverage[{quantile}]"] = coverage(df.loc[df["location"]==location, "true_value"],
                                                                     df.loc[df["location"]==location, f"{quantile}"])

                evaluator_df.loc[evaluator_df.item_id == str(f"{location} " + "{" + f"{week_ahead}" + "}"),
                                 f"StrictCoverage[{quantile}]"] = strict_coverage(df.loc[df["location"]==location, "true_value"],
                                                                                 df.loc[df["location"]==location, f"{quantile}"])

            evaluator_df.loc[evaluator_df.item_id == str("aggregated {" + f"{week_ahead}" + "}"),
                             f"QuantileLoss[{quantile}]"] = quantile_loss(df["true_value"], df[f"{quantile}"], quantile)

            evaluator_df.loc[evaluator_df.item_id == str("aggregated {" + f"{week_ahead}" + "}"),
                             f"Coverage[{quantile}]"] = coverage(df["true_value"], df[f"{quantile}"])

            evaluator_df.loc[evaluator_df.item_id == str("aggregated {" + f"{week_ahead}" + "}"),
                             f"StrictCoverage[{quantile}]"] = strict_coverage(df["true_value"], df[f"{quantile}"])

    # add the aggregate metrics
    evaluator_df["abs_target_sum"] = abs_target_sum(processed_df["true_value"])

    for quantile in config.quantiles:
        evaluator_df[f"wQuantileLoss[{quantile}]"] = (evaluator_df[f"QuantileLoss[{quantile}]"] / evaluator_df["abs_target_sum"])

    for item_id in evaluator_df.item_id.unique():   
        df = evaluator_df[evaluator_df.item_id == item_id].copy()
        df["mean_absolute_QuantileLoss"] = np.array([df[f"QuantileLoss[{quantile}]"] for quantile in config.quantiles]).mean()
        df["mean_wQuantileLoss"] = np.array([df[f"wQuantileLoss[{quantile}]"]for quantile in config.quantiles]).mean()
        df["MAE_Coverage"] = np.mean([np.abs(df[f"Coverage[{quantile}]"] - np.array([q])) for q in config.quantiles])   
        df["MAE_StrictCoverage"] = np.mean([np.abs(df[f"StrictCoverage[{quantile}]"] - np.array([q])) for q in config.quantiles]) 
        evaluator_df.loc[evaluator_df.item_id == item_id, ["mean_absolute_QuantileLoss", "mean_wQuantileLoss", "MAE_Coverage", "MAE_StrictCoverage"]] = df[["mean_absolute_QuantileLoss", "mean_wQuantileLoss", "MAE_Coverage", "MAE_StrictCoverage"]]
    
    # produce the average Quantile Loss metric by dividing the mean absolute QL through the number of involved locations per weekahead, which is usually 411 (each district)
    included_locations = [item_id for item_id in evaluator_df.item_id.unique() if "aggregated" not in item_id if "1" in item_id]
    evaluator_df.loc[evaluator_df.item_id.isin([item_id for item_id in evaluator_df.item_id if "aggregate" in item_id]), 
                     "mean_WIS"] = evaluator_df.loc[evaluator_df.item_id.isin([item_id for item_id in evaluator_df.item_id if "aggregate" in item_id]), "mean_absolute_QuantileLoss"] / len(included_locations)
        
    return evaluator_df