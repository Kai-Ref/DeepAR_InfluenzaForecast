import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import itertools
#import gluonts
#from gluonts.mx import Trainer, DeepAREstimator
#from gluonts.dataset.common import ListDataset
#from gluonts.dataset.pandas import PandasDataset
#from gluonts.evaluation import make_evaluation_predictions, Evaluator
#from gluonts.dataset.split import split, TestData
#from gluonts.dataset.util import to_pandas
#from gluonts.dataset.rolling_dataset import generate_rolling_dataset,StepStrategy
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
                                   (corrected_df.index >= plot_start_time),'value'], c=config.colors[0])
            plt.grid(which="both")
            # select the right week-ahead forecast entry for a set location
            forecast_entry = forecast_dict[list(forecast_dict.keys())[week_ahead-1]][locations.index(location)]
            prediction_intervals = (50.0, 90.0)
            forecast_entry.plot(prediction_intervals=prediction_intervals, color=config.colors[2])
            plt.grid(which="both")
            plt.show()


def plot_coverage(config, evaluator_df_dict, locations=None):
    """
    Given a dictionary, where the values consist of evaluation_df's, this function is going to create plots of the 4 different week-ahead coverages.  
    However, the weekly performances have to be under the "item_id" with f.e. "aggregated {1}" for the 1 week-ahead metrics.
    """
    week_coverage_dict = {}
    coverage_columns = [col for col in evaluator_df_dict[list(evaluator_df_dict.keys())[0]].columns if "Coverage" in col]
    if "MAE_Coverage" in coverage_columns:
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
            axs[plotnumber].plot([0.0, 1.0], [0.0, 1.0], c= config.colors[0])
            axs[plotnumber].scatter(config.quantiles, evaluator_df_dict[key].loc[evaluator_df_dict[key].item_id.isin(["aggregated {" + f"{week}" + "}"]), coverage_columns], c=config.colors[0])
            axs[plotnumber].plot(config.quantiles, evaluator_df_dict[key].loc[evaluator_df_dict[key].item_id.isin(["aggregated {" + f"{week}" + "}"]), coverage_columns].T, label=f"{key}", c=config.colors[list(evaluator_df_dict.keys()).index(key)+1])
            axs[plotnumber].title.set_text(f"{week}-Week Ahead Coverage")
            axs[plotnumber].legend()
            

def plot_model_results_by_hp(config, model_results_by_hp, hp_search_space, number_of_plots=30, col="mean_WIS",figsize=(16, 9), overall_df=None, sort_by="model_WIS_mean", plottype="unordered", plot = "bp"):
    '''
    Creates boxplots of different combinations. 
    
    col: "mean_WIS", "time_this_iter_s"
    sort_by: "mean_WIS", "time_this_iter_s", "model_WIS_mean", "model_WIS_variance", "model_WIS_sd", "model_WIS_median", "model_time_mean", "model_time_variance", "model_time_sd",\
             "model_time_median"(, "shape") 
    plottype: "unordered"(not ordered), "best" or "worst"
    '''
    number_of_plots = min(number_of_plots, len(model_results_by_hp.keys()))
    if (type(overall_df) != type(None)) & (plottype != "unordered"):
        # create a sorted_df, from which the best/worst combinations can be plotted
        column_names = ["config/"+str(hyperparameter) for hyperparameter in hp_search_space.keys() if not "cardinality" in hyperparameter]
        sorted_df = overall_df.sort_values(sort_by)[[col for col in column_names] + [sort_by]].drop_duplicates()
        sorted_hps=[*zip(*map(sorted_df[[col for col in column_names]].get, sorted_df[[col for col in column_names]]))]
        sorted_hps = [str(hp) for hp in sorted_hps]
        #sorted_hps = np.unique(sorted_hps).tolist() -> may be needed if we filter by mean_WIS (individual modelrun filtering) and duplicates occur
        if plottype == "best":
            hp_configurations = sorted_hps
        if plottype == "worst":
            hp_configurations = sorted_hps
            hp_configurations.reverse()
    if plottype == "unordered":
        hp_configurations = list(model_results_by_hp.keys())
    dfs, labels, lengths = [], [], [] 
    for key in hp_configurations[:number_of_plots]:
        dfs.append(model_results_by_hp[key])
        lengths.append(len(model_results_by_hp[key]))
        labels.append(key)
    lengths = np.unique(lengths).tolist()
    # Create a figure and axis object
    fig, ax = plt.subplots(figsize=figsize)

    # Create a list of the positions for each boxplot
    pos = range(1, len(dfs) * 2, 2)

    if plot == "scatter":
        # Loop through each dataframe and plot a boxplot
        for i, model_df in enumerate(dfs):
            for value in model_df[col]:
                ax.scatter(pos[i], value, marker =".", c=config.colors[0])#, fc="None", ec="black")
    else:
        # Loop through each dataframe and plot a boxplot
        for i, model_df in enumerate(dfs):
            ax.boxplot(model_df[col], positions=[pos[i]])

    # Set the x-axis ticks and tick labels
    ax.set_xticks(pos)
    ax.set_xticklabels(labels)

    # Set the y-axis label
    ax.set_ylabel(f'{col}')

    # Set the title
    ax.set_title(f'Boxplots of {plottype} {number_of_plots} models based on {sort_by} and {lengths} runs per combination.')

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

def hp_color_plot(config, overall_df, hp_search_space, x_axis="model_WIS_mean", y_axis="model_time_mean"):
    added_cols =["model_WIS_mean", "model_WIS_variance", "model_WIS_sd", "model_WIS_median",
                 "model_time_mean", "model_time_variance", "model_time_sd","model_time_median", "shape"] 

    unique_df = overall_df[added_cols+[col for col in overall_df.columns if ("config" in col)&("cardinality" not in col)]].drop_duplicates()
    without_card =[key for key in hp_search_space.keys() if "cardinality" not in key]
    nrows = int(len(without_card)/2) + int(len(without_card)%2)
    fig, axs = plt.subplots(nrows=nrows, ncols=2, figsize=(16, 16), sharey=True)
    fig.tight_layout(pad=2.9)
    plotnumber = [0, 0]
    for key in without_card:
        column = "config/"+key
        if list(without_card).index(key)%2 == 1:
            plotnumber[1] = 1
        else:
            if list(without_card).index(key) > 1:
                plotnumber[0] += 1
            plotnumber[1] = 0
        values = unique_df[column].unique().tolist()
        for value in values:
            print_df = unique_df.loc[unique_df[column]==value,:]
            axs[tuple(plotnumber)].scatter(print_df[x_axis],print_df[y_axis], c=config.colors[values.index(value)], label=value)
        axs[tuple(plotnumber)].legend()
        axs[tuple(plotnumber)].set_title(key)
        axs[tuple(plotnumber)].set_ylabel(y_axis)
        axs[tuple(plotnumber)].set_xlabel(x_axis)
    plt.show()
    
    
def plot_forecast_entry(config, fe, show_mean=False,ax=plt, prediction_intervals=(50.0, 90.0), meancolor=None, mediancolor=None, fillcolor=None, axis=False):
    '''
    Overwritten version of the forecast_entry.plot() method.
    Includes customizable colors and axis.
    '''
    if meancolor == None:
        meancolor = config.colors[1]
    if mediancolor == None:
        mediancolor = config.colors[0] 
    if fillcolor == None:
        fillcolor = config.colors[4]
    
    # Determining the Prediction Intervals alpha levels for plotting
    for c in prediction_intervals:
        assert 0.0 <= c <= 100.0

    ps = [50.0] + [
        50.0 + f * c / 2.0
        for c in prediction_intervals
        for f in [-1.0, +1.0]
    ]
    percentiles_sorted = sorted(set(ps))

    def alpha_for_percentile(p):
        return (p / 100.0) ** 0.3

    ps_data = [fe.quantile(p / 100.0) for p in percentiles_sorted]
    i_p50 = len(percentiles_sorted) // 2
    # Plotting the Median of the forecast entry
    p50_data = ps_data[i_p50]
    p50_series = pd.Series(data=p50_data, index=fe.index)
    if axis == True:
        plt.sca(ax)
    p50_series.plot(color=mediancolor, ls="-", label="median")
    
    # Plotting the mean of the forecast entry
    if show_mean:
        mean_data = np.mean(fe._sorted_samples, axis=0)
        pd.Series(data=mean_data, index=fe.index).plot(
            color=meancolor,
            ls=":",
            label=f"mean",
        )
    # Plotting the 
    for i in range(len(percentiles_sorted) // 2):
        ptile = percentiles_sorted[i]
        alpha = alpha_for_percentile(ptile)
        plt.fill_between(
            fe.index,
            ps_data[i],
            ps_data[-i - 1],
            facecolor=fillcolor,
            alpha=alpha,
            interpolate=True
        )
        # Hack to create labels for the error intervals. Doesn't actually
        # plot anything, because we only pass a single data point
        pd.Series(data=p50_data[:1], index=fe.index[:1]).plot(
            color=config.colors[2],
            alpha=alpha,
            linewidth=10,
            label=f"{100 - ptile * 2}%",
        )