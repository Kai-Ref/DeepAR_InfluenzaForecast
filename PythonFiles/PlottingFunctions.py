import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import itertools

def print_forecasts_by_week(config, df, forecasts_dict, locations, week_ahead_list, plot_begin_at_trainstart=False, savepath=None, filename=None):
    '''
    @Overwritten 
    Prints out plots for the given week-Ahead forecasts of given locations. It needs the initial corrected dataframe, as well as the forecast_dict
    that contains the different week-ahead forecasts.
    The start of the plot time axis, can be set to the training start time (TRUE) or the testing start time (FALSE).
    '''
    if (savepath != None)&(filename!=None):
        import os
    all_locations=list(df.location.unique())
    for key in list(forecasts_dict.keys()):
        forecast_dict = forecasts_dict[key] 
        for week_ahead in week_ahead_list:
            nrows = int(len(locations)/2) + int(len(locations)%2)
            fig, ax = plt.subplots(nrows, 2, figsize=(16, 9))
            fig.tight_layout(pad=2.9)
            plotnumber = [0, 0]
            for location in locations:
                if list(locations).index(location)%2 == 1:
                    plotnumber[1] = 1
                else:
                    if list(locations).index(location) > 1:
                        plotnumber[0] += 1
                    plotnumber[1] = 0
                #plot the forecasts
                ax[tuple(plotnumber)].set_title(f'{location} - {week_ahead} Week Ahead Forecast')
                plt.xticks(rotation=0)
                # determine the beginning of the time series
                if plot_begin_at_trainstart == True:
                    plot_start_time = config.train_start_time
                elif type(plot_begin_at_trainstart)==type(datetime(2016,1,1,1,1,1)):
                    plot_start_time = plot_begin_at_trainstart
                else:
                    plot_start_time = config.train_end_time
                #first plot the time series as a whole (x-axis: Date, y-axis: influenza-values)
                ax[tuple(plotnumber)].plot((df.loc[(df['location'] == location) &
                                        (df.index <= config.test_end_time) &
                                        (df.index >= plot_start_time)].index),
                         df.loc[(df['location'] == location) &
                                       (df.index <= config.test_end_time) &
                                       (df.index >= plot_start_time),'value'], c=config.colors[0])
                #ax[tuple(plotnumber)].grid(which="both")
                # run the overwritten version of the forecast_entry.plot() function to plot to a specific axis and change colors
                if key == "hhh4":
                    # Here forecast_entry is a df
                    forecast_entry = forecast_dict[list(forecast_dict.keys())[week_ahead-1]].copy()
                    plot_forecast_entry(config, forecast_entry.loc[forecast_entry["location"]==location], show_mean=False, ax=ax[tuple(plotnumber)], mediancolor=config.colors[1], fillcolor=config.colors[2], axis=True,\
                                        prediction_intervals=(50.0, 80.0), R_entry=True)
                else:
                    # select the right week-ahead forecast entry for a set location
                    forecast_entry = forecast_dict[list(forecast_dict.keys())[week_ahead-1]][all_locations.index(location)]
                    plot_forecast_entry(config, forecast_entry, show_mean=False,ax=ax[tuple(plotnumber)], mediancolor=config.colors[1],fillcolor=config.colors[2], axis=True, prediction_intervals=(50.0, 80.0))
                    #forecast_entry.plot(prediction_intervals=prediction_intervals, color=config.colors[0])
                plt.xticks([config.train_end_time, config.test_end_time], rotation=0, ha="center")
                plt.legend(loc="upper left")
            if (savepath != None)&(filename!=None):
                os.chdir(savepath)
                plt.savefig(f"{filename}{key}_{week_ahead}_WA.png")
                os.chdir('/home/reffert/DeepAR_InfluenzaForecast')
            plt.show()
                
def side_by_side_print_forecasts_by_week(config, df, forecasts_dict, locations, week_ahead_list, plot_begin_at_trainstart=False, savepath=None, filename=None, figsize=(25, 20)):
    '''
    @Overwritten 
    Prints out plots for the given week-Ahead forecasts of given locations. It needs the initial corrected dataframe, as well as the forecast_dict
    that contains the different week-ahead forecasts.
    The start of the plot time axis, can be set to the training start time (TRUE) or the testing start time (FALSE).
    '''
    if (savepath != None)&(filename!=None):
        import os
    all_locations=list(df.location.unique())
    for week_ahead in week_ahead_list:
        fig, ax = plt.subplots(len(locations), len(list(forecasts_dict.keys())), figsize=figsize)
        fig.tight_layout(pad=2.9)
        for location in locations:
            for key in list(forecasts_dict.keys()):
                row = locations.index(location)
                column = list(forecasts_dict.keys()).index(key)
                plotnumber = [row, column]
                plt.sca(ax[tuple(plotnumber)])
                plt.xticks(rotation=0)
                plt.title(f'{key} - {location} - {week_ahead} Week Ahead Forecast')
                
                if plot_begin_at_trainstart == True:
                    plot_start_time = config.train_start_time
                elif type(plot_begin_at_trainstart)==type(datetime(2016,1,1,1,1,1)):
                    plot_start_time = plot_begin_at_trainstart
                else:
                    plot_start_time = config.train_end_time
                    
                #first plot the time series as a whole (x-axis: Date, y-axis: influenza-values)
                plt.plot((df.loc[(df['location'] == location) &
                                        (df.index <= config.test_end_time) &
                                        (df.index >= plot_start_time)].index),
                         df.loc[(df['location'] == location) &
                                       (df.index <= config.test_end_time) &
                                       (df.index >= plot_start_time),'value'], c=config.colors[0])
                #ax[tuple(plotnumber)].grid(which="both")
                # run the overwritten version of the forecast_entry.plot() function to plot to a specific axis and change colors
                forecast_dict = forecasts_dict[key] 
                if key == "hhh4":
                    # Here forecast_entry is a df
                    forecast_entry = forecast_dict[list(forecast_dict.keys())[week_ahead-1]].copy()
                    plot_forecast_entry(config, forecast_entry.loc[forecast_entry["location"]==location], show_mean=False, ax=ax[tuple(plotnumber)], mediancolor=config.colors[1], fillcolor=config.colors[2], axis=True,\
                                        prediction_intervals=(50.0, 95.0), R_entry=True)
                else:
                    # select the right week-ahead forecast entry for a set location
                    forecast_entry = forecast_dict[list(forecast_dict.keys())[week_ahead-1]][all_locations.index(location)]
                    plot_forecast_entry(config, forecast_entry, show_mean=False,ax=ax[tuple(plotnumber)], mediancolor=config.colors[1],fillcolor=config.colors[2], axis=True, prediction_intervals=(50.0, 95.0))
                    #forecast_entry.plot(prediction_intervals=prediction_intervals, color=config.colors[0])
                plt.xticks([config.train_end_time, config.test_end_time], rotation=0, ha="center")
                plt.legend(loc="upper left")
        if (savepath != None)&(filename!=None):
            os.chdir(savepath)
            plt.savefig(f"{filename}Combined_{week_ahead}_WA.png")
            os.chdir('/home/reffert/DeepAR_InfluenzaForecast')
        plt.show()

def plot_coverage(config, evaluator_df_dict, colors=None, strict=False):
    """
    Given a dictionary, where the values consist of evaluation_df's, this function is going to create plots of the 4 different week-ahead coverages.  
    However, the weekly performances have to be under the "item_id" with f.e. "aggregated {1}" for the 1 week-ahead metrics.
    """
    week_coverage_dict = {}
    coverage_columns = [col for col in evaluator_df_dict[list(evaluator_df_dict.keys())[0]].columns if ("Coverage" in col)&("Strict" not in col)]
    if strict:
        week_strict_coverage_dict = {}
        strict_coverage_columns = [col for col in evaluator_df_dict[list(evaluator_df_dict.keys())[0]].columns if "StrictCoverage" in col]
        if "MAE_StrictCoverage" in strict_coverage_columns:
            strict_coverage_columns.remove("MAE_StrictCoverage")
    if colors==None:
        colors = config.colors
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
        axs[plotnumber].plot([0.0, 1.0], [0.0, 1.0], c=config.colors[0])
        for key in evaluator_df_dict.keys():
            axs[plotnumber].scatter(config.quantiles, evaluator_df_dict[key].loc[evaluator_df_dict[key].item_id.isin(["aggregated {" + f"{week}" + "}"]), coverage_columns], c=colors[list(evaluator_df_dict.keys()).index(key)+1])
            week_coverage_dict[week] = evaluator_df_dict[key].loc[evaluator_df_dict[key].item_id.isin(["aggregated {"+ f"{week}" + "}"]), coverage_columns]
            axs[plotnumber].plot(config.quantiles, evaluator_df_dict[key].loc[evaluator_df_dict[key].item_id.isin(["aggregated {" + f"{week}" + "}"]), coverage_columns].T, label=f"{key}", c=colors[list(evaluator_df_dict.keys()).index(key)+1])
            if strict:
                axs[plotnumber].scatter(config.quantiles, evaluator_df_dict[key].loc[evaluator_df_dict[key].item_id.isin(["aggregated {" + f"{week}" + "}"]), strict_coverage_columns], c=colors[list(evaluator_df_dict.keys()).index(key)+1])
                week_strict_coverage_dict[week] = evaluator_df_dict[key].loc[evaluator_df_dict[key].item_id.isin(["aggregated {"+ f"{week}" + "}"]), strict_coverage_columns]
                axs[plotnumber].plot(config.quantiles, evaluator_df_dict[key].loc[evaluator_df_dict[key].item_id.isin(["aggregated {" + f"{week}" + "}"]), strict_coverage_columns].T, c=colors[list(evaluator_df_dict.keys()).index(key)+1])
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
            hp_plots[key] = {"cols" : [f"{i} {key}" for i in search_grid], "df": [results_df.loc[results_df[f'config/{key}'].apply(lambda x: x == tuple(str(i))), col] if (isinstance(i, list)) else results_df.loc[results_df[f'config/{key}']==i, col] for i in search_grid]} # the if isinstance is necessary for the num_hidden_layers of the FNN
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
    
    
def plot_forecast_entry(config, fe, show_mean=False, ax=plt, prediction_intervals=(50.0, 80.0), meancolor=None, mediancolor=None, fillcolor=None, axis=False, R_entry=False):
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

    if axis == True:
        plt.sca(ax)
    if R_entry:
        ps_data = [(p / 100.0) for p in percentiles_sorted]
        plt.plot(fe.date, fe["0.5"], c=mediancolor, label=f"median")
    else:
        ps_data = [fe.quantile(p / 100.0) for p in percentiles_sorted]
        i_p50 = len(percentiles_sorted) // 2
        # Plotting the Median of the forecast entry
        p50_data = ps_data[i_p50]
        p50_series = pd.Series(data=p50_data, index=fe.index)
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
        if R_entry:
            plt.fill_between(fe.date, fe[f"{ps_data[i]}"],fe[f"{ps_data[-i - 1]}"],
                facecolor=fillcolor,
                alpha=alpha,
                interpolate=True, 
                label=f"{100 - ptile * 2}%"
            )
            # Hack to create labels for the error intervals. Doesn't actually
            # plot anything, because we only pass a single data point
            #pd.Series(data=fe["0.5"].tolist()[:1], index=fe["date"].tolist()[:1]).plot(
             #   color=fillcolor,
              #  alpha=alpha,
               # linewidth=10,
                #label=f"{100 - ptile * 2}%",
           # )
        else:
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