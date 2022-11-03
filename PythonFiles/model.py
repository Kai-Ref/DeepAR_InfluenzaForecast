import matplotlib
import pandas as pd
from gluonts.mx import DeepAREstimator
from PythonFiles.Configuration import Configuration
from gluonts.dataset.common import ListDataset
from gluonts.dataset.pandas import PandasDataset
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.dataset.split import split
import gluonts
from gluonts.dataset.util import to_pandas
import matplotlib.pyplot as plt

from gluonts.transform import (
    AddAgeFeature,
    AddObservedValuesIndicator,
    Chain,
    ExpectedNumInstanceSampler,
    InstanceSplitter,
    SetFieldIfNotPresent,
)

def create_transformation(config):
    return Chain(
        [
            AddObservedValuesIndicator(
                target_field=FieldName.TARGET,
                output_field=FieldName.OBSERVED_VALUES,
            ),
            AddAgeFeature(
                target_field=FieldName.TARGET,
                output_field=FieldName.FEAT_AGE,
                pred_length=config.prediction_length,
                log_scale=True,
            ),
            InstanceSplitter(
                target_field=FieldName.TARGET,
                is_pad_field=FieldName.IS_PAD,
                start_field=FieldName.START,
                forecast_start_field=FieldName.FORECAST_START,
                instance_sampler=ExpectedNumInstanceSampler(
                    num_instances=1,
                    min_future=config.prediction_length,
                ),
                past_length=config.context_length,
                future_length=config.prediction_length,
                time_series_fields=[
                    FieldName.FEAT_AGE,
                    FieldName.FEAT_DYNAMIC_REAL,
                    FieldName.OBSERVED_VALUES,
                ],
            ),
        ]
    )

#IMPLEMENTED FROM Splitting
def highlight_entry(entry, color):
    start = entry["start"]
    end = entry["start"] + len(entry["target"])
    plt.axvspan(start, end, facecolor=color, alpha=0.2)


def plot_dataset_splitting(original_dataset, training_dataset, test_pairs):
    for original_entry, train_entry in zip(original_dataset, training_dataset):
        to_pandas(original_entry).plot()
        highlight_entry(train_entry, "red")
        plt.legend(["sub dataset", "training dataset"], loc="upper left")
        plt.show()

    for original_entry in original_dataset:
        for test_input, test_label in test_pairs:
            to_pandas(original_entry).plot()
            highlight_entry(test_input, "green")
            highlight_entry(test_label, "blue")
            plt.legend(["sub dataset", "test input", "test label"], loc="upper left")
            plt.show()


def model(config,training_data,test_data):
    """
    This function defines the estimator based on the attributes set in Configuration.py. Then this estimator is fit with
    the given training_data and the forecasts, aswell as the true values for the test_data are calculated via the 
    make_evaluation_predictions function from gluonts. 
    """
    #defining the estimator
    estimator = DeepAREstimator(freq=config.freq,
    context_length=config.context_length,
    prediction_length=config.prediction_length,
    num_layers=config.num_layers, num_cells=config.num_cells,
    cell_type=config.cell_type,trainer=config.trainer,distr_output=config.distr_output)
    #train the estimator
    predictor=estimator.train(training_data=training_data)
    if type(test_data)==gluonts.dataset.split.TestData:
      
        forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_data,  # test dataset
        predictor=predictor,  # predictor
        num_samples=100,  # number of sample paths we want for evaluation
        )
        '''
        forecasts,tss=[],[]
        #test_input=Zeitraum vor Vorhergesagten Werten, Test_label= Vorherzusagende(r) Wert(e)
        for test_input, test_label in test_data:
            forecasts.append(test_input)
            tss.append(test_label)'''
        return forecast_it, ts_it
    else:
        #predictions = predictor.predict(test_data)
        forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_data,  # test dataset
        predictor=predictor,  # predictor
        num_samples=100,  # number of sample paths we want for evaluation
        )
        forecasts = list(forecast_it)
        tss = list(ts_it)
        
    return forecasts,tss
    
    

def preprocessing(config,df,check_count=False,output_type="PD"):
    """
    This function processes the data into either a correctly spaced pd.DataFrame, a PandasDataset, a ListDataset or
    a pd.Dataframe where only the index has been set.
    We also have the option to receive an output of the count of each location, with fewer observations than the maximum
    observations within the training and testing period.
    """
    df['date']=pd.to_datetime(df['date'])
    df=df.set_index('date')
    if check_count:
        count_dict={}
        for location in df.location.unique():
            #save the amount of values within the train & test time period into the count_dict
            location_df=df.loc[(df['location']==location) & (df.index >config.train_start_time)&(df.index <=config.test_end_time),:]
            count_dict[location]=location_df.shape[0]
        #print out the distribution of each region with missing values
        print('LK mit weniger als'+ str(max(count_dict.values())))
        missing_values_dict={k:v for k, v in count_dict.items() if v < max(count_dict.values())}
        print(missing_values_dict)
        return df,missing_values_dict
    
    if output_type in ['PD','LD','corrected_df']:
        #Create a DataFrame Blueprint
        correctly_spaced_index=pd.date_range(start=config.train_start_time, end=config.test_end_time,freq="W-SUN")
        correctly_spaced_location_df=pd.DataFrame(index=correctly_spaced_index)
        correctly_spaced_df=pd.DataFrame()
        location_list=df.loc[:,'location'].unique()
        #location_list=['LK Bad Dürkheim','SK Speyer', "LK Emsland"]
        for location in location_list:
            temporary_df=correctly_spaced_location_df.join(df.loc[df.location==location])
            temporary_df['location']=temporary_df['location'].fillna(location)
            temporary_df['age_group']=temporary_df['age_group'].fillna("00+")
            correctly_spaced_df=pd.concat([correctly_spaced_df, temporary_df])
        if output_type == "PD":
            df=PandasDataset.from_long_dataframe(dataframe=correctly_spaced_df,item_id='location', target="value",freq="W-SUN")
        if output_type =="LD":
            df=ListDataset([{"start": min(correctly_spaced_index),"target": correctly_spaced_df.loc[correctly_spaced_df.location == x,'value']} for x in location_list],freq=config.freq)
        if output_type=="corrected_df":
            return correctly_spaced_df
    return df

def data_split(config,df,test_pairs=True):
    """
    This function performs a data split into a training set and a testing set, this split is based upon the training and testing
    times set within the Configuration.py File. 
    We differentiate between a split performed via the gluonts split module (test_pairs==True) or a direct assignment into 
    ListDatasets (test_pairs==False).
    """
    df=df.copy()
    start=df.loc[(df.index>=config.train_start_time)].index[0]
    if not test_pairs:
        #Train Data
        training_data= ListDataset(
            [{"start": start,"target": df.loc[(df.index>=config.train_start_time)&(df.index<=config.train_end_time)& (df.location == x),config.target]} for x in df.loc[:,'location'].unique()],freq=config.freq)

        #Test Data
        test_start=df.loc[(df.index>=config.train_end_time)&(df.index<=config.test_end_time)].index[0]
        test_data= ListDataset(
            [{"start": start,"target": df.loc[(df.index<=config.test_end_time) &(df.index>=config.train_start_time)& (df.location == x),config.target]} for x in df.loc[: ,'location'].unique()],freq=config.freq)
        return training_data,test_data
    else:
        #SPLIT WITH GLUON TS SPLIT
        dataset=ListDataset([{"start": start,"target": df.loc[(df.index<=config.test_end_time) &(df.index>=config.train_start_time)& (df.location == x),config.target]} for x in df.loc[: ,'location'].unique()],freq=config.freq)

        training_data, test_template = split(dataset, date=pd.Period(config.train_end_time, freq=config.freq))
        test_pairs = test_template.generate_instances(prediction_length=config.prediction_length,windows=config.windows,)

        return training_data,test_pairs