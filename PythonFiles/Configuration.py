from datetime import datetime
from gluonts.mx import Trainer, DeepAREstimator
from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.mx.distribution import NegativeBinomialOutput

class Configuration:
    def __init__(self):
        #Time parameter
        self.train_start_time = datetime(2010,1,1,0,0,0)
        self.train_end_time = datetime(2017,12,31,23,0,0)
        self.test_end_time = datetime(2019,12,31,23,0,0)
        
        #deepAR parameters
        self.freq = "W-SUN"
        self.context_length = 52   # in number of weeks
        self.prediction_length = 4   # in number of weeks ->1 Week (104 Test Windows), 13W(8TW), 26W(4TW), 52W(2TW),... 
        self.windows = int(104 / self.prediction_length)
        self.num_layers = 2
        self.num_cells = 128
        self.cell_type = "lstm"
        self.trainer = Trainer(epochs=4)
        self.distr_output = NegativeBinomialOutput()
        
        self.deeparestimator = DeepAREstimator(freq=self.freq,
                        context_length=self.context_length,
                        prediction_length=self.prediction_length,
                        num_layers=self.num_layers,
                        num_cells=self.num_cells,
                        cell_type=self.cell_type,
                        trainer=self.trainer,
                        distr_output=self.distr_output,
                        )
        
        self.num_hidden_dimensions = [10]
        self.feedforwardestimator = SimpleFeedForwardEstimator(num_hidden_dimensions=self.num_hidden_dimensions,
                                                              prediction_length=self.prediction_length,
                                                              context_length=self.context_length,
                                                              distr_output=self.distr_output,
                                                              trainer=self.trainer
                                                              )
        
        self.target = "value"
        self.quantiles = [0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975]
        
