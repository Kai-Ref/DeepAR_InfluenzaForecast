from datetime import datetime
from gluonts.mx import Trainer
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
        
        self.target = "value"
