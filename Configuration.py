from datetime import datetime
from gluonts.mx import Trainer
from gluonts.mx.distribution import NegativeBinomialOutput

class Configuration:
    def __init__(self):
        #Time parameter
        self.train_start_time=datetime(2010,1,1,0,0,0)
        self.train_end_time=datetime(2017,12,31,23,0,0)
        self.test_end_time=datetime(2019,12,31,23,0,0)
        self.target='value'
        
        #deepAR parameters
        self.freq="W-SUN"
        self.context_length=52#number of weeks
        self.prediction_length=26#number of weeks -> Vielfache:1 Woche (104 Test Windows), 13 Wochen (8 TW),26 Wochen (4 TW),52W(2TW),104W(1TW)
        self.windows=int(104/self.prediction_length)
        self.num_layers=2
        self.num_cells=128
        self.cell_type="lstm"
        self.trainer=Trainer(epochs=4)
        self.distr_output=NegativeBinomialOutput()
