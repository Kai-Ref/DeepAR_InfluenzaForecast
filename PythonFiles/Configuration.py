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
        
        self.specific_matches={'Altenkirchen (Westerwald)': ['LK Altenkirchen'],
                          'Amberg': ['SK Amberg'],
                          'Ansbach': ['LK Ansbach', 'SK Ansbach'],
                          'Aschaffenburg': ['LK Aschaffenburg', 'SK Aschaffenburg'],
                          'Augsburg': ['LK Augsburg', 'SK Augsburg'],
                          'Bamberg': ['LK Bamberg', 'SK Bamberg'],
                          'Bayreuth': ['SK Bayreuth', 'LK Bayreuth'],
                          #Berlin isn't correctly split in Shapefile
                          'Berlin': ['SK Berlin Charlottenburg-Wilmersdorf', 'SK Berlin Friedrichshain-Kreuzberg',
                                     'SK Berlin Lichtenberg', 'SK Berlin Marzahn-Hellersdorf', 'SK Berlin Mitte',
                                     'SK Berlin Neukölln', 'SK Berlin Pankow', 'SK Berlin Reinickendorf', 'SK Berlin Spandau',
                                     'SK Berlin Steglitz-Zehlendorf', 'SK Berlin Tempelhof-Schöneberg', 'SK Berlin Treptow-Köpenick'],
                          'Brandenburg an der Havel': ['SK Brandenburg a.d.Havel'],
                          'Coburg': ['LK Coburg', 'SK Coburg'],
                          'Darmstadt': ['SK Darmstadt'],
                          'Dillingen a.d. Donau': ['LK Dillingen a.d.Donau'],
                          'Eifelkreis-Bitburg-Prüm': ['LK Bitburg-Prüm'],
                          'Eisenach': [],                # not available in the influenza location names
                          'Erlangen': ['SK Erlangen'],
                          'Flensburg': ['SK Flensburg'],
                          'Frankenthal (Pfalz)': ['SK Frankenthal'],
                          'Freiburg im Breisgau': ['SK Freiburg i.Breisgau'],
                          'Fürth': ['LK Fürth', 'SK Fürth'],
                          'Gera': ['SK Gera'],
                          'Halle (Saale)': ['SK Halle'],
                          'Heilbronn': ['LK Heilbronn', 'SK Heilbronn'],
                          'Hof': ['LK Hof', 'SK Hof'],
                          'Kaiserslautern': ['LK Kaiserslautern', 'SK Kaiserslautern'],
                          'Karlsruhe': ['LK Karlsruhe', 'SK Karlsruhe'],
                          'Kassel': ['LK Kassel', 'SK Kassel'],
                          'Kempten (Allgäu)': ['SK Kempten'],
                          'Koblenz': ['SK Koblenz'],
                          'Landau in der Pfalz': ['SK Landau i.d.Pfalz'],
                          'Landsberg am Lech': ['LK Landsberg a.Lech'],
                          'Landshut': ['LK Landshut', 'SK Landshut'],
                          'Leipzig': ['LK Leipzig', 'SK Leipzig'],
                          'Lindau (Bodensee)': ['LK Lindau'],
                          'Ludwigshafen am Rhein': ['SK Ludwigshafen'],
                          'Mainz': ['SK Mainz'],
                          'Mühldorf a. Inn': ['LK Mühldorf a.Inn'],
                          'Mülheim an der Ruhr': ['SK Mülheim a.d.Ruhr'],
                          'München': ['LK München', 'SK München'],
                          'Neumarkt i.d. OPf.': ['LK Neumarkt i.d.OPf.'],
                          'Neustadt a.d. Aisch-Bad Windsheim': ['LK Neustadt a.d.Aisch-Bad Windsheim'],
                          'Neustadt a.d. Waldnaab': ['LK Neustadt a.d.Waldnaab'],
                          'Neustadt an der Weinstraße': ['SK Neustadt a.d.Weinstraße'],
                          'Nürnberg': ['SK Nürnberg'], 
                          'Offenbach': ['LK Offenbach', 'SK Offenbach'], 
                          'Offenbach am Main': [],            # not available in the influenza location names
                          'Oldenburg': ['SK Oldenburg', 'LK Oldenburg'],
                          'Oldenburg (Oldenburg)': [],        # not available in the influenza location names
                          'Osnabrück': ['LK Osnabrück', 'SK Osnabrück'],
                          'Osterode am Harz': [],             # not available in the influenza location names
                          'Passau': ['LK Passau', 'SK Passau'],
                          'Pfaffenhofen a.d. Ilm': ['LK Pfaffenhofen a.d.Ilm'],
                          'Potsdam': ['SK Potsdam'],
                          'Regen': ['LK Regen'],
                          'Regensburg': ['LK Regensburg', 'SK Regensburg'],
                          'Rosenheim': ['LK Rosenheim', 'SK Rosenheim'],
                          'Rostock': ['LK Rostock', 'SK Rostock'],
                          'Schweinfurt': ['LK Schweinfurt', 'SK Schweinfurt'],
                          'St. Wendel': ['LK Sankt Wendel'],
                          'Straubing': ['SK Straubing'],
                          'Trier': ['SK Trier'],
                          'Ulm': ['SK Ulm'],
                          'Weiden i.d. OPf.': ['SK Weiden i.d.OPf.'], 
                          'Weimar': ['SK Weimar'],
                          'Worms': ['SK Worms'],
                          'Wunsiedel i. Fichtelgebirge': ['LK Wunsiedel i.Fichtelgebirge'],
                          'Würzburg': ['LK Würzburg', 'SK Würzburg']}
