from datetime import datetime
from gluonts.mx import Trainer, DeepAREstimator, SimpleFeedForwardEstimator
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
                          'Brandenburg an der Havel': ['SK Brandenburg a.d.Havel'],
                          'Coburg': ['LK Coburg', 'SK Coburg'],
                          'Darmstadt': ['SK Darmstadt'],
                          'Dillingen a.d. Donau': ['LK Dillingen a.d.Donau'],
                          'Eifelkreis-Bitburg-Pr??m': ['LK Bitburg-Pr??m'],
                          'Eisenach': [],                # not available in the influenza location names
                          'Erlangen': ['SK Erlangen'],
                          'Flensburg': ['SK Flensburg'],
                          'Frankenthal (Pfalz)': ['SK Frankenthal'],
                          'Freiburg im Breisgau': ['SK Freiburg i.Breisgau'],
                          'F??rth': ['LK F??rth', 'SK F??rth'],
                          'Gera': ['SK Gera'],
                          'Halle (Saale)': ['SK Halle'],
                          'Heilbronn': ['LK Heilbronn', 'SK Heilbronn'],
                          'Hof': ['LK Hof', 'SK Hof'],
                          'Kaiserslautern': ['LK Kaiserslautern', 'SK Kaiserslautern'],
                          'Karlsruhe': ['LK Karlsruhe', 'SK Karlsruhe'],
                          'Kassel': ['LK Kassel', 'SK Kassel'],
                          'Kempten (Allg??u)': ['SK Kempten'],
                          'Koblenz': ['SK Koblenz'],
                          'Landau in der Pfalz': ['SK Landau i.d.Pfalz'],
                          'Landsberg am Lech': ['LK Landsberg a.Lech'],
                          'Landshut': ['LK Landshut', 'SK Landshut'],
                          'Leipzig': ['LK Leipzig', 'SK Leipzig'],
                          'Lindau (Bodensee)': ['LK Lindau'],
                          'Ludwigshafen am Rhein': ['SK Ludwigshafen'],
                          'Mainz': ['SK Mainz'],
                          'M??hldorf a. Inn': ['LK M??hldorf a.Inn'],
                          'M??lheim an der Ruhr': ['SK M??lheim a.d.Ruhr'],
                          'M??nchen': ['LK M??nchen', 'SK M??nchen'],
                          'Neumarkt i.d. OPf.': ['LK Neumarkt i.d.OPf.'],
                          'Neustadt a.d. Aisch-Bad Windsheim': ['LK Neustadt a.d.Aisch-Bad Windsheim'],
                          'Neustadt a.d. Waldnaab': ['LK Neustadt a.d.Waldnaab'],
                          'Neustadt an der Weinstra??e': ['SK Neustadt a.d.Weinstra??e'],
                          'N??rnberg': ['SK N??rnberg'], 
                          'Offenbach': ['LK Offenbach'], 
                          'Offenbach am Main': ['SK Offenbach'], 
                          'Oldenburg': ['LK Oldenburg'],
                          'Oldenburg (Oldenburg)': ['SK Oldenburg'],    
                          'Osnabr??ck': ['LK Osnabr??ck', 'SK Osnabr??ck'],
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
                          'W??rzburg': ['LK W??rzburg', 'SK W??rzburg']}
        
        self.berlin_neighbors = {'SK Berlin Charlottenburg-Wilmersdorf':['SK Berlin Mitte', 'SK Berlin Reinickendorf','SK Berlin Spandau',
                                                                        'SK Berlin Steglitz-Zehlendorf','SK Berlin Tempelhof-Sch??neberg'],
                                 'SK Berlin Friedrichshain-Kreuzberg':['SK Berlin Lichtenberg','SK Berlin Mitte','SK Berlin Neuk??lln',
                                                                      'SK Berlin Pankow','SK Berlin Tempelhof-Sch??neberg',
                                                                      'SK Berlin Treptow-K??penick'],
                                 'SK Berlin Lichtenberg':['SK Berlin Friedrichshain-Kreuzberg','SK Berlin Marzahn-Hellersdorf',
                                                         'SK Berlin Pankow', 'SK Berlin Treptow-K??penick', 'LK Barnim'],
                                 'SK Berlin Marzahn-Hellersdorf':['SK Berlin Lichtenberg', 'SK Berlin Treptow-K??penick',
                                                                 'LK Barnim', 'LK M??rkisch-Oderland'],
                                 'SK Berlin Mitte':['SK Berlin Charlottenburg-Wilmersdorf','SK Berlin Friedrichshain-Kreuzberg',
                                                   'SK Berlin Pankow','SK Berlin Reinickendorf', 'SK Berlin Tempelhof-Sch??neberg'],
                                 'SK Berlin Neuk??lln':['SK Berlin Friedrichshain-Kreuzberg', 'SK Berlin Tempelhof-Sch??neberg',
                                                      'SK Berlin Treptow-K??penick', 'LK Dahme-Spreewald'],
                                 'SK Berlin Pankow':['SK Berlin Friedrichshain-Kreuzberg','SK Berlin Lichtenberg','SK Berlin Mitte',
                                                    'SK Berlin Reinickendorf', 'LK Barnim', 'LK Oberhavel'],
                                 'SK Berlin Reinickendorf':['SK Berlin Charlottenburg-Wilmersdorf','SK Berlin Mitte','SK Berlin Pankow',
                                                           'SK Berlin Spandau', 'LK Oberhavel'],
                                 'SK Berlin Spandau':['SK Berlin Charlottenburg-Wilmersdorf','SK Berlin Reinickendorf',
                                                     'SK Berlin Steglitz-Zehlendorf', 'LK Oberhavel', 'LK Havelland', 'SK Potsdam'],
                                 'SK Berlin Steglitz-Zehlendorf':['SK Berlin Charlottenburg-Wilmersdorf', 'SK Berlin Spandau',
                                                                 'SK Berlin Tempelhof-Sch??neberg', 'SK Potsdam', 'LK Potsdam-Mittelmark',
                                                                 'LK Teltow-Fl??ming'],
                                 'SK Berlin Tempelhof-Sch??neberg':['SK Berlin Charlottenburg-Wilmersdorf','SK Berlin Friedrichshain-Kreuzberg',
                                                                  'SK Berlin Mitte','SK Berlin Neuk??lln','SK Berlin Steglitz-Zehlendorf',
                                                                  'LK Dahme-Spreewald', 'LK Teltow-Fl??ming'],
                                 'SK Berlin Treptow-K??penick':['SK Berlin Friedrichshain-Kreuzberg','SK Berlin Lichtenberg',
                                                               'SK Berlin Marzahn-Hellersdorf','SK Berlin Neuk??lln', 'LK M??rkisch-Oderland',
                                                              'LK Dahme-Spreewald', 'LK Oder-Spree'],
                                 'SK Potsdam': ['LK Havelland', 'LK Potsdam-Mittelmark', 'SK Berlin Spandau',
                                                'SK Berlin Steglitz-Zehlendorf'],
                                 'LK Barnim': ['LK M??rkisch-Oderland', 'LK Oberhavel', 'LK Uckermark', 'SK Berlin Lichtenberg',
                                              'SK Berlin Marzahn-Hellersdorf', 'SK Berlin Pankow'],
                                 'LK Dahme-Spreewald': ['LK Elbe-Elster', 'LK Oberspreewald-Lausitz', 'LK Oder-Spree',
                                                        'LK Spree-Nei??e', 'LK Teltow-Fl??ming', 'SK Berlin Treptow-K??penick', 
                                                        'SK Berlin Tempelhof-Sch??neberg', 'SK Berlin Neuk??lln'],
                                 'LK Havelland': ['SK Brandenburg a.d.Havel', 'SK Potsdam', 'LK Oberhavel', 'LK Ostprignitz-Ruppin',
                                                  'LK Potsdam-Mittelmark', 'LK Jerichower Land', 'LK Stendal', 'SK Berlin Spandau'],
                                 'LK M??rkisch-Oderland': ['SK Frankfurt (Oder)', 'LK Barnim', 'LK Oder-Spree',
                                                         'SK Berlin Marzahn-Hellersdorf', 'SK Berlin Treptow-K??penick'],
                                 'LK Oberhavel': ['LK Barnim', 'LK Havelland', 'LK Ostprignitz-Ruppin', 'LK Uckermark',
                                                  'LK Mecklenburgische Seenplatte', 'SK Berlin Pankow', 'SK Berlin Reinickendorf',
                                                  'SK Berlin Spandau'],
                                 'LK Oder-Spree': ['SK Frankfurt (Oder)', 'LK Dahme-Spreewald', 'LK M??rkisch-Oderland', 'LK Spree-Nei??e',
                                                  'SK Berlin Treptow-K??penick'],
                                 'LK Potsdam-Mittelmark': ['SK Brandenburg a.d.Havel', 'SK Potsdam', 'LK Havelland', 'LK Teltow-Fl??ming',
                                                           'LK Anhalt-Bitterfeld', 'LK Wittenberg', 'LK Jerichower Land',
                                                           'SK Berlin Steglitz-Zehlendorf'],
                                 'LK Teltow-Fl??ming': ['LK Dahme-Spreewald', 'LK Elbe-Elster', 'LK Potsdam-Mittelmark', 'LK Wittenberg',
                                                      'SK Berlin Tempelhof-Sch??neberg', 'SK Berlin Steglitz-Zehlendorf']}
