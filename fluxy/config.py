import numpy as np
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

color_palette = {0:[['blue','dodgerblue'],
                    ['dodgerblue','skyblue'],
                    ['deepskyblue','cyan']],
                 1:[['purple','mediumpurple'],
                    ['deeppink','pink'],
                    ['darkorange','red']],
                 2:[['darkgreen','green'],
                    ['limegreen','palegreen'],
                    ['olive','lightgreen']]}

point_source_dict = {
                    'paris':[2.3404,48.8600],
                    'nw_england':[-2.7969,53.7748],
                    'london':[-0.1278, 51.5074],
                    'edinburgh':[-3.1883, 55.9533],
                    'cardiff':[-3.1791, 51.4816],
                    'belfast':[-5.9301, 54.5973],
                    'zurich': [8.5417, 47.3769],
                    'geneva': [6.1432, 46.2044],
                    'basel': [7.5886, 47.5596],
                    'lausanne': [6.6323, 46.5197],
                    'bern': [7.4474, 46.9481],
                    'berlin': [13.4050, 52.5200],
                    'hamburg': [9.9937, 53.5511],
                    'munich': [11.5820, 48.1351],
                    'koeln': [6.9603, 50.9375],
                    'frankfurt': [8.6821, 50.1109],
                    'essen': [7.0115, 51.4556],
                    'rome': [12.4964, 41.9028],
                    'milan': [9.1900, 45.4642],
                    'naples': [14.2681, 40.8518],
                    'turin': [7.6869, 45.0703],
                    'palermo': [13.3615, 38.1157],
                    'amsterdam': [4.9041, 52.3676],
                    'rotterdam': [4.4777, 51.9244],
                    'hague': [4.3007, 52.0705],
                    'utrecht': [5.1214, 52.0907],
                    'eindhoven': [5.4797, 51.4416],
                    'dublin': [-6.2603, 53.3498],
                    'cork': [-8.4727, 51.8985],
                    'limerick': [-8.6305, 52.6638],
                    'galway': [-9.0579, 53.2707],
                    'waterford': [-7.1119, 52.2593],
                    'budapest': [19.0402, 47.4979],
                    'debrecen': [21.6273, 47.5316],
                    'szeged': [20.1488, 46.2530],
                    'miskolc': [20.7852, 48.0993],
                    'pecs': [18.2324, 46.0784],
                    'oslo': [10.7522, 59.9139],
                    'bergen': [5.3241, 60.3929],
                    'sandnes': [58.8514, 58.8500],
                    'stavanger': [5.7382, 58.9690],
                    'drammen': [10.2045, 59.7438],
                    'brussels':[4.3525, 50.8466],
                    'antwerp':[4.4002, 51.2177],
                    'ghent':[3.7252, 51.0536],
                    'charleroi':[4.4444, 50.4113],
                    'liege':[5.5674, 50.6337],
                    'luxembourg':[6.1319, 49.6116]
                    }

countrycodes_dict = {'IRELAND':'IRL',
                     'UK':'GBR',
                     'FRANCE':'FRA',
                     'NETHERLANDS':'NLD',
                     'GERMANY':'DEU',
                     'DENMARK':'DNK',
                     'SWITZERLAND':'CHE',
                     'AUSTRIA':'AUT',
                     'ITALY':'ITA',
                     'BELGIUM': 'BEL',
                     'LUXEMBOURG': 'LUX',
                     'HUNGARY':'HUN',
                     'SWEDEN':'SWE',
                     'POLAND':'POL',
                     'CZECHIA':'CZE',
                     'CROATIA':'HRV',
                     'SLOVAKIA':'SVK',
                     'FINLAND':'FIN',
                     'SLOVENIA':'SVN',
                     'GREECE':'GRC',
                     'SPAIN':'ESP',
                     'PORTUGAL':'PRT',
                     'NORWAY':'NOR'}

regions_dict = {'BELUX':'BEL-LUX',
                'BENELUX':'BEL-LUX-NLD',
                'CW_EU':'AUT-BEL-CHE-CZE-DEU-ESP-FRA-GBR-HRV-HUN-IRL-ITA-LUX-NLD-POL-PRT-SVK-SVN',
                'EU_GRP2':'AUT-BEL-CHE-DEU-DNK-FRA-GBR-IRL-ITA-LUX-NLD',
                'NW_EU':'BEL-DEU-DNK-FRA-GBR-IRL-LUX-NLD',
                'NW_EU2':'BEL-DEU-FRA-GBR-IRL-LUX-NLD',
                'NW_EU_CONTINENT':'BEL-DEU-FRA-LUX-NLD'}

regions_dict_old = {'CW_EU':'AUT-BEL-CHE-CZE-DEU-ESP-FRA-GBR-HRV-HUN-IRL-ITA-LUX-NLD-POL-PRT-SVK-SVK'}

countrycodes_dict.update(regions_dict)

# population from 2018 to 2023 (at Jan 1 each year)
bel_pop = np.array([11.399,11.455,11.522,11.555,11.618,11.723])
lux_pop = np.array([0.602,0.614,0.626,0.635,0.645,0.661])
bel_pop_r = np.round(np.mean(bel_pop/(bel_pop+lux_pop)),3)

mf_labels = {'Yapriori':'prior mf',
             'Yapost':'posterior mean mf',
             'YaprioriBC':'prior baseline',
             'YapostBC':'posterior mean baseline',
             'Yapost_bias':'posterior bias',
             'YaprioriOUTER':'prior outer region mf',
             'YapostOUTER':'posterior outer region mf',
             'Yobs':'observed mf',
             'uYobs_repeatability':'obs repeatability mf uncertainty',
             'uYobs_variability':'obs variability mf uncertainty',
             'uYmod':'model uncertainty',
             'uYtotal':'total uncertainty'
             }

mf_color_index = {'Yapriori':1,
                  'Yapost':0,
                  'YaprioriBC':1,
                  'YapostBC':0,
                  'Yapriori_bias':1,
                  'Yapost_bias':0,
                  'YaprioriOUTER':1,
                  'YapostOUTER':0,
                  'Yobs':1,
                  'uYobs_repeatability':0,
                  'uYobs_variability':0,
                  'uYmod':1,
                  'uYtotal':1
                  }

flux_labels = {'flux_total_prior':'Prior',
               'flux_total_posterior_inversion_grid':'Posterior',
               'flux_total_posterior':'Posterior',
              }

stat_labels = {'pearson': 'Pearson correlation coefficient',
               'rmse'   : 'RMSE',
               'nrmse'  : 'Normalised RMSE',
               'std'    : 'Standard deviation'
               }

# Acceptable units and conversion factor to base unit
units_scale = {'mf': {'mol mol-1' : 1, # mf base unit
                      'ppm' : 1e-6,
                      'ppb' : 1e-9,
                      'ppt' : 1e-12
                      },
               'amount': {'kmol' : 1e3,
                          'mol' : 1 # amount of substance base unit
                          },
               'mass': {'Tg' : 1e12,
                        'Gg' : 1e9,
                        'Mg' : 1e6,
                        'kg' : 1e3,
                        'g'  : 1  # mass base unit
                        },
                'time':{'yr': 60*60*24*365,
                        'a': 60*60*24*365,
                        's' : 1  # time base unit
                        },
                'length':{'km': 1e3,
                          'm' : 1 # length base unit
                          },
                'nd':{'1': 1} # non-dimensional
              }

def set_print_settings(presentation_mode: bool = False) -> dict[int, list]:
    """
    Sets font size and annotation coordinates.

    Args:
        ppt_mode (logical) (optional):
            If True, use bigger fonts (ideal for presentation slides)

    Returns:
        annotate_coords (dict of lists):
            Coordinates to annotate histogram.
    """

    if (presentation_mode):
        # Set big font size (ideal for presentation slides)
        plt.rc('font', size=15)
        plt.rc('axes', titlesize=18)
        plt.rc('axes', labelsize=16)
        plt.rc('xtick', labelsize=15)
        plt.rc('ytick', labelsize=15)
        plt.rc('legend', fontsize=14)

        annotate_coords = {0:[0.58,0.65],
                           1:[0.58,0.40],
                           2:[0.58,0.15]}

        logger.warning('Using big fonts when plotting. You might need to define shorter labels.')
    
    else:
        # Set small font size (ideal for text documents)
        plt.rc('font', size=11)
        plt.rc('axes', titlesize=11)
        plt.rc('axes', labelsize=10)
        plt.rc('xtick', labelsize=11)
        plt.rc('ytick', labelsize=11)
        plt.rc('legend', fontsize=10)

        annotate_coords = {0:[0.6,0.80],
                           1:[0.6,0.60],
                           2:[0.6,0.40]}

    return annotate_coords

def set_model_colors(models: list[str]) -> dict[str, list]:
    """
    Sets plotting colors for each model.

    Args:
        models (list of str):
            Keys specifying model names, e.g. ['intem','elris']

    Returns:
        model_colors (dict of lists):
            List of colors to be used by each model.
    """

    model_colors = dict()
    max_color_groups = len(color_palette)

    # Get unique inversion systems
    # dict.fromkeys() is used because it conserves the order of the models
    unique_models = list(dict.fromkeys(m.split('_')[0] for m in models))
    n_unique_models = len(unique_models)

    if n_unique_models == 1:
        # The results to plot are from a single inversion system
        i = 0; index_colors = 0

        # Use color_palette in order
        for m in models:            
            if index_colors == len(color_palette[i]):
                i = i+1
                index_colors = 0

            if i == max_color_groups:
                raise ValueError(f'Number of models to plot is greater than number of pre-defined colors. Add more colors to color_palette.')

            model_colors[m] = color_palette[i][index_colors]
            index_colors = index_colors+1

    else:
        # The results to plot are from multiple inversion systems
        index_colors = [0]*n_unique_models
        index_colors_max = [len(color_palette[i]) for i in color_palette]

        for m in models:
            model_name = m.split('_')[0]
            i = unique_models.index(model_name)

            if i == max_color_groups:
                raise KeyError(f'color_palette has only {i} keys. Add more keys to plot results from more than {i} distinct inversion models.')
            
            if index_colors[i] == index_colors_max[i]:
                raise KeyError(f'There are more than {index_colors[i]} results from {model_name} but only {index_colors[i]} elements in color_palette[{i}]. Add more pairs of colors to the list.')

            # For each inversion system, get plotting colors from a single color_palette key
            model_colors[m] = color_palette[i][index_colors[i]]
            index_colors[i] = index_colors[i]+1

    return model_colors

def set_model_labels(models: list[str], config_data: dict[str, dict], get_labels_from_file: bool
) -> dict[str, str]:

    model_labels = {}

    for m in models:
        # Get model name components
        name_tags = m.split('_')

        # Get labels
        label = None
        if get_labels_from_file and "model_labels" in config_data["models_info"]:
            label = config_data["models_info"]["model_labels"].get(m, None)

        if label is None:
            label = " ".join(name_tags)

        model_labels[m] = label

    return model_labels
