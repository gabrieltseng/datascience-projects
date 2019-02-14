import torch
from bs4 import BeautifulSoup
from pathlib import Path


def plot_county_errors(model, svg_file=Path('data/counties.svg')):
    """
    For the most part, reformatting of
    https://github.com/JiaxuanYou/crop_yield_prediction/blob/master/6%20result_analysis/yield_map.py

    Generates an svg of the counties, coloured by their prediction error.

    Parameters
    ----------
    model: pathlib Path
        Path to the model being plotted.
    svg_file: pathlib Path, default=Path('data/counties.svg')
        Path to the counties svg file used as a base
    """

    model_sd = torch.load(model)

    model_dir = model.parents[0]

    real_values = model_sd['val_real']
    pred_values = model_sd['val_pred']

    gp = True
    try:
        gp_values = model_sd['val_pred_gp']
    except KeyError:
        gp = False

    indices = model_sd['val_indices']

    pred_err = pred_values - real_values
    pred_dict = {}
    for idx, err in zip(indices, pred_err):
        state, county = idx

        state = str(state).zfill(2)
        county = str(county).zfill(3)

        pred_dict[state + county] = err

    model_info = model.name[:-8].split('_')

    _single_plot(pred_dict, svg_file, model_dir / f'{model_info[0]}_{model_info[1]}.svg')

    if gp:
        gp_pred_err = gp_values - real_values
        gp_dict = {}
        for idx, err in zip(indices, gp_pred_err):
            state, county = idx

            state = str(state).zfill(2)
            county = str(county).zfill(3)

            gp_dict[state + county] = err

    _single_plot(gp_dict, svg_file, model_dir / f'{model_info[0]}_{model_info[1]}_gp.svg')


def _single_plot(err_dict, svg_file, savepath):

    # load the svg file
    svg = svg_file.open('r').read()
    # Load into Beautiful Soup
    soup = BeautifulSoup(svg)
    # Find counties
    paths = soup.findAll('path')

    path_style = 'font-size:12px;fill-rule:nonzero;stroke:#FFFFFF;stroke-opacity:1;stroke-width:0.1' \
                 ';stroke-miterlimit:4;stroke-dasharray:none;stroke-linecap:butt;marker-start' \
                 ':none;stroke-linejoin:bevel;fill:'
    colors = ["#b2182b", "#d6604d", "#f4a582", "#fddbc7", "#d1e5f0", "#92c5de", "#4393c3", "#2166ac"]

    for p in paths:
        if p['id'] not in ["State_Lines", "separator"]:
            try:
                rate = err_dict[p['id']]
            except KeyError:
                continue
            if rate > 15:
                color_class = 7
            elif rate > 10:
                color_class = 6
            elif rate > 5:
                color_class = 5
            elif rate > 0:
                color_class = 4
            elif rate > -5:
                color_class = 3
            elif rate > -10:
                color_class = 2
            elif rate > -15:
                color_class = 1
            else:
                color_class = 0

            color = colors[color_class]
            p['style'] = path_style + color
    soup = soup.prettify()
    with savepath.open('w') as f:
        f.write(soup)
