import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

import numpyro
from numpyro.infer import Predictive, MCMC, NUTS
import jax.numpy as np
from jax.random import PRNGKey
import arviz as az

from fetching import fetch_all
from preprocessing import make_all_datasets, plot_dataset
from model import multi_model
from rt_mobility import reorder
from postprocess import plot_results
from train_test_split import mask_ix, split_train_test

numpyro.enable_x64()
# matplotlib.use('agg')


NB_DAYS_BEFORE_TEN_DEATHS = 30

MOBILITY_CATEGORIES = [
#         'transit', 'walking', 'driving', 
    'grocery_and_pharmacy', 'retail_and_recreation', 'workplaces', 'transit_stations']

TRAIN_COUNTRIES = ['Italy']
TEST_COUNTRIES = ['United Kingdom', 'Sweden']

datasets = fetch_all()
for name, data in datasets.items():
    globals()[name] = data

SELECTED_COUNTRIES = [country_code_lookup.loc[name] 
                      for name in 
                      ('Denmark', 'Sweden', 'France', 'Germany', 'United Kingdom', 
                       'Spain', 'Italy'
                      )
                     ]

all_countries, all_populations, all_mobilities, all_deaths, all_times = make_all_datasets(
    mobility,
    ecdc,
    populations_country, 
    SELECTED_COUNTRIES, 
    NB_DAYS_BEFORE_TEN_DEATHS,
    MOBILITY_CATEGORIES 
    )

mask_train, mask_test = split_train_test(
    [country_code_lookup.loc[name] for name in TRAIN_COUNTRIES], 
    [country_code_lookup.loc[name] for name in TEST_COUNTRIES], 
    all_countries)

if __name__ == "__main__":    
    print(mask_ix(all_countries, mask_train), mask_ix(all_countries, mask_test))
    mcmc_multi = MCMC(NUTS(multi_model, dense_mass=True), 200, 200, num_chains=1)
    mcmc_multi.run(PRNGKey(0), 
                all_mobilities=mask_ix(all_mobilities, mask_train),
                all_populations=mask_ix(all_populations, mask_train),
                observations=mask_ix(all_deaths, mask_train)
                )

    mcmc_multi.print_summary()

    inference_data = az.from_numpyro(mcmc_multi)
    fmt_train_countries = '-'.join(TRAIN_COUNTRIES)
    az.to_netcdf(inference_data, f'../output/inference/{fmt_train_countries}.nc')
