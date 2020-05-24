from io import StringIO
import re
import requests

import jax.numpy as np
import pandas as pd

# country codes
def fetch_isocodes():
    isocodes = pd.read_csv('../input/countries-iso-codes/wikipedia-iso-country-codes.csv')
    isocodes.columns = isocodes.columns.str.replace(' ', '_').str.lower()
    isocodes = isocodes.rename({"english_short_name_lower_case": 'country_name'}, axis=1)
    return isocodes

# ACAPS
def fetch_acaps(isocodes, url=None):
    filepath = '../input/acaps-covid19-government-measures/acaps_covid19_goverment_measures_dataset.xlsx'
#     if url is not None:
#         measures = pd.read_excel(url, sheet_name='Database')
#         measures.to_csv(filepath, index=False)
    
    measures = pd.read_excel(filepath, sheet_name='Database')
    measures.columns = measures.columns.str.lower()
    measures['date_implemented'] = pd.to_datetime(measures['date_implemented'])

    measures = measures.merge(isocodes, left_on='iso', right_on='alpha-3_code')
    return measures

# ECDC for names of lactions
def fetch_ecdc():
    ecdc = (pd.read_csv('https://covid.ourworldindata.org/data/owid-covid-data.csv')
            .assign(date=lambda f: f['date'].pipe(pd.to_datetime))
           )
    
    return ecdc

# Apple mobility

def fetch_apple(location_code, url=None):
    filepath = '../input/apple-covid-mobility/applemobilitytrends.csv'
    if url is not None:
        response = requests.get(url)
        apple_mobility = (pd.read_csv(StringIO(response.content.decode())))
        apple_mobility.to_csv(filepath, index=False)

    apple_mobility = (pd.read_csv(filepath)
                      .drop('alternative_name', axis=1)
                      .set_index(['geo_type', 'region', 'transportation_type'])
                      .rename_axis("date", axis=1)
                      .stack()
                      .rename('change')
                      .reset_index('date')
                      .assign(date=lambda f: pd.to_datetime(f['date']))
                      .set_index('date', append=True)
                      )

    apple_mobility = (apple_mobility
                      .reset_index()
                      .replace({'UK': 'United Kingdom', 'Republic of Korea': 'South Korea', 'Macao': 'Macau'})  # Only missing is Macao
                      .merge(location_code, left_on='region', right_on='location', how='left')
                     .assign(change=lambda f: f['change'].div(100).sub(1)))

    apple_mobility = apple_mobility.loc[lambda f: f['iso_code'].notna()].set_index(['iso_code', 'date', 'transportation_type'])['change'].unstack()
    return apple_mobility

def fetch_google(isocodes, location_code):
    google_mobility = pd.read_csv('https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv', parse_dates=['date'])

    def safe_match(pat, text):
        match = re.match(pat, text)
        return match.groups()[0] if match else text

    google_mobility.columns = google_mobility.columns.map(lambda col: safe_match("(.*)_percent", col))
    google_mobility = (google_mobility
                .merge(isocodes, left_on='country_region_code', right_on='alpha-2_code', how='left')
                .merge(location_code, left_on='alpha-3_code', right_on='iso_code', how='left'))

    google_mobility = google_mobility.loc[lambda f: f['sub_region_1'].isna()].set_index(['iso_code', 'date']).select_dtypes(float).div(100)
    return google_mobility

# oxford.columns[mask].str.extract(r'(..)_.*', expand=False)

def fetch_oxford():
    oxford = pd.read_csv('https://github.com/OxCGRT/covid-policy-tracker/raw/master/data/OxCGRT_latest.csv')
    oxford.columns = oxford.columns.map(str.lower).str.replace(' ', '_')
    oxford['date'] = pd.to_datetime(oxford['date'], format='%Y%m%d')

    ordinal_columns = oxford.columns[oxford.columns.str.contains("^c._.*_.*")]
    geographic_columns = oxford.columns[oxford.columns.str.contains('^c._flag')]
    strip_measure_name = lambda name: name.split('_')[0]
    measures_ix = dict(zip(ordinal_columns.map(strip_measure_name), ordinal_columns.map(lambda s: '_'.join(s.split('_')[1:]))))

    oxford_long = (oxford
                   .set_index(['countrycode', 'date'])
                   [geographic_columns]
                   .rename(columns=strip_measure_name)
                   .rename_axis('category', axis=1)
                   .stack()
                   .to_frame('flag')
                   .join(oxford.set_index(['countrycode', 'date'])
                         [ordinal_columns]
                         .rename(columns=strip_measure_name)
                         .replace(0, np.nan)
                         .rename_axis('category', axis=1)
                         .stack()
                         .rename('measure'),
                         how='outer'
                        )
                  )

    oxford_long = (oxford_long
                   .rename_axis('variable', axis=1)
                   .stack()
                   .rename('value'))
    return oxford_long

def fetch_all():
    isocodes = fetch_isocodes()
    # acaps = fetch_acaps(isocodes)
    ecdc = fetch_ecdc().set_index(['iso_code', 'date'])
    location_code = ecdc.groupby(['location', 'iso_code']).first().iloc[:, 0].reset_index().iloc[:,:2]
    apple = fetch_apple(location_code)
    google = fetch_google(isocodes, location_code)
    mobility = google.join(apple, how='outer')
    populations_country = (ecdc.assign(population = lambda f: 
                                    f['new_cases']
                                    .div(f['new_cases_per_million'])
                                    .mul(1_000_000))
                        .groupby('iso_code')
                        ['population']
                        .last())

    country_code_lookup = ecdc.reset_index()[['iso_code', 'location']].drop_duplicates().set_index('location').iloc[:, 0]
    country_name_lookup = country_code_lookup.reset_index().set_index('iso_code').iloc[:, 0]

    return {
        "ecdc": ecdc,
        "location_code": location_code, 
        "mobility": mobility, 
        "populations_country": populations_country, 
        "country_code_lookup": country_code_lookup, 
        "country_name_lookup": country_name_lookup
        }
