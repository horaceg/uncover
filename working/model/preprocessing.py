
ix = pd.IndexSlice

def diff_pop(cumulative, pop_country):
    daily = np.hstack((np.array([0.]), np.diff(cumulative * pop_country)))
    daily += 1
    return daily

def make_dataset(subset, days_before_deaths):
    deaths_subset = subset.loc[ix[:, 'total_deaths']]

    ten_deaths_date = deaths_subset.gt(10).idxmax()
    begin_date = ten_deaths_date - pd.Timedelta(days_before_deaths, unit='days')

    total_deaths = deaths_subset.loc[begin_date:].to_numpy()
    times = deaths_subset.loc[begin_date:].index.map(onp.datetime64).to_numpy()

    mobility = (subset
                .loc[ix[:, 'transit']]
                .to_frame('transit'))

    for cat in ['walking', 'driving', 
                'grocery_and_pharmacy', 'retail_and_recreation', 'workplaces', 'transit_stations']:
        mobility = mobility.join(subset.loc[ix[:, cat]].rename(cat))

    mobility = (mobility
                .reindex(deaths_subset.loc[begin_date:].index)
                .fillna(method='ffill')
                .fillna(method='bfill')
                #.fillna(mobility.iloc[-1])
                .rolling('7d').mean())
    
    mobility_data = np.asarray(mobility.to_numpy())
    return total_deaths, times, mobility_data

def make_all_datasets(full, populations_country, country_names, days_before_deaths):
    all_countries = []
    all_populations = []
    all_mobilities = []
    all_deaths = []
    all_times = []
    for country, subset in full.groupby('location'):
        if country not in country_names:
            continue
        subset = subset.reset_index('location', drop=True)
        all_countries.append(country)
#         try:
        total_deaths, times, mobility_data = make_dataset(subset, days_before_deaths)
#         except KeyError:
#             continue
        pop_country = populations_country.loc[country]
        daily_deaths = diff_pop(total_deaths / pop_country, pop_country)
        all_populations.append(pop_country)
        all_mobilities.append(mobility_data)
        all_deaths.append(daily_deaths)
        all_times.append(times)
    return all_countries, all_populations, all_mobilities, all_deaths, all_times
