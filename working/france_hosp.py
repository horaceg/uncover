ix_france = np.where(np.asarray(all_countries) == 'FRA')[0][0]
times_fr = all_times[ix_france]

hosp = (pd.read_csv('donnees-hospitalieres-covid19-2020-05-07-19h00.csv', sep=';', parse_dates=['jour'])
        .loc[lambda f: f['sexe'] == 0]
        .groupby('jour')
        .sum()
        .drop('sexe', axis=1))

hosp_indexed = hosp.reindex(times_fr).fillna(0)
hosp_indexed.plot()
hosp_indexed.eval('dc.diff() / rea').plot()


BUMP_CRITICAL = abs(hosp_indexed['rea'].diff().min()) + 1
BUMP_HOSP = abs(hosp_indexed['hosp'].diff().min()) + 1