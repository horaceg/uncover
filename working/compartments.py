
def run_sim_samples(integrator, samples, N, pop_country):
    ts = np.arange(float(N))
    res = []
    corr_samples = {**samples, 
                    **{'alpha': samples['alpha'] / samples['alpha'].sum(axis=1)[:, np.newaxis]},
                    **{'i_init': samples['i_init'] / pop_country}
                   }
    nb_samples = corr_samples['r0'].shape[0]
    z_init = np.array([1. - corr_samples['i_init'], 
                       np.zeros(nb_samples), 
                       corr_samples['i_init'], 
                       np.zeros(nb_samples), 
                       np.zeros(nb_samples), 
                       np.zeros(nb_samples), 
                       np.zeros(nb_samples)
                      ]).T

    arg_names = 'r0, r1, t_inc, t_inf, t_hosp, t_crit, m_a, c_a, f_a'.split(', ')
    args = (corr_samples[name] for name in arg_names)
    alpha = corr_samples['alpha']
    res = vmap(integrator)(z_init, np.repeat(ts[np.newaxis, :], nb_samples, axis=0), *args, *[alpha[:, i] for i in range(alpha.shape[1])])
    
    return res

def plot_compartment(pred_data, true_data, pop_country, times):
    pi = np.percentile(pred_data, (10., 90.), 0)
    
    plt.plot(onp.asarray(times), np.mean(pred_data, axis=0) * pop_country, label='pred')
#     plt.plot(times, true_data, label='true')
    plt.fill_between(onp.asarray(times), pi[0, :] * pop_country, pi[1, :] * pop_country, interpolate=True, alpha=0.3)
    plt.legend()

def plot_hcd(res, pop_country, times, title=None):
    for i, name in enumerate(['hospitalized', 'critical', 'deceased' ], start=5):
        plt.subplots()
#         plot_compartment(res[:, :, -i], hosp_indexed[name].to_numpy(), pop_country)
        plot_compartment(res[:, :, i], np.zeros(res.shape[1]), pop_country, times)
        plt.title(name + ' - ' + title)
    
def plot_seir(res, pop_country, times, title=None):
    for i, name in enumerate(['susceptible', 'exposed', 'infected', 'recovered']):
        plt.subplots()
        plot_compartment(res[:, :, i], np.zeros(res.shape[1]), pop_country, times)
        plt.title(name + ' - ' + title)
