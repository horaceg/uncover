
def run_sim_samples(integrator, samples):
    ts = np.arange(float(data.shape[0]))
    res = []
    for i in range(samples['c_a'].shape[0]):
        post_params = dict()
        for param in 'r0, r1, t_inc, t_inf, t_hosp, t_crit, m_a, c_a, f_a, alpha'.split(', '):
            post_params[param] = samples[param][i]
    #         post_params[param] = inference_data.posterior[param].values[0, -1]

        i_init = samples['i_init'][i]
        i_init /= pop_country
        z_init = np.array([1. - i_init, 0., i_init, 0., 0., 0., 0.])
        args = list(post_params.values())[:-1]
        
        alpha = post_params['alpha']
        alpha /= np.sum(alpha)
        
        sim_res = integrator(z_init, ts, *args, *alpha)
        res.append(sim_res)

    res = np.stack(res)
    return res

def plot_compartment(pred_data, true_data):
    pi = np.percentile(pred_data, (10., 90.), 0)
    
    plt.plot(times, np.mean(pred_data, axis=0) * pop_country, label='pred')
    plt.plot(times, true_data, label='true')
    plt.fill_between(times, pi[0, :] * pop_country, pi[1, :] * pop_country, interpolate=True, alpha=0.3)
    plt.legend()

def plot_hcd(res, hosp_indexed):
    for i, name in enumerate(['dc', 'rea', 'hosp'], start=1):
        plt.subplots()
        plot_compartment(res[:, :, -i], hosp_indexed[name].to_numpy())
        plt.title(name)
    
def plot_seir(res):
    for i, name in enumerate(['susceptible', 'exposed', 'infected', 'recovered']):
        plt.subplots()
        plot_compartment(res[:, :, i], np.zeros(res.shape[1]))
        plt.title(name)
