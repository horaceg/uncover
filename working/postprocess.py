import jax.numpy as np
import matplotlib.pyplot as plt
import arviz as az
from rt_mobility import reorder, compute_mu_pi_3, compute_rt_samples


def compute_mu_pi_2(y_pred):
    pop_pred = np.stack([y_pred['hosp'], y_pred['critical'], y_pred['deceased']]).T
    mu = np.mean(pop_pred, 1)
    pi = np.percentile(pop_pred.astype(float), (10., 90.), 1)
    return mu, pi

def compute_mu_pi(y_pred, observation_name='y'):
    pop_pred = y_pred[observation_name]
    mu = np.mean(pop_pred, 0)
    pi = np.percentile(pop_pred.astype(float), (10., 90.), 0)
    return mu, pi

def plot_compartment_results(mu, y_true, times, pi=None):
    plt.plot(times, y_true, "bx", label="true")
    plt.plot(times, mu, "b--", label="pred")
    if pi is not None:
        plt.fill_between(times, pi[0, :], pi[1, :], color="b", alpha=0.3)
    plt.legend()
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.title('Daily deaths')

def plot_daily_cumulated(mu, pi, data, times, name):
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_compartment_results(mu, data, times, pi)
    plt.title('Daily ' + name)

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_compartment_results(np.cumsum(mu), np.cumsum(data), times)
    plt.title('Cumulated ' + name)

# # Control for HCD params
def plot_hcd_results(mu, pi, data, times):
    for i, name in enumerate(['deaths', 'critical', 'hospitalized'], start=1):
        plot_daily_cumulated(mu[:, -i], pi[:, :, -i], data[:, -i], times, name)
        
def plot_forest(inference_data):
    az.plot_forest(inference_data, var_names=['t_inc', 't_inf', 't_hosp', 't_crit'], 
                   kind='forestplot', ridgeplot_overlap=3, combined=True, figsize=(9, 3))
    plt.grid()
    az.plot_forest(inference_data, var_names=['m_a', 'c_a', 'f_a'], figsize=(9, 2))
    plt.grid()
    az.plot_forest(inference_data, var_names=['r0', 'r1'], figsize=(9, 2))
    plt.grid()
    az.plot_forest(inference_data, var_names=['alpha'], figsize=(9, 4))
    plt.grid()

def plot_results(y_pred, samples, all_countries, mask_train, mask_test, all_times, all_deaths, all_mobilities):
    for i, country in enumerate(reorder(all_countries, mask_train, mask_test)):
        mu, pi = compute_mu_pi(y_pred, f'deceased_{i}')
        times = reorder(all_times, mask_train, mask_test)[i]
        plot_daily_cumulated(mu, pi, reorder(all_deaths, mask_train, mask_test)[i], times, 
                            f'deaths - {country}')
        mobility_data = reorder(all_mobilities, mask_train, mask_test)[i]
        rt_pi = compute_rt_samples(samples, mobility_data, times.shape[0])    
        mu, pi = compute_mu_pi_3(rt_pi)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        plt.plot(times, mu)
        plt.xticks(rotation=45)
        plt.fill_between(times, pi[0], pi[1], alpha=0.3, interpolate=True)
        plt.title(f'Rt - {country}')
        
        plt.tight_layout()
