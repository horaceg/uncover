
from jax import vmap, jit
import jax.numpy as np

from train_test_split import mask_ix

def reorder(d, mask_train, mask_test):
    return mask_ix(d, mask_train) + mask_ix(d, mask_test)

def compute_mu_pi_3(data):
    mu = np.mean(data, axis=0)
    pi = np.percentile(data, (10, 90), axis=0)
    return mu, pi

def compute_rt_samples(samples, mobility_data, N):
    corr_samples = {**samples, 
                    **{'alpha': samples['alpha'] / samples['alpha'].sum(axis=1)[:, np.newaxis]}}
    mu, pi = dict(), dict()
    for name in 'r0', 'r1', 'alpha':
        mu[name], pi[name] = compute_mu_pi_3(corr_samples[name])

    ts = np.arange(N)
    
    mobility_data = np.asarray(mobility_data)
    
    @jit
    def compute_rt(r0, r1, alpha):    
        rt_u = r0 * (1. + mobility_data[ts]) - r1 * mobility_data[ts]
        rt = np.dot(rt_u, alpha)
        return rt

    rt_pi = vmap(compute_rt)(corr_samples['r0'], corr_samples['r1'], corr_samples['alpha'])
    return rt_pi
