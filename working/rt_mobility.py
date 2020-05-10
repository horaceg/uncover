
def reorder(d, mask_train, mask_teset):
    return mask_ix(d, mask_train) + mask_ix(d, mask_test)

def compute_mu_pi_3(data):
    mu = np.mean(data, axis=0)
    pi = np.percentile(data, (10, 90), axis=0)
    return mu, pi

def compute_rt(ts, mobility_data, r0, r1, alpha):    
    rt_u = r0 * (1 + mobility_data[ts]) - r1 * mobility_data[ts]
    rt = np.dot(rt_u, alpha)
    return rt

def compute_rt_samples(samples, mobility_data, N):
    corr_samples = {**samples, **{'alpha': samples['alpha'] / samples['alpha'].sum(axis=1)[:, np.newaxis]}}
    mu, pi = dict(), dict()
    for name in 'r0', 'r1', 'alpha':
        mu[name], pi[name] = compute_mu_pi_3(corr_samples[name])

    ts = np.arange(N)

    rt_pi = onp.empty((corr_samples['r0'].shape[0], times.shape[0]))
    for i in range(corr_samples['r0'].shape[0]):
        rt_pi[i] = compute_rt(ts, mobility_data, 
                              corr_samples['r0'][i], 
                              corr_samples['r1'][i], 
                              corr_samples['alpha'][i])
    
    return rt_pi
