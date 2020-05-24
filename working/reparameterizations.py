import numpyro.distributions as dist

def reparametrize_beta(mean, sample_size):
#     v = numpyro.sample(f'sample_size_{i}', dist.Gamma(*reparametrize_gamma(10., 5)))
    alpha = mean * sample_size
    beta = (1 - mean) * sample_size
    return alpha, beta 

def reparametrize_gamma(mean, std):
    var = std ** 2
    alpha = mean ** 2 / var
    beta = mean / var
    return alpha, beta

def Gamma_2(mean, std):
    return dist.Gamma(*reparametrize_gamma(mean, std))

def Beta_2(mean, sample_size):
    return dist.Beta(*reparametrize_beta(mean, sample_size))
