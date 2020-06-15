import numpyro
import numpyro.distributions as dist
import jax.numpy as np

from reparameterizations import Gamma_2, Beta_2
from prior_means import PRIOR_MEANS
from preprocessing import diff_pop
from ode import build_my_odeint


def make_target_dist(psi_h, psi_c, psi, 
                     daily_hosp, daily_critical, daily_deaths, 
                     bump_hosp, bump_critical, N):
    
    target_dist = dist.GammaPoisson(
                   np.array([psi_h, psi_c, psi]),
                   rate=np.stack([psi_h / (daily_hosp + bump_hosp), 
                                  psi_c / (daily_critical + bump_critical), 
                                  psi / daily_deaths]
                                ).T
    )

    reloc = dist.transforms.AffineTransform(
        loc=np.stack([
            - bump_hosp,
            - bump_critical,
            np.zeros(N)]
        ).T,
        scale=1.)

    reloc_target_dist = dist.TransformedDistribution(target_dist, [reloc])
    return reloc_target_dist

def sample_parameters(nb_mobilities):
    kappa0 = numpyro.sample('kappa0', dist.TruncatedNormal(0, 0., 0.5))
    kappa1 = numpyro.sample('kappa1', dist.TruncatedNormal(0, 0, 0.5))
    r0 = numpyro.sample('r0', dist.TruncatedNormal(0, PRIOR_MEANS.r0, kappa0))
    r1 = numpyro.sample('r1', dist.TruncatedNormal(0, PRIOR_MEANS.r1, kappa1))

    alpha = numpyro.sample('alpha', Gamma_2(1., 0.5), sample_shape=(nb_mobilities,))
    alpha /= np.sum(alpha)
    
    t_inc = numpyro.sample('t_inc', Gamma_2(PRIOR_MEANS.t_inc, .86))
    t_inf = numpyro.sample('t_inf', Gamma_2(PRIOR_MEANS.t_inf, 3.))
    t_hosp = numpyro.sample('t_hosp', Gamma_2(PRIOR_MEANS.t_hosp, 3.))
    t_crit = numpyro.sample('t_crit', Gamma_2(PRIOR_MEANS.t_crit, 3.))
    
    sample_size_m = numpyro.sample('sample_size_m', Gamma_2(7., 2))
    sample_size_c = numpyro.sample('sample_size_c', Gamma_2(7., 2))
    sample_size_f = numpyro.sample('sample_size_f', Gamma_2(7., 2))
    m_a = numpyro.sample('m_a', Beta_2(PRIOR_MEANS.m_a, sample_size_m))
#     m_a = 0.8
    c_a = numpyro.sample('c_a', Beta_2(PRIOR_MEANS.c_a, sample_size_c))
    f_a = numpyro.sample('f_a', Beta_2(PRIOR_MEANS.f_a, sample_size_f))
    
    params = (r0, r1, t_inc, t_inf, t_hosp, t_crit, m_a, c_a, f_a)
    
    return params, alpha
    

def sample_compartment_init(pop_country, country_name=None):
#     tau = numpyro.sample('tau', dist.Exponential(0.03))
#     kappa_i0 = numpyro.sample('kappa_i0', dist.TruncatedNormal(0, 0., 0.5))
    i_init = numpyro.sample(f'i_init_{country_name}', 
                            dist.TruncatedNormal(loc=50., scale=10.)
#                             Gamma_2(50, 10.))
#                             dist.Exponential(1. / tau)
                           )
    i_init /= pop_country
    z_init = np.array([1. - i_init, 0., i_init, 0., 0., 0., 0.])
    return z_init
    
def model(seirhcd_int, N, pop_country, y=None, compartments='d', nb_mobilities=1):
    ts = np.arange(float(N))
    params, alpha = sample_parameters(nb_mobilities=nb_mobilities)
    z_init = sample_compartment_init(pop_country)
    
    z = seirhcd_int(z_init, ts, *params, *alpha)
    
    daily_deaths = diff_pop(z[:, -1], pop_country)
    psi = numpyro.sample('psi', dist.TruncatedNormal(scale=5.))

    if compartments == 'd':
        numpyro.sample('deceased', dist.GammaPoisson(psi, rate=psi / daily_deaths), obs=y)

    elif compartments == 'hcd':
        daily_hosp = diff_pop(z[:, -3], pop_country)
        daily_critical = diff_pop(z[:, -2], pop_country)

        hosp = z[:, -3] * pop_country
        critical = z[:, -2] * pop_country

        hosp_m1 = np.hstack(([0.],  hosp[:-1]))
        critical_m1 = np.hstack(([0.],  critical[:-1]))

        bump_hosp = np.min(np.stack([hosp_m1, BUMP_HOSP * np.ones(N)]), axis=0)
        bump_critical = np.min(np.stack([critical_m1, BUMP_CRITICAL * np.ones(N)]), axis=0)

        psi_h = numpyro.sample('psi_h', dist.TruncatedNormal(scale=5.))
        psi_c = numpyro.sample('psi_c', dist.TruncatedNormal(scale=5.))
        
        target_dist = make_target_dist(psi_h, psi_c, psi, 
                                       daily_hosp, daily_critical, daily_deaths, 
                                       bump_hosp, bump_critical, N)
        
        numpyro.sample('y', target_dist, obs=y)


def multi_model(all_mobilities, all_populations, observations=None):
    nb_mobilities = all_mobilities[0].shape[1]
    params, alpha = sample_parameters(nb_mobilities=nb_mobilities)
    psi = numpyro.sample('psi', dist.TruncatedNormal(scale=5.))
    i_init = numpyro.sample('i_init', dist.TruncatedNormal(loc=50, scale=10))
    
    for country in range(len(all_mobilities)):
        mobility_data = all_mobilities[country]
        pop_country = all_populations[country]
        seirhcd_int = build_my_odeint(mobility_data, 
#                                       atol= 1. / pop_country
                                     )
        if observations is not None:
            y = observations[country]
        else:
            y = None

        ts = np.arange(float(mobility_data.shape[0]))

        z_init = np.array([1. - i_init / pop_country, 0., i_init / pop_country, 0., 0., 0., 0.])
        z = seirhcd_int(z_init, ts, *params, *alpha)

        daily_deaths = diff_pop(z[:, -1], pop_country)

        numpyro.sample(f'deceased_{country}', dist.GammaPoisson(psi, rate=psi / daily_deaths), obs=y)
    
#     vmap(sample_deceased)(np.arange(0, len(all_mobilities), step=1, dtype=int))
#     jax.lax.fori_loop(0, len(all_mobilities), lambda i, val: sample_deceased(i), 0)
