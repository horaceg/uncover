Params = namedtuple('Params', 
                    ['r0', 'r1', 
                     't_inc', 't_inf', 't_hosp', 't_crit', 
                     'm_a', 'c_a', 'f_a']
                   )

# PRIOR_MEANS = Params(
#     r0=3.3,
#     r1=0.5, 
#     t_inc=5.1, 
#     t_inf=2.79, 
#     t_hosp=5.14, 
#     t_crit=5., 
#     m_a=0.85, 
#     c_a=0.2, 
#     f_a=0.33)

PRIOR_MEANS = Params(
    r0=3.28,
    r1=0.2, 
    t_inc=5.6, 
    t_inf=7.9, 
    t_hosp=4., 
    t_crit=14., 
    m_a=0.8, 
    c_a=0.1, 
    f_a=0.35)
