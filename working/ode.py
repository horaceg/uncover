def build_my_odeint(mobility_data, rtol=1e-5, atol=1e-9, mxstep=500):
    """
    code based on jax.experimental.ode.build_ode to make it work with mobility data
    """
    def dz_dt(z, t, r0, r1, t_inc, t_inf, t_hosp, t_crit, m_a, c_a, f_a, gamma, *alpha):
        s = z[0]
        e = z[1]
        i = z[2]
        r = z[3]
        h = z[4]
        c = z[5]
        
        alpha_ = np.array(alpha)
        int_t = np.array([t]).astype(int)[0]
        rt_u = gamma * r0 * mobility_data[int_t] + r1 * (1 - gamma * mobility_data[int_t])
        rt = np.dot(rt_u, alpha_)
        
        ds = - (rt / t_inf) * i * s
        de = (rt / t_inf) * i * s - (e / t_inc)
        di = e / t_inc - i / t_inf
        dr = m_a * i / t_inf + (1 - c_a) * (h / t_hosp)
        dh = (1 - m_a) * (i / t_inf) + (1 - f_a) * c / t_crit - h / t_hosp
        dc = c_a * h / t_hosp - c / t_crit
        dd = f_a * c / t_crit

        return np.stack([ds, de, di, dr, dh, dc, dd])

    ct_odeint = jax.custom_transforms(
        lambda y0, t, *args: odeint(dz_dt, y0, t, *args, rtol=rtol, atol=atol, mxstep=mxstep))

    v = lambda y0, t, *args: vjp_odeint(dz_dt, y0, t, *args, rtol=rtol, atol=atol, mxstep=mxstep)

    jax.defvjp_all(ct_odeint, v)

    return jax.jit(ct_odeint)
