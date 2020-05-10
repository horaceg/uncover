import os
from collections import namedtuple
from io import StringIO
import re

import requests

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

import jax
from jax.experimental.ode import odeint, vjp_odeint
import jax.numpy as np
from jax.random import PRNGKey
import numpy as onp

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
import arviz as az

NUM_CHAINS = 1
numpyro.set_host_device_count(NUM_CHAINS)
numpyro.enable_x64()

plt.style.use('ggplot')
