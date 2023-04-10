from functools import partial
from scipy.stats import *
from random import randint, uniform
import pandas as pd

expected_value = 150

distributions = {
    'Normal': partial(norm.rvs, loc=expected_value, scale=1),
    'Gamma': partial(gamma.rvs, a=expected_value ** 2, scale=1),
    'T': partial(t.rvs, df=expected_value, loc=expected_value, scale=1),
    'Chisq': partial(chi2.rvs, df=expected_value, loc=expected_value / 16, scale=1),
    'Betaprime': partial(betaprime.rvs, a=expected_value, b=expected_value, loc=expected_value, scale=1),
    'Weibull_min': partial(weibull_min.rvs, c=expected_value, loc=expected_value, scale=1),
    'Weibull_max': partial(weibull_max.rvs, c=expected_value, loc=expected_value, scale=1),
    'Burr12': partial(burr12.rvs, c=expected_value, d=expected_value, loc=expected_value, scale=1),
    'F': partial(f.rvs, dfn=expected_value, dfd=expected_value, loc=expected_value, scale=1),
    'Alpha': partial(alpha.rvs, a=0.015, loc=expected_value, scale=1),
    'Genlogistic': partial(genlogistic.rvs, c=expected_value, loc=expected_value, scale=1),
    'Johnsonsb': partial(johnsonsb.rvs, a=expected_value, b=expected_value, loc=expected_value, scale=1),
    'Johnsonsu': partial(johnsonsu.rvs, a=expected_value, b=expected_value, loc=expected_value, scale=1),
    'Dgamma': partial(dgamma.rvs, a=expected_value, loc=expected_value * 2, scale=1)
}


# works as wanted!
def randomDistributionData(distname):
    scale = 10
    if distname == "Gamma" or distname == 'Dgamma':
        scale = uniform(0.02, 0.04)
    else:
        scale = randint(5, 20)
    print()
    return pd.DataFrame(distributions[distname](scale=scale, size=200), columns=["demand"])


dist_ppf_dict = {
    'Gamma': gamma.cdf,
    'Normal': norm.cdf,
    'T': t.cdf,
    'Chisq': chi2.cdf,
    'Betaprime': betaprime.cdf,
    'Weibull_min': weibull_min.cdf,
    'Weibull_max': weibull_max.cdf,
    'Burr12': burr12.cdf,
    'F': f.cdf,
    'Alpha': alpha.cdf,
    'Genlogistic': genlogistic.cdf,
    'Johnsonsb': johnsonsb.cdf,
    'Johnsonsu': johnsonsu.cdf,
    'Dgamma': dgamma.cdf
}

# idea:
# f.getparams gives you the the params needed to be for the CDF func
# luo tänne func johon voi pistää getparamsit ja palauttaa funktion tuloksen sijainnissa x
# pääskripti: generoi kertymäfunktion arvot jotka voidaan plotata.

def distcdfvalue(x, params, correctdist):
    func = dist_ppf_dict[correctdist]
    correctdist = correctdist.lower()
    params = params[correctdist]
    return func(x, **params)