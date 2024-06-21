import numpy as np

# Try to import relevant algorithms
try:
    import dnest4
except:
    dnest4 = None

try:
    import pypolychord
    from pypolychord.priors import UniformPrior
except:
    pypolychord = None
    UniformPrior = None


def dnest4_prior(ndim, pmin, pmax):
    return np.random.uniform(size=ndim) * (pmax - pmin) + pmin


def dnest4_loglikelihood(cube, data, uncert, model):
    ymodel = model(cube)
    loglike = (-0.5 * ((ymodel - data) / uncert)**2).sum()
    return loglike


def dnest4_perturb(coords, ndim, width):
    i = np.random.randint(ndim)
    coords[i] += (width[i])*dnest4.randh()
    # Note: use the return value of wrap, unlike in C++
    coords[i] = dnest4.wrap(coords[i], -0.5*width[i], 0.5*width[i])
    return 0.0


def dream_prior(cube, pmin, pmax):
    cube = cube.copy()
    # Cube begins as [0,1] interval -- scale to [pmin, pmax]
    return cube * (pmax - pmin) + pmin


def dream_loglikelihood(cube, data, uncert, model):
    ymodel = model(cube)
    loglike = (-0.5 * ((ymodel - data) / uncert)**2).sum()
    return loglike


def dynesty_prior(cube, pmin, pmax):
    cube = cube.copy()
    # Cube begins as [0,1] interval -- scale to [pmin, pmax]
    return cube * (pmax - pmin) + pmin


def dynesty_loglikelihood(cube, data, uncert, model):
    ymodel = model(cube)
    loglike = (-0.5 * ((ymodel - data) / uncert)**2).sum()
    return loglike


def emcee_prior(cube, pmin, pmax):
    if np.all(cube <= pmax) and np.all(cube >= pmin):
        return 0.0
    return -np.inf


def emcee_loglikelihood(cube, data, uncert, model, prior):
    lp = prior(cube)
    if not np.isfinite(lp):
        return -np.inf
    ymodel = model(cube)
    loglike = (-0.5 * ((ymodel - data) / uncert)**2).sum()
    return loglike


def multinest_prior(cube, ndim, nparams, pmin, pmax):
    # Cube begins as [0,1] interval -- scale to [pmin, pmax]
    for i in range(ndim):
        cube[i] = cube[i] * (pmax[i] - pmin[i]) +  pmin[i]
    return cube


def multinest_loglikelihood(cube, ndim, nparams, data, uncert, model):
    ymodel = model(cube)
    loglike = (-0.5 * ((ymodel - data) / uncert)**2).sum()
    return loglike


def polychord_prior(cube, pmin, pmax):
    return UniformPrior(pmin, pmax)(cube)


def polychord_loglikelihood(cube, data, uncert, model):
    ymodel = model(cube)
    loglike = (-0.5 * ((ymodel - data) / uncert)**2).sum()
    return loglike, []


def ultranest_prior(cube, pmin, pmax):
    cube = cube.copy()
    # Cube begins as [0,1] interval -- scale to [pmin, pmax]
    for i in range(cube.shape[-1]):
        cube[:,i] = cube[:,i] * (pmax[i] - pmin[i]) + pmin[i]
    return cube


def ultranest_loglikelihood(cube, data, uncert, model):
    ymodel = model(cube)
    loglike = (-0.5 * ((ymodel - data) / uncert)**2).sum(-1)
    return loglike


