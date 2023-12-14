import sys, os
import configparser
import copy
import functools
import multiprocessing as mp
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), 'lib'))
import config as C
import retrieval as R
import utils  as U


def main(cfg):
    """
    Function to be used when executing the software via a configuration file.

    Inputs
    ------
    cfg: string.  Path/to/configuration file.
    """
    basecfg, outputdir, pnames, pmin, pmax, log, \
           wav, data, uncert, filters, algorithm, alg_kwargs, \
           PTfunc, nPT, Tmin, Tmax, \
           ncores, psg_url, api_key, pypsg_timeout, verb = C.load_config(cfg)
    retrieved = driver(basecfg, outputdir, pnames, pmin, pmax, log, 
                       wav, data, uncert, filters, algorithm, alg_kwargs, 
                       PTfunc, nPT, Tmin, Tmax, 
                       ncores, psg_url, api_key, pypsg_timeout, verb)
    return retrieved


def driver(basecfg, outputdir, pnames, pmin, pmax, log, 
           wav, data, uncert, filters, algorithm, alg_kwargs, 
           PTfunc, nPT, Tmin, Tmax, 
           ncores=1, psg_url='http://localhost:3000/api.php', 
           api_key=None, pypsg_timeout=60, verb=1):
    """
    Main driver for the software.

    Inputs
    ------
    basecfg: string. Base PSG configuration file.
    outputdir: string. Output directory to save configuration files/grids (if applicable).
    pnames: list, strings. PSG parameter names to comprise the grid.
    pmin: array, floats. Minimum of each grid parameter.
    pmax: array, floats. Like `pmin`, but the maxima.
    log: array, bools. Determines which parameters are sampled logarithmically.
    wav: array, floats.  Wavelengths that correspond to the data.
    data: array, floats. Target spectrum.
    uncert: array, floats. Uncertainty in the target spectrum.
    filters: list or None.  If None, assumes the forward model's wavelengths exactly corresponds to the 
                            data's, with no band integration.
                            If a list, must match the number of data points.  Each element in the list 
                            must be a 2D array, where `filters[i][:,0]` specifies wavelengths and 
                            `filters[i][:,1]` specifies the filter's transmission at that wavelength.
    algorithm: string. Bayesian inference algorithm to use.
    alg_kwargs: dictionary.  Key-value pairs of parameters for the chosen `algorithm`.
    PTfunc: function.  Pressure-temperature profile to use.  
                       Use None if it isn't varied.
    nPT: int.  Number of parameters in `PTfunc`.
    Tmin: float. Maximum allowed temperature in thermal profile.  Use None for no limit.
                 Default: None
                 Note: If you have not carefully chosen your parameter space, this may 
                 lead to invalid temperatures, which will crash the program.
    Tmax: float. Minimum allowed temperature in thermal profile.  Use None for no limit.
                 Default: None
                 Note: Same as above.
    ncores: int. Number of parallel cores to use when building and running configuration files.
                 Default: 1 (Recommended: 1/4 of desired total thread usage)
    psg_url: str. URL to use for PSG API calls.
                  If using the main PSG online API, you must provide an API key.
                  Default: local installation.
    api_key: str or None. If string, a key to bypass normal API limit.
                          If you don't know what this is, then don't worry about it.
                          Default: None
    pypsg_timeout: int.  Maximum number of seconds to wait for pypsg to receive a response 
                         from PSG before raising a timeout error.
    verb: int. Controls verbosity of output.
               Default: 1 (recommended)
               Use 0 for minimal output.
               Use 6 or more for all detail.
    """
    # Initialize the retrieval
    if verb:
        time_start = U.tprint("Beginning retrieval\n")
    retriever = R.pyspgRetrieval(basecfg, pnames, pmin, pmax, log, 
                                 wav, data, uncert, filters, algorithm, alg_kwargs, 
                                 PTfunc, nPT, Tmin, Tmax, 
                                 psgserver=psg_url, api_key=api_key, pypsg_timeout=pypsg_timeout, 
                                 ncores=ncores, outputdir=outputdir, verb=verb)

    print("\n")
    time_end = U.tprint("Retrieval complete")
    tot_time = U.time_elapsed(time_start, time_end)
    print("Total time elapsed:", tot_time)


if __name__ == '__main__':
    main(sys.argv[1])


