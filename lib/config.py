"""
Loads configuration files.

load_config: main function to load a configuration file.

load_algorithm_kwargs: helper function to load configuration file options related 
                       to the selected Bayesian algorithm.
"""

import sys, os
import configparser
import glob
import numpy as np
from inspect import signature

import lisa

import utils as U
import PT


def load_config(cfg):
    """
    Loads all parameters from the configuration file.
    """
    defaults = {"verb" : 1, "clean" : False, 
                "psg_url" : "http://localhost:3000/api.php", "api_key" : None, 
                "pypsg_timeout" : 120}
    # Load configuration file parameters
    config = configparser.ConfigParser(defaults, allow_no_value=True)
    config.read_file(open(cfg, 'r'))
    for section in config:
        if section != 'DEFAULT':
            conf = config[section]
            basecfg = conf['basecfg']
            outputdir  = conf['outputdir']
            if not os.path.isabs(outputdir):
                outputdir = os.path.abspath(outputdir)
            U.make_dir(outputdir) #ensure it exists
            pnames  = [p        for p in conf['pnames'].split()]
            pmin    = np.array([float(p) for p in conf['pmin']  .split()])
            pmax    = np.array([float(p) for p in conf['pmax']  .split()])
            log     = np.array([p == 'True' for p in conf['log'].split()])
            assert len(pnames) == len(pmin), 'Received '+str(len(pnames))+' parameter names, but received '+\
                                             str(len(pmin))+' parameter minima'
            assert len(pnames) == len(pmax), 'Received '+str(len(pnames))+' parameter names, but received '+\
                                             str(len(pmax))+' parameter maxima'
            assert len(pnames) == len(log),  'Received '+str(len(pnames))+' parameter names, but received '+\
                                             str(len(log))+' booleans for log-scaled parameters'
            try:
                tmp = np.loadtxt(conf['data'])
                wav    = tmp[:,0]
                data   = tmp[:,1]
                uncert = tmp[:,2]
            except:
                wav     = np.array([float(p) for p in conf['wav']   .split()])
                data    = np.array([float(p) for p in conf['data']  .split()])
                uncert  = np.array([float(p) for p in conf['uncert'].split()])
            try:
                filter_files = conf['filters'].split('\n')
                if len(filter_files) == 1:
                    # Directory of files
                    filter_files = sorted(glob.glob(os.path.join(filter_files[0], '*')))
                assert len(filter_files) == data.size, "Received "+str(len(filter_files))+" filter files, but "+\
                                                       "the provided data array has "+str(data.size)+" elements."
                filters = []
                for foo in filter_files:
                    filters.append(np.loadtxt(foo))
            except:
                filters = None
            algorithm = conf['algorithm']
            assert algorithm in U.algorithms, "Invalid Bayesian framework.  Allowed options:\n"+\
                                              str(U.algorithms)
            alg_kwargs = load_algorithm_kwargs(conf, algorithm)
            alg_kwargs['pnames'] = pnames
            ncores    = conf.getint("ncores")
            verb = conf.getint("verb")
            clean = conf.getboolean("clean")
            try:
                PTfuncname = conf['PTfunc']
                PTfunc = getattr(PT, PTfuncname)
                nPT = len(list(signature(PTfunc).parameters.keys())) - 1
                if PTfuncname == 'line':
                    nPT -= 6 # 6 params will remain constant
                    PT_kwargs = {'R_star' : conf.getfloat('R_star'), 
                                 'T_star' : conf.getfloat('T_star'), 
                                 'sma'    : conf.getfloat('sma'),
                                 'grav'   : conf.getfloat('grav')}
                    try:
                        PT_kwargs['T_int'] = conf.getfloat('T_int')
                    except:
                        # Using default
                        pass
                    try:
                        PT_kwargs['T_int_type'] = conf['T_int_type']
                    except:
                        pass
                    # Set those params to be constant
                    PTfunc = functools.partial(PTfunc, **PT_kwargs)
                PTfunc.__name__ = PTfuncname
            except:
                PTfunc = None
                nPT = 0
            try:
                Tmin = conf.getfloat('Tmin')
            except:
                Tmin = None
            try:
                Tmax = conf.getfloat('Tmax')
            except:
                Tmax = None
            psg_url = conf['psg_url'].split('\n')
            if '\n' in psg_url:
                psg_url = [foo.strip() for foo in psg_url.split('\n')]
            elif ',' in psg_url:
                psg_url = [foo.strip() for foo in psg_url.split(',')]
            api_key = conf['api_key']
            pypsg_timeout = conf.getint("pypsg_timeout")
            if api_key is None and 'https://psg.gsfc.nasa.gov/api.php' in psg_url:
                raise ValueError("The main PSG server may only be used with an API key.")
    return basecfg, outputdir, pnames, pmin, pmax, log, \
           wav, data, uncert, filters, algorithm, alg_kwargs, \
           PTfunc, nPT, Tmin, Tmax, \
           ncores, psg_url, api_key, pypsg_timeout, verb


def load_algorithm_kwargs(conf, algorithm):
    """
    Loads config parameters related to the specified Bayesian algorithm

    Inputs
    ------
    conf: ConfigParser section.
    algorithm: string.  Specified Bayesian algorithm.

    Outputs
    -------
    kwargs: dictionary.  Key-value pairs for the parameters to the specified 
                         Bayesian algorithm.
    """
    samp = getattr(lisa.wrappers, algorithm+'_wrapper').Sampler()
    kwargs = {}
    for param in samp.reqpar + samp.optpar:
        if param in ['loglike', 'model', 'perturb', 'pnames', 'prior', 'data', 'uncert']:
            continue
        try:
            # Assume it is a string
            val = conf[param]
        except:
            # parameter not provided - if it is optional, no problem
            assert param in samp.optpar, "Parameter `"+param+"` is required, "+\
                                         "but it is not found in the supplied"+\
                                         " configuration file."
            continue
        if val.lower() in ['true', 't', 'false', 'f']:
            val = bool(val)
        elif not np.any([v.isalpha() for v in val if v != 'e']):
            # Not a simple string - load the value
            try:
                # simple value?
                val = float(val)
                if not val%1:
                    # no remainder -> int
                    val = int(val)
            except:
                # set of values
                val = np.array([float(p) for p in val.split()])
        # add it to dictionary
        kwargs[param] = val
        # clear the value, in case the next parameter is optional and not provided
        del val
    return kwargs


