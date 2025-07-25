import sys, os
import copy
import ctypes
import functools
import gc
import glob
import itertools
import time
import h5py
import multiprocessing as mp
import multiprocessing.managers as mpm
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as si

import pypsg

import alg_funcs as AF
import utils as U


class pyspgRetrieval:
    """
    A class to set up a Bayesian retrieval using PSG.
    """
    def __init__(self, basecfg, pnames, pmin, pmax, log, 
                       wav, data, uncert, filters, algorithm, alg_kwargs, 
                       PTfunc=None, nPT=0, Tmin=None, Tmax=None, 
                       psgserver=['http://localhost:3000/api.php'], api_key=None, pypsg_timeout=120,
                       pool=None, outputdir='./output', verb=1):
        """
        Inputs
        ------
        basecfg: string. Base PSG configuration file.
        pnames: list, strings. PSG parameter names that are free parameters.
        pmin: array, floats. Minimum of each free parameter.
        pmax: array, floats. Like `pmin`, but the maxima.
        log: array, bools. Determines which parameters are sampled logarithmically.
        wav: array, floats.  Wavelengths that correspond to the data.
        data: array, floats. Target spectrum.
        uncert: array, floats. Uncertainty in the target spectrum.
        algorithm: string. Bayesian inference algorithm to use.
        alg_kwargs: dictionary.  Key-value pairs of parameters for the chosen `algorithm`.
        PTfunc: function.  Pressure-temperature profile to use.
                           Default: None (PT profile is not changed)
        nPT: int.  Number of parameters for `PTfunc`.
                   Default: 0
        Tmin: float. Maximum allowed temperature in thermal profile.  Use None for no limit.
                     Default: None
        Tmax: float. Minimum allowed temperature in thermal profile.  Use None for no limit.
                     Default: None
        psgserver: string. URL to the PSG server, whether local or remote.
                           Default: http://localhost:3000/api.php
        api_key: string. Key for API access, if needed.
                         Default: None
        pypsg_timeout: int. Maximum number of seconds to wait for pypsg to receive a response 
                            from PSG before raising a timeout error.
        pool: multiprocessing pool object.  Pool to be used for parallel evaluations.
        outputdir: string. Output directory to save configuration files/grids (if applicable).
                        Default: current directory.
        verb: int. Controls verbosity of output.
                   Default: 1

        """
        # Sanity checks
        assert len(pmin) == len(pnames), "`pmin` must be an array of minima of the same size as `pnames`."
        assert len(pmax) == len(pnames), "`pmax` must be an array of maxima of the same size as `pnames`."
        assert len(log)  == len(pnames), "`log` must be an array of booleans of the same size as `pnames`."

        # Read in arguments
        self.basecfg   = basecfg
        self.pnames    = pnames
        self.pmin      = np.array(pmin)
        self.pmax      = np.array(pmax)
        self.log       = np.array(log)
        self.wav       = wav
        self.data      = data
        self.uncert    = uncert
        self.filters   = filters
        assert algorithm in U.algorithms, "Invalid Bayesian algorithm.  Available options:\n"+\
                                          str(U.algorithms)
        self.algorithm  = algorithm
        self.alg_kwargs = alg_kwargs
        self.PTfunc     = PTfunc
        self.nPT        = nPT
        self.Tmin       = Tmin
        self.Tmax       = Tmax
        self.outputdir  = outputdir
        self.verb       = verb

        # Ensure output directory exists
        U.make_dir(self.outputdir)

        # Check if config needs to be re-calculated before submitting to PSG
        if any(x in self.pnames for x in U.kempton_keys):
            self.watm = 'y'
        else:
            self.watm = None

        # Initialize connection with PSG
        self.psg = [pypsg.PSG(server_url=psgserver[isrv], api_key=api_key, timeout_seconds=pypsg_timeout, 
                              default_config=os.path.abspath(self.basecfg)) 
                    for isrv in range(len(psgserver))]
        self.config = copy.deepcopy(self.psg[0].default_config)
        self.wavelength = self.run_forward(self.config)['spectrum'][:,0]

        if self.filters is not None:
            if self.verb:
                U.tprint("Preparing filters...")
            # Interpolate the filters onto the same wavelength grid
            for i in range(len(self.filters)):
                filt_wavl = self.filters[i][:,0]
                filt_tran = self.filters[i][:,1]
                f = si.interp1d(filt_wavl, filt_tran, fill_value=0., bounds_error=False)
                self.filters[i] = np.zeros((self.wavelength.size, 2))
                self.filters[i][:,0] =   self.wavelength
                self.filters[i][:,1] = f(self.wavelength)
                self.filters[i][:,1] /= self.filters[i][:,1].max() # Normalize the filters
        else:
            if self.verb:
                U.tprint("No filters provided.  Assuming the data's wavelength grid matches the forward model's wavelength grid exactly.")

        # Load pressure grid, if needed
        if any(x in self.pnames for x in U.pt_keys):
            if PTfunc is None:
                raise ValueError("No PTfunc specified, but free parameters include PT parameters.")
            nlayer = int(self.config['ATMOSPHERE-LAYERS'])
            self.pressure = np.zeros(nlayer, dtype=float) + np.nan
            for k in range(nlayer):
                key = 'ATMOSPHERE-LAYER-' + str(k+1)
                layer = self.config[key].split(',')
                self.pressure[k] = float(layer[0])
            assert not np.isnan(self.pressure).sum(), "Error loading pressure grid."
            # Ensure correct order of parameters
            self.iPT = np.zeros(self.nPT, dtype=int)
            if PTfunc.__name__ == 'adiabat':
                self.iPT[0] = self.pnames.index('PT-GAMMA')
                self.iPT[1] = self.pnames.index('PT-T0')
                self.iPT[2] = self.pnames.index('PT-LOGP0')
            elif PTfunc.__name__ == 'line':
                self.iPT[0] = self.pnames.index('PT-KAPPA')
                self.iPT[1] = self.pnames.index('PT-GAMMA1')
                self.iPT[2] = self.pnames.index('PT-GAMMA2')
                self.iPT[3] = self.pnames.index('PT-ALPHA')
                self.iPT[4] = self.pnames.index('PT-BETA')
            elif PTfunc.__name__ == 'spt':
                self.iPT[0] = self.pnames.index('PT-T0')
                self.iPT[1] = self.pnames.index('PT-DT-BOT-32')
                self.iPT[2] = self.pnames.index('PT-DT-32-10')
                self.iPT[3] = self.pnames.index('PT-DT-10-0')
                self.iPT[4] = self.pnames.index('PT-DT-0-1')
                self.iPT[5] = self.pnames.index('PT-DT-1-01')
                self.iPT[6] = self.pnames.index('PT-DT-01-001')
                self.iPT[7] = self.pnames.index('PT-DT-001-TOP')
            # Ensure that `iPT` is the integers in [0, nPT)
            if self.nPT > 1:
                self.iPT -= self.iPT.min()
                for n in range(1, self.nPT):
                    nmin = self.iPT[self.iPT >= n].min()
                    if nmin != n:
                        self.iPT[self.iPT >= n] -= nmin - n
            self.iPT = self.iPT.tolist()

        # Set up TEA grid interpolator if using TEA chemistry model
        if any(x in self.pnames for x in U.tea_keys):
            # Sanity checks
            assert self.watm != 'y', "Cannot use both the Kempton and TEA chemistry models in the same retrieval."
            if self.PTfunc is not None:
                for x in self.pnames:
                    if x in U.pt_keys:
                        assert self.pnames.index(x) < self.pnames.index(U.tea_keys[0]), \
                               "PT parameters must come before TEA parameters."
            tea_data = np.load(os.path.join(os.path.dirname(__file__), '..', 'data', 'TEA_grid.npz'))
            points = [tea_data['press'], tea_data['temp'], tea_data['MH'], tea_data['C2O']]
            grid = np.stack([tea_data[mol] for mol in U.tea_molecules], axis=-1)
            self.tea = si.RegularGridInterpolator(points, grid)
            self.tea_pmin = tea_data['press'].min()
            self.tea_pmax = tea_data['press'].max()
            self.tea_tmin = tea_data['temp' ].min()
            self.tea_tmax = tea_data['temp' ].max()
            # Set the atmosphere molecules
            self.config['ATMOSPHERE-LAYERS-MOLECULES'] = ','.join(U.tea_molecules)
        else:
            self.tea = None
            self.tea_pmin = None
            self.tea_pmax = None
            self.tea_tmin = None
            self.tea_tmax = None

        # Set up functions for the inference and add to kwargs
        if algorithm in ['dnest4', 'dream', 'dynesty', 'emcee', 'multinest', 'polychord', 'ultranest']:
            self.prior = functools.partial(getattr(AF, algorithm+'_prior'), pmin=self.pmin, pmax=self.pmax)
            loglike_kwargs = {'data' : self.data, 
                              'uncert' : self.uncert, 
                              'model' : self.forward_model}
            if algorithm == 'emcee':
                loglike_kwargs['prior'] = self.prior
            self.loglike = functools.partial(getattr(AF, algorithm+'_loglikelihood'), **loglike_kwargs)
            self.alg_kwargs['prior'] = self.prior
            self.alg_kwargs['loglike'] = self.loglike
            if algorithm == 'dnest4':
                self.perturb = functools.partial(getattr(AF, algorithm+'_perturb'), 
                                                 ndim=self.pnames.size, width=self.pmax - self.pmin)
                self.alg_kwargs['perturb'] = self.perturb
        else:
            # DEMC and snooker
            self.alg_kwargs['data']   = self.data
            self.alg_kwargs['uncert'] = self.uncert

        # Ensure there is a worker pool
        if pool is not None:
            self.pool = pool
        else:
            raise ValueError("Must supply a multiprocessing pool for workers.")

        self.alg_kwargs['model'] = self.forward_model

    def __getstate__(self):
        """
        Override normal __getstate__ functionality so that the object is pickle-able
        """
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict

    def forward_model(self, params):
        """
        Helper method that updates the configuration file and runs PSG for it.  
        """
        params = np.atleast_2d(params)
        # TODO: fix
        # Allow workers to manually garbage collect to avoid memory issues
        # It is unclear why the memory growth happens - it is stepwise, it happens at irregular intervals, 
        # and I can't find what is actually causing that memory growth
        # Manual garbage collection solves this but adds some overhead...worth it for the memory savings, 
        # but it would be best to fix the underlying memory growth issue
        gc.disable()
        res = self.pool.map(self._worker, [params[i] for i in range(params.shape[0])])
        gc.enable()
        return res

    def _worker(self, params, manual_gc=True):
        """
        Worker function for calculating the forward model in parallel.
        
        Inputs
        ------
        params: array, floats.  Retrieval model parameters.
        manual_gc: bool.  Determines whether to manually garbage collect.
        """
        # Determine which PSG instance to use
        # -1 because worker identity is 1-based
        try:
            worker_id = (mp.current_process()._identity[0] - 1) % len(self.psg)
        except:
            worker_id = 0
        self.update_config(params)
        res = self.run_forward(self.config, worker_id)['spectrum']
        if self.filters is None:
            ret = res[:,1] # Keep full resolution spectrum
        else:
            # Band integrate for each filter
            ret = np.zeros(len(self.filters), dtype=float)
            for i in range(len(self.filters)):
                inonzero = np.where(self.filters[i][:,1] > 0)[0]
                ret[i] = np.sum(res[:,1][inonzero] * self.filters[i][:,1][inonzero])
                if self.config['GENERATOR-RADUNITS'] in ['rel', 'rif', 'V']:
                    # Average over the filter
                    ret[i] /= inonzero.size
        if manual_gc:
            gc.collect()
        return ret

    def run_forward(self, cfg, wid=0):
        """
        Runs the PSG configuration file, which may involve retrieving an intermediate configuration file.

        Inputs
        ------
        cfg: string, or dictionary.  PSG configuration file to be submitted via pypsg.
        wid: int. Worker ID.  Determines which PSG to submit the config.
        """
        if self.verb > 5:
            start = time.time()
        cfg = copy.deepcopy(cfg)
        if self.watm == 'y':
            modcfg = self.psg[wid].run(cfg, watm=self.watm, otype='cfg')['config']
            # Update the original config
            modcfg = self.psg[wid].config_str_to_dict(modcfg)
            for k in modcfg.keys():
                if k in U.watm_config_keys:
                    cfg[k] = modcfg[k]
        # Run the config
        result = self.psg[wid].run(cfg)
        if self.verb > 5:
            end = time.time()
            _ = U.tprint("Running config took {:.2g} sec".format(end-start))
        return result

    def update_config(self, params):
        """
        Updates the configuration file with the requested parameters.

        Inputs
        ------
        params: array, floats. Values for each grid parameter.
        """
        if self.verb > 5:
            start = time.time()
        for i in range(len(params)):
            if self.log[i]:
                p = 10**params[i]
            else:
                p = params[i]
            if self.pnames[i] in U.molecules:
                # Update atmospheric profile for this molecule
                self.update_molecule(self.pnames[i], p)
            elif self.pnames[i] in U.kempton_keys:
                # Update atmospheric layers w/ Kempton chemistry model
                self.update_kempton(self.pnames[i], p)
            elif self.pnames[i] in U.tea_keys:
                # Update atmospheric layers w/ TEA chemistry model
                if i < len(params) - 1:
                    if self.pnames[i+1] in U.tea_keys:
                        self.update_tea(self.pnames[i:i+2], params[i:i+2])
                elif i > 0:
                    if self.pnames[i-1] in U.tea_keys:
                        continue
                else:
                    raise ValueError("When using TEA chemistry models, both " + \
                                     "TEA parameters must occur consecutively.")
            elif self.pnames[i] in U.pt_keys:
                # Upate T(p) profile
                if i > 0:
                    if self.pnames[i-1] in U.pt_keys:
                        continue
                for n in range(self.nPT):
                    assert self.pnames[i+n] in U.pt_keys, "PT parameters must occur consecutively."
                self.update_temperature(params[i:i+self.nPT])
            else:
                # Update config parameter
                self.config[self.pnames[i]] = p
        if self.verb > 5:
            end = time.time()
            _ = U.tprint("Updating config took {:.2g} sec".format(end-start))

    def update_molecule(self, mol, val):
        """
        Updates values for a molecule in the configuration file

        Inputs
        ------
        mol: string. Molecule names.
        val: float.  Value for that molecule.
        """
        # Find the index of the species
        imol = self.config['ATMOSPHERE-LAYERS-MOLECULES'].split(',').index(mol) + 2 #P, T, then mols
        for k in range(int(self.config['ATMOSPHERE-LAYERS'])):
            # Update each layer
            key = 'ATMOSPHERE-LAYER-' + str(k+1)
            layer = self.config[key].split(',')
            layer[imol] = str(val)
            self.config[key] = ','.join(layer)

    def update_temperature(self, val):
        """
        Updates temperature profile in the configuration file

        Inputs
        ------
        val: list/array, floats.  T(p) profile parameter values.
        """
        val = [val[self.iPT[i]] for i in range(self.nPT)]
        temp = self.PTfunc(self.pressure, *val)
        # Prevent negative values by choosing the closest non-negative value, if possible
        mask = temp < 0
        if mask.sum() < temp.size:
            temp[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), temp[~mask])
        else:
            # All values are negative - set it to 1 everywhere so the result is still valid 
            # even if Tmin is not specified
            temp[:] = 1.
        if self.Tmin is not None:
            temp[temp < self.Tmin] = self.Tmin
        if self.Tmax is not None:
            temp[temp > self.Tmax] = self.Tmax
        for k in range(int(self.config['ATMOSPHERE-LAYERS'])):
            key = 'ATMOSPHERE-LAYER-' + str(k+1)
            layer = self.config[key].split(',')
            layer[1] = "{:.4e}".format(temp[k])
            self.config[key] = ','.join(layer)
        self.config['SURFACE-TEMPERATURE'] = temp[0]

    def update_kempton(self, par, val):
        """
        Updates configuration file parameter for Kempton chemistry model.

        Inputs
        ------
        pars: list/array, strings. Parameter names.
        vals: list/array, floats.  Parameter values.
        """
        # Get parameter index
        ipar = U.kempton_keys.index(par)
        # Update configuration file parameter
        pars = self.config['ATMOSPHERE-ABUN'].split(',')
        pars[ipar] = str(val)
        self.config['ATMOSPHERE-ABUN'] = ','.join(pars)
        if par == 'KEMPTON-TFILL' and pars[0] == 'isothermal':
            self.config['SURFACE-TEMPERATURE'] = val

    def update_tea(self, pars, vals):
        """
        Updates configuration file parameters for TEA chemistry model.
        
        Inputs
        ------
        pars: list/array, strings. Parameter names.
        vals: list/array, floats.  Parameter values.
        """
        # Load values
        assert len(pars) == 2, "Must supply exactly 2 parameters for TEA model."
        assert len(vals) == 2, "Must supply exactly 2 values for TEA model."
        if pars[0] == 'TEA-MH':
            mh  = vals[0]
            c2o = vals[1]
        else:
            mh  = vals[1]
            c2o = vals[0]
        # Update the atmosphere layers
        for i in range(int(self.config['ATMOSPHERE-LAYERS'])):
            key = 'ATMOSPHERE-LAYER-'+str(i+1)
            lay = self.config[key].split(',')
            abuns = self.tea((float(lay[0]), float(lay[1]), mh, c2o))
            strabuns = np.array2string(abuns, precision=3).replace('\n', '').replace('[', '').replace(']', '').split()
            self.config[key] = ','.join(lay[:2] + strabuns)

    def plot_best_fit(self):
        """
        Produces a plot of the best-fit spectrum with the data.
        """
        #bestfit = self.forward_model(self.bestp)[0]
        bestfit = self._worker(self.bestp, manual_gc=False)
        cfg_str = self.psg[0].config_dict_to_str(self.config)
        with open(os.path.join(self.outputdir, 'bestfit.cfg'), 'w') as foo:
            foo.write(cfg_str)
        plt.errorbar(self.wav, self.data, yerr=self.uncert, label='Data')
        plt.plot(self.wavelength, bestfit, label='Best fit', lw=2, zorder=100)
        plt.legend(loc='best')
        plt.xlabel(U.wave_units[self.config['GENERATOR-RANGEUNIT']])
        plt.ylabel(U.rad_units[self.config['GENERATOR-RADUNITS']])
        plt.savefig(os.path.join(self.outputdir, 'bestfit'))
        plt.close()
        return








