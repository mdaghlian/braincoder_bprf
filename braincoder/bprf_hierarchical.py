import pandas as pd
import numpy as np
from .utils import format_data, format_paradigm, get_rsq, calculate_log_prob_t, calculate_log_prob_gauss_loc0, calculate_log_prob_gauss, format_parameters
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb

from .bprf import *
from .bprf_GPs import *

import copy
class BPRF_hier(BPRF):
    ''' Wrapper to do bayesian prf fitting + hiearchical
    designed by Marcus Daghlian
    '''

    def __init__(self, model, data,  **kwargs):
        ''' __init__
        Set up and run "bayesian" pRF analysis. This class contains:
        * model + stimulus/paradigm (from .models, used to generate predictions)
        * prior definitions (per parameter)
        '''
        super().__init__(model, data, **kwargs)
        # Do we want to fit our priors?
        # -> hierarchical priors
        self.h_prior_to_apply = {} # How is the hierarchical prior applied to the parameters? i.e., take loc and scale...
        # The rest follows as similar form as the non-hiearchical priors, but with "h_" in front
        self.h_labels = {} # What are the labels for the hierarchical parameters? & their idx
        self.h_bijector = {} # What bijector to apply to the hierarchical parameters
        self.h_prior_type = {} # Meta prior
        self.h_gp_function = {} # Using gaussian process? (GPs over all n_voxels)
        self.h_gp_space = {}    # GP on the 'constrained' or 'unconstrained' parameter
        self.h_gp_active = {}   # GPs restricted to the vertices currently being fit (see _update_gps_for_idx)
        self.h_prior = {}
        self.h_mcmc_sampler = {}
        self.h_mcmc_summary = None
        self.h_mcmc_mean = None
        self.h_fixed_pars = {}
        # Tie the stationary-kernel lengthscales (gpk{xid}_l) of GPs on different parameters?
        # -> {pid}_gpk{xid}_l all point to one h-parameter, shared_gpk{xid}_l (per kernel xid)
        self.shared_lengthscale = kwargs.get('shared_lengthscale', False)
        self.h_label_alias = {}     # {pid}_{gp key} -> shared label
        self._shared_l_median = {}  # shared label -> median distance of the first GP (to check units)
        self.masked_voxels = np.array([], dtype='int64')
        # MAP
        self.h_MAP_parameters = None
        self.MAP_fit_info = None

    def _h_label(self, pid, k):
        ''' h-parameter label of GP key k on parameter pid (resolving shared lengthscales) '''
        label = f'{pid}_{k}'
        return self.h_label_alias.get(label, label)

    def h_add_param(self, pid, h_prior_to_apply, **kwargs):
        ''' Add the parameters for the hierarchical priors
        i.e., fit the prior across all vertices

        Say we look at parameter 'x'
        we could make the prior for 'x' be N(x_loc,x_scale);
        Here we need to add the extra parameters x_loc, x_scale
        -> these parameters in turn can have their own priors...

        h_prior_to_apply='gp' kwargs:
            gp          : the GP object (braincoder.bprf_GPs), built over all n_voxels
            gp_space    : 'constrained' (default; GP on the parameter value) or 'unconstrained'
                          (GP on the bijector-inverse of the parameter, e.g. log amplitude with an Exp bijector)
            hyperpriors : {gp key: tfd distribution or None} overriding gp.default_hyperpriors()
            shared_lengthscale : tie gpk{xid}_l to the other GPs with a kernel xid (default: self.shared_lengthscale).
                          The shared h-parameter is called shared_gpk{xid}_l; its bijector & prior come from
                          the first GP added with that kernel.
        '''
        # [1] make sure the 'p_prior' is none, because we are doing hierarchical priors now! (i.e., the direct prior is not used)
        self.p_prior[pid] = PriorNone()
        self.p_prior_type[pid] = 'none'

        # [2]
        self.h_prior_to_apply[pid] = h_prior_to_apply # This tells us we need to apply the learnt prior to our parameter
        # Only doing hierarchical priors for normal distributions for now...
        if h_prior_to_apply=='normal':
            # [2] add the h_priors to the model_labels
            self.h_labels[f'{pid}_loc'] = len(self.h_labels)
            self.h_labels[f'{pid}_scale'] = len(self.h_labels)
            self.h_add_bijector(pid=f'{pid}_loc', bijector_type=tfb.Identity())
            self.h_add_bijector(pid=f'{pid}_scale', bijector_type=tfb.Exp())

            # [3] add the priors for the new parameters
            # ... none for now...
            self.h_add_prior(pid=f'{pid}_loc', prior_type='none', **kwargs)
            self.h_add_prior(pid=f'{pid}_scale', prior_type='none', **kwargs)

        elif h_prior_to_apply=='gp':
            # Gaussian process
            gp = kwargs.pop('gp')
            if gp.n_vx != self.n_voxels:
                raise ValueError(f'GP has {gp.n_vx} vertices, data has {self.n_voxels}')
            self.h_gp_function[pid] = gp
            self.h_gp_space[pid] = kwargs.pop('gp_space', 'constrained')
            assert self.h_gp_space[pid] in ('constrained', 'unconstrained')
            hyperpriors = {**gp.default_hyperpriors(), **kwargs.pop('hyperpriors', {})}
            positive = gp.positive_pids()
            shared = kwargs.pop('shared_lengthscale', self.shared_lengthscale)
            shared_keys = {f'gpk{s}_l': s for s in gp.stat_kernel_list} if shared else {}
            for k in gp.pids_inv.keys():
                # Bijector & prior are decided on the GP's own key (k), never on the full label
                label = f'{pid}_{k}'
                if k in shared_keys:
                    s = shared_keys[k]
                    self.h_label_alias[label] = f'shared_{k}'
                    label = f'shared_{k}'
                    if label in self.h_labels:
                        # already added by another parameter's GP
                        other = self._shared_l_median[label]
                        if not np.isclose(other, gp.median_dX[s], rtol=1e-2):
                            print(f'Warning: {label} is shared, but the median distances of the GPs differ '
                                  f'({other:.3g} vs {gp.median_dX[s]:.3g}); are they in the same units?')
                        continue
                    self._shared_l_median[label] = gp.median_dX[s]
                self.h_labels[label] = len(self.h_labels)
                self.h_add_bijector(pid=label, bijector_type=tfb.Softplus() if k in positive else tfb.Identity())
                if hyperpriors.get(k) is None:
                    self.h_add_prior(pid=label, prior_type='none')
                else:
                    self.h_add_prior(pid=label, prior_type=type(hyperpriors[k]).__name__,
                                     distribution=hyperpriors[k])


    def h_add_bijector(self, pid, bijector_type, **kwargs):
        ''' add transformations to parameters so that they are fit smoothly

        identity        - do nothing
        softplus        - don't let anything be negative

        '''
        if bijector_type == 'identity':
            self.h_bijector[pid] = tfb.Identity()
        elif bijector_type == 'softplus':
            # Don't let anything be negative
            self.h_bijector[pid] = tfb.Softplus()
        elif bijector_type == 'sigmoid':
            self.h_bijector[pid] = tfb.Sigmoid(
                low=kwargs.get('low'), high=kwargs.get('high')
            )
        else:
            self.h_bijector[pid] = bijector_type

    def h_add_prior(self, pid, prior_type, **kwargs):
        '''
        Adds the prior to each parameter:

        Options:
            uniform:    uniform probability b/w the specified bounds (low, high). Otherwise infinite
            normal:     normal probability. (loc, scale)
            none:       The parameter can still vary, but it will not influence the outcome...

        '''
        if pid not in self.h_labels.keys(): # Is this a valid parameter to add?
            print(f'error... not {pid} not in h labels')
            return
        self.h_prior_type[pid] = prior_type
        if prior_type=='normal':
            # Get loc, and scale
            loc = kwargs.get('loc')
            scale = kwargs.get('scale')
            self.h_prior[pid]   = PriorNorm(loc, scale)
        elif prior_type=='uniform' :
            low = kwargs.get('low')
            high = kwargs.get('high')
            self.h_prior[pid] = PriorUniform(low, high)
        elif prior_type=='none':
            self.h_prior[pid] = PriorNone()
        elif prior_type=='fixed':
            fixed_val = kwargs.get('fixed_val')
            self.h_prior[pid] = PriorFixed(fixed_val)
        else:
            self.h_prior[pid] = PriorGeneral(prior_type=prior_type, distribution=kwargs.get('distribution'))


    def h_prep_for_fitting(self, **kwargs):
        ''' Get everything ready for fitting...
        h-parameters can be fixed by self.h_fixed_pars ({label: value} or a 1-row DataFrame), and are
        fixed automatically for GPs with fixed hyperparameters. Fixing only applies to this fit
        (the stored priors / bijectors are left untouched).
        '''
        # hyperparameters frozen in a GP (gp.set_log_prob_fixed) are held fixed, so they are reported correctly
        h_fixed = {}
        for pid, gp in self.h_gp_function.items():
            if not gp.is_fixed:
                continue
            for k, v in gp._fixed_kwargs.items():
                label = self._h_label(pid, k)
                if label in h_fixed and not np.isclose(h_fixed[label], float(v)):
                    raise ValueError(f'{label} is shared, but fixed to {h_fixed[label]} and {float(v)} by different GPs')
                h_fixed[label] = float(v)
        user_fixed = self.h_fixed_pars
        if isinstance(user_fixed, pd.DataFrame):
            user_fixed = user_fixed.iloc[0].to_dict()
        for k, v in user_fixed.items():
            if k in self.h_labels:
                h_fixed[k] = float(np.asarray(v).ravel()[0])
        self.h_fixed_pars = h_fixed

        if len(h_fixed) != 0:
            # 1 row - this is hierarchical!
            index = [[0, self.h_labels[k]] for k in h_fixed]
            self.h_fix_update_fn = FixUdateFn(index, list(h_fixed.values())).update_fn
        else:
            self.h_fix_update_fn = FixUdateFn().update_fn

        # Set the bijectors (to project to a useful fitting space); fixed values are used as they are
        self.h_bijector_list = [
            tfb.Identity() if p in h_fixed else self.h_bijector[p] for p in self.h_labels.keys()
        ]
        # Only loop through those parameters with a prior (and not fixed)
        self.h_priors_to_loop = [
            p for p,t in self.h_prior_type.items() if t not in ('fixed', 'none') and p not in h_fixed
        ]
        print(f'Priors to loop through: {self.h_priors_to_loop}')
        self.h_labels_inv = {v:k for k,v in self.h_labels.items()}

    # (plain methods, not tf.function: they are only called inside the per-fit tf.functions, and a
    #  tf.function method would keep the bijector list of its first trace for every later fit)
    def _h_bprf_transform_parameters_forward(self, h_parameters):
        # Loop through parameters & bijectors (forward)
        h_out = [
            self.h_bijector_list[i].forward(h_parameters[:,i]) for i in range(self.h_n_params)
            ]
        return tf.stack(h_out, axis=-1)

    def _h_bprf_transform_parameters_backward(self, h_parameters):
        # Loop through parameters & bijectors (backward)
        h_out = [
            self.h_bijector_list[i].inverse(h_parameters[:,i]) for i in range(self.h_n_params)
            ]
        return tf.stack(h_out, axis=-1)

    @staticmethod
    def _clean_idx(idx, n):
        """None / int / list -> sorted unique index array (rows of the fit are in this order)"""
        if idx is None:
            return np.arange(n)
        return np.unique(np.atleast_1d(np.asarray(idx, dtype='int64')))

    def _update_gps_for_idx(self, idx):
        ''' GPs for the vertices being fit: the GP itself if all vertices, else a new gp.subset(idx).
        (GPs are never edited in place, so earlier traces can't go stale.)
        '''
        idx = np.asarray(idx)
        full = len(idx) == self.n_voxels and np.array_equal(idx, np.arange(self.n_voxels))
        self.h_gp_active = {k: (gp if full else gp.subset(idx)) for k, gp in self.h_gp_function.items()}

    def _restore_gps(self):
        self.h_gp_active = dict(self.h_gp_function)

    def _valid_voxel_mask(self):
        ''' Voxels with finite, non-constant data [n_voxels] '''
        y = self.data.values
        finite = np.all(np.isfinite(y), axis=0)
        return finite & (np.nanstd(y, axis=0) > 0)

    def _drop_invalid_idx(self, idx, valid):
        ''' Remove vertices that are not valid from idx. They are not fit (and are left out of the GPs),
        so they keep their init values. Returns (idx, keep) with keep a bool mask over the input idx
        '''
        keep = valid[idx]
        self.masked_voxels = idx[~keep]
        if not keep.all():
            print(f'Masking {int((~keep).sum())} constant/non-finite voxel(s); they keep their init values '
                  f'(see self.masked_voxels)')
        if not keep.any():
            raise ValueError('No voxels left to fit after masking constant/non-finite voxels')
        return idx[keep], keep

    @staticmethod
    def _subset_fixed_pars(fixed_pars, keep):
        ''' Per-vertex fixed values (one per vertex in the original idx) -> the kept vertices only '''
        n = len(keep)
        if isinstance(fixed_pars, pd.DataFrame):
            return fixed_pars.iloc[keep].reset_index(drop=True) if len(fixed_pars) == n else fixed_pars
        out = {}
        for k, v in fixed_pars.items():
            a = np.asarray(v)
            out[k] = a[keep] if (a.ndim > 0 and len(a) == n and n > 1) else v
        return out

    @staticmethod
    def _pop_optimizer_kwargs(kwargs):
        ''' The _run_MAP_optimizer options, popped from kwargs '''
        names = ['clipnorm', 'patience', 'min_rel_improvement', 'restore_best', 'max_nonfinite']
        return {n: kwargs.pop(n) for n in names if n in kwargs}

    def _run_MAP_optimizer(self, neg_log_posterior_fn, opt_vars, num_steps, desc,
                           adam_kwargs=None, clipnorm=1.0, patience=None, min_rel_improvement=1e-6,
                           restore_best=True, max_nonfinite=20, step_callback=None):
        ''' Adam on neg_log_posterior_fn (no arguments) over opt_vars, with guard rails:
        * non-finite loss: no update; the variables go back to the last finite state and the
          learning rate is halved. Stops after max_nonfinite non-finite steps in a row.
        * non-finite gradient entries are zeroed (as in ParameterFitter)
        * gradients are clipped to a global norm of clipnorm (None: no clipping)
        * early stopping after patience steps without a relative improvement of min_rel_improvement
          in the best loss (patience=None: always run num_steps)
        * restore_best: finish on the lowest-loss state seen rather than the last one
        step_callback(step, loss, gradients) is called after each update.
        Returns a dict with info on the run (also stored as self.MAP_fit_info)
        '''
        optimizer = tf.optimizers.Adam(**(adam_kwargs or {}))
        snapshot = lambda: [tf.identity(v) for v in opt_vars]
        def assign(state):
            for v, x in zip(opt_vars, state):
                v.assign(x)

        best_loss, best_step = np.inf, -1
        best_state = last_good = snapshot()
        n_nonfinite, nonfinite_run, since_improved = 0, 0, 0
        warned_grad, stopped_early = False, False
        loss_history = []
        progress_bar = tqdm(range(num_steps), desc=desc)
        for step in progress_bar:
            with tf.GradientTape() as tape:
                loss = neg_log_posterior_fn()
            loss_np = float(loss.numpy())
            loss_history.append(loss_np)
            if not np.isfinite(loss_np):
                n_nonfinite += 1
                nonfinite_run += 1
                assign(last_good)
                try:
                    optimizer.learning_rate = float(optimizer.learning_rate) * 0.5
                except Exception:   # e.g. a learning rate schedule
                    pass
                if nonfinite_run >= max_nonfinite:
                    print(f'Warning: {nonfinite_run} non-finite losses in a row at step {step}; stopping')
                    break
                continue
            nonfinite_run = 0
            last_good = snapshot()

            # best state & early stopping
            significant = (not np.isfinite(best_loss)) or (
                best_loss - loss_np > min_rel_improvement * max(1.0, abs(best_loss)))
            since_improved = 0 if significant else since_improved + 1
            if loss_np < best_loss:
                best_loss, best_step, best_state = loss_np, step, last_good
            if patience is not None and since_improved >= patience:
                stopped_early = True
                break

            gradients = tape.gradient(loss, opt_vars)
            gradients = [tf.zeros_like(v) if g is None else g for g, v in zip(gradients, opt_vars)]
            n_bad = sum(int(tf.reduce_sum(tf.cast(~tf.math.is_finite(g), tf.int32))) for g in gradients)
            if n_bad:
                if not warned_grad:
                    print(f'Warning: {n_bad} non-finite gradient value(s) at step {step}; zeroed '
                          f'(further occurrences not reported)')
                    warned_grad = True
                gradients = [tf.where(tf.math.is_finite(g), g, tf.zeros_like(g)) for g in gradients]
            if clipnorm is not None:
                gradients, _ = tf.clip_by_global_norm(gradients, clipnorm)
            optimizer.apply_gradients(zip(gradients, opt_vars))
            progress_bar.set_description(f'{desc}, Loss: {loss_np:.4f} (best {best_loss:.4f})')
            if step_callback is not None:
                step_callback(step, loss_np, gradients)

        # the state after the last update has not been evaluated yet
        final_loss = float(neg_log_posterior_fn().numpy())
        if np.isfinite(final_loss) and final_loss < best_loss:
            best_loss, best_step, best_state = final_loss, len(loss_history), snapshot()
        if restore_best or not np.isfinite(final_loss):
            assign(best_state if restore_best else last_good)
        if n_nonfinite:
            print(f'Warning: {n_nonfinite} step(s) with a non-finite loss were skipped')
        self.MAP_fit_info = dict(
            n_steps=len(loss_history), best_step=best_step, best_loss=best_loss, final_loss=final_loss,
            n_nonfinite=n_nonfinite, stopped_early=stopped_early, loss=loss_history,
        )
        return self.MAP_fit_info

    def _gp_kwargs(self, pid, h_parameters):
        ''' Hyperparameters of the GP on pid, by exact label '''
        gp = self.h_gp_active[pid]
        return {k: h_parameters[:, self.h_labels[self._h_label(pid, k)]] for k in gp.pids_inv}

    def _gp_log_prior(self, pid, parameters, h_parameters):
        ''' log GP prior of parameter pid, as a density over the constrained parameter '''
        gp = self.h_gp_active[pid]
        x = parameters[:, self.model_labels[pid]]
        gpkwargs = self._gp_kwargs(pid, h_parameters)
        if self.h_gp_space[pid] == 'unconstrained':
            # prior on u = bijector^-1(x); change of variables back to x
            bij = self.p_bijector[pid]
            u = bij.inverse(x)
            return gp.return_log_prob(parameter=u, **gpkwargs) - tf.reduce_sum(
                bij.forward_log_det_jacobian(u, event_ndims=0))
        return gp.return_log_prob(parameter=x, **gpkwargs)

    def _print_gradients(self, gradients):
        print('Using tape.gradient to check gradients w/respect to each parameter')
        for i, grad in enumerate(gradients[self.n_params:]):
            print(f' Hierarchical {self.h_labels_inv[i]}: {None if grad is None else grad.numpy()}')
        for i, grad in enumerate(gradients[:self.n_params]):
            print(f' Gradient for {self.model_labels_inv[i]}: {None if grad is None else grad.numpy()}')

    def fit_mcmc_hier(self,
            init_pars=None,
            h_init_pars=None,
            num_steps=100,
            idx = None,
            fixed_pars={},
            **kwargs):
        '''
        Experimental - can we fit everything at once?
        Does that even make sense?
        '''
        verbose = kwargs.pop('verbose', False)
        idx = self._clean_idx(idx, self.n_voxels)
        self.idx_to_fit = idx
        self._update_gps_for_idx(idx)


        vx_bool = np.zeros(self.n_voxels, dtype=bool)
        vx_bool[idx] = True
        self.n_vx_to_fit = len(idx)
        self.fixed_pars = fixed_pars
        self.h_fixed_pars = kwargs.pop('h_fixed_pars', {})
        self.prep_for_fitting(**kwargs)
        self.n_params = len(self.model_labels)
        self.h_prep_for_fitting(**kwargs)
        self.h_n_params = len(self.h_labels)

        step_size = kwargs.pop('step_size', 0.0001) # rest of the kwargs go to "hmc_sample"
        self.total_n_params = self.n_params + self.h_n_params
        step_size = [tf.constant(step_size, np.float32) for _ in range(self.total_n_params)]
        paradigm = kwargs.pop('paradigm', self.paradigm)

        y = self.data.values
        init_pars = self.sort_parameters(init_pars)
        init_pars = format_parameters(init_pars)
        init_pars = init_pars.values.astype(np.float32)
        h_init_pars = self.sort_h_parameters(h_init_pars)
        h_init_pars = format_parameters(h_init_pars)
        h_init_pars = h_init_pars.values.astype(np.float32)

        # Clean the paradigm
        paradigm_ = self.model.stimulus._clean_paradigm(paradigm)

        # Define the prior in 'tf'
        log_prior_fn = self._create_log_prior_fn()

        # Calculating the likelihood
        # First lets make the function which returns the likelihood of the residuals
        residual_ln_likelihood_fn = self._create_residual_ln_likelihood_fn()

        @tf.function
        def log_posterior_fn(parameters, h_parameters):

            # [1] Mask fixed parameters
            f_parameters = self.fix_update_fn(parameters)
            f_h_parameters = self.h_fix_update_fn(h_parameters)

            # Jacobian not needed - it does this under the hood with HMC
            # For the same reason, do not need to pass parameters forward...

            par4pred = f_parameters[:,:self.n_model_params] # chop out any hyper / noise parameters
            predictions = self.model._predict(
                paradigm_[tf.newaxis, ...], par4pred[tf.newaxis, ...], None)     # Only include those parameters that are fed to the model
            residuals = y[:, vx_bool] - predictions[0]
            log_likelihood = residual_ln_likelihood_fn(f_parameters, residuals)
            log_prior = log_prior_fn(f_parameters, f_h_parameters)
            return tf.reduce_sum(log_prior + log_likelihood)

        # -> make sure we are in the correct dtype
        p_initial_state = [tf.convert_to_tensor(init_pars[vx_bool,i], dtype=tf.float32, name=n) for i,n in enumerate(self.model_labels.keys())]
        h_initial_state = [tf.convert_to_tensor(h_init_pars[:,i], dtype=tf.float32, name=n) for i,n in enumerate(self.h_labels)]

        def target_log_prob_fn(*all_parameters):
            parameters = tf.stack(all_parameters[:self.n_params], axis=-1)
            h_parameters = tf.stack(all_parameters[self.n_params:], axis=-1)
            return log_posterior_fn(parameters, h_parameters)
        all_initial_state = [*p_initial_state, *h_initial_state]

        print('Lets run some checks with everything...')
        # quick test - does it work?
        initial_ll = target_log_prob_fn(*all_initial_state)
        print(f'initial_ll={initial_ll}')
        if verbose:
            # Check the gradient with respect to each parameter
            with tf.GradientTape() as tape:
                tape.watch(all_initial_state)
                log_prob = target_log_prob_fn(*all_initial_state)
            self._print_gradients(tape.gradient(log_prob, all_initial_state))

        # CALLING GILLES' "sample_hmc" from .utils.mcmc
        print(f"Starting NUTS sampling...")
        samples, stats = bprf_sample_NUTS(
            init_state = all_initial_state,
            target_log_prob_fn=target_log_prob_fn,
            unconstraining_bijectors=[*self.p_bijector_list, *self.h_bijector_list],
            num_steps=num_steps,
            # OTHER STUFF TO OPTIMIZE
            step_size=step_size,
            **kwargs
            )
        # stuff to save...
        all_samples = tf.stack(samples[:self.n_params], axis=-1).numpy()
        # nsteps, n_voxels, n_params

        for ivx_loc,ivx_fit in enumerate(idx):
            estimated_p_dict = {}
            for i,p in enumerate(self.model_labels):
                estimated_p_dict[p] = all_samples[:,ivx_loc,i]
            for p,v in self.fixed_pars.items():
                estimated_p_dict[p] = estimated_p_dict[p]*0 + np.asarray(v)[ivx_loc]
            self.mcmc_sampler[ivx_fit] = pd.DataFrame(estimated_p_dict)

        h_samples = tf.stack(samples[self.n_params:], axis=-1).numpy().squeeze()
        self.h_mcmc_sampler = {}
        for h in self.h_labels:
            self.h_mcmc_sampler[h] = h_samples[:,self.h_labels[h]]
        self.h_mcmc_sampler = pd.DataFrame(self.h_mcmc_sampler)
        self.mcmc_stats = stats
        self._restore_gps()

    def get_mcmc_summary(self, burnin=100, pc_range=25):
        bpars = {}
        bpars_m = {}
        for p in list(self.model_labels.keys()):
            m = []
            q1 = []
            q2 = []
            uc = []
            for idx in range(self.n_voxels):
                this_p = self.mcmc_sampler[idx][p][burnin:].to_numpy()
                m.append(np.percentile(this_p,50))
                tq1 = np.percentile(this_p, pc_range)
                tq2 = np.percentile(this_p, 100-pc_range)
                tuc = tq2 - tq1

                q1.append(tq1)
                q2.append(tq2)
                uc.append(tuc)
            bpars_m[p] = np.array(m)
            bpars[f'm_{p}'] = np.array(m)
            bpars[f'q1_{p}'] = np.array(q1)
            bpars[f'q2_{p}'] = np.array(q2)
            bpars[f'uc_{p}'] = np.array(uc)

        self.mcmc_summary = pd.DataFrame(bpars)
        self.mcmc_mean = pd.DataFrame(bpars_m)

        hpars = {}
        hpars_m = {}
        for p in list(self.h_labels.keys()):

            this_p = self.h_mcmc_sampler[p][burnin:].to_numpy()
            m = np.percentile(this_p,50)
            tq1 = np.percentile(this_p, pc_range)
            tq2 = np.percentile(this_p, 100-pc_range)
            tuc = tq2 - tq1

            hpars_m[p] = np.array(m)
            hpars[f'm_{p}'] = np.array([m])
            hpars[f'q1_{p}'] = np.array([tq1])
            hpars[f'q2_{p}'] = np.array([tq2])
            hpars[f'uc_{p}'] = np.array([tuc])

        self.h_mcmc_summary = pd.DataFrame(hpars)
        self.h_mcmc_mean = pd.DataFrame.from_dict(hpars_m, orient='index').T.astype('float32')

    def fit_MAP_hier(self,
                init_pars=None,
                h_init_pars=None,
                num_steps=100,
                idx = None,
                fixed_pars={},
                **kwargs):
        '''
        Maximum a posteriori (MAP) optimization for hierarchical model.
        Finds the parameter values that maximize the posterior distribution.

        Vertices not in idx keep their init_pars values in self.MAP_parameters.
        kwargs: h_fixed_pars, adam_kwargs, verbose (print per-parameter gradients), track_params/loss/gradients
            mask_constant (True): leave out voxels with constant / non-finite data (they keep their init values)
            clipnorm (1.0), patience (None), min_rel_improvement (1e-6), restore_best (True), max_nonfinite (20):
                see _run_MAP_optimizer. Info on the run is stored in self.MAP_fit_info
        '''
        optimizer_type = kwargs.get('optimizer', 'adam' )
        track_params = kwargs.get('track_params', False)
        track_loss = kwargs.get('track_loss', False)
        track_gradients = kwargs.get('track_gradients', False)
        verbose = kwargs.pop('verbose', False)
        adam_kwargs = kwargs.pop('adam_kwargs', {})
        opt_kwargs = self._pop_optimizer_kwargs(kwargs)
        idx = self._clean_idx(idx, self.n_voxels)
        if kwargs.pop('mask_constant', True):
            idx, keep = self._drop_invalid_idx(idx, self._valid_voxel_mask())
            fixed_pars = self._subset_fixed_pars(fixed_pars, keep)
        self.idx_to_fit = idx
        self._update_gps_for_idx(idx)

        vx_bool = np.zeros(self.n_voxels, dtype=bool)
        vx_bool[idx] = True
        self.n_vx_to_fit = len(idx)
        self.fixed_pars = fixed_pars
        self.h_fixed_pars = kwargs.pop('h_fixed_pars', {})
        self.prep_for_fitting(**kwargs)
        self.n_params = len(self.model_labels)
        self.h_prep_for_fitting(**kwargs)
        self.h_n_params = len(self.h_labels)

        self.total_n_params = self.n_params + self.h_n_params
        paradigm = kwargs.pop('paradigm', self.paradigm)

        y = self.data.values
        init_pars = self.sort_parameters(init_pars)
        init_pars = format_parameters(init_pars)
        init_pars = init_pars.values.astype(np.float32)
        h_init_pars = self.sort_h_parameters(h_init_pars)
        h_init_pars = format_parameters(h_init_pars)
        h_init_pars = h_init_pars.values.astype(np.float32)

        # Clean the paradigm
        paradigm_ = self.model.stimulus._clean_paradigm(paradigm)

        # Define the prior in 'tf'
        log_prior_fn = self._create_log_prior_fn()
        log_jac_fn = self._create_log_jac_fn()

        # Calculating the likelihood
        # -> based on our 'noise_method'
        residual_ln_likelihood_fn = self._create_residual_ln_likelihood_fn()

        # Now create the log_posterior_fn
        @tf.function
        def log_posterior_fn(parameters_unc, h_parameters_unc):
            # [1] Mask fixed parameters
            f_parameters_unc = self.fix_update_fn(parameters_unc)
            f_h_parameters_unc = self.h_fix_update_fn(h_parameters_unc)

            # [3] Jacobian on unstransformed
            log_jac = log_jac_fn(f_parameters_unc, f_h_parameters_unc)
            # [4] Push through bijectors (constrain)
            f_parameters = self._bprf_transform_parameters_forward(f_parameters_unc)
            f_h_parameters = self._h_bprf_transform_parameters_forward(f_h_parameters_unc)

            # [5] Compute the priors
            log_prior = log_prior_fn(f_parameters, f_h_parameters)

            # [6] Compute the log likelihood
            par4pred = f_parameters[:,:self.n_model_params] # chop out any hyper / noise parameters
            predictions = self.model._predict(
                paradigm_[tf.newaxis, ...], par4pred[tf.newaxis, ...], None)     # Only include those parameters that are fed to the model
            residuals = y[:, vx_bool] - predictions[0]
            log_likelihood = residual_ln_likelihood_fn(f_parameters, residuals)
            return tf.reduce_sum(log_likelihood + log_prior + log_jac)

        # -> make sure we are in the correct dtype
        # Convert initial parameters to tensors
        p_state_tensors = [
            tf.convert_to_tensor(init_pars[vx_bool, i], dtype=tf.float32, name=name)
            for i, name in enumerate(self.model_labels.keys())
        ]

        h_state_tensors = [
            tf.convert_to_tensor(h_init_pars[:, i], dtype=tf.float32, name=name)
            for i, name in enumerate(self.h_labels)
        ]

        # Transform initial states using bijectors
        p_unconstrained = [bijector.inverse(state) for bijector, state in zip(self.p_bijector_list, p_state_tensors)]
        h_unconstrained = [bijector.inverse(state) for bijector, state in zip(self.h_bijector_list, h_state_tensors)]

        print("Starting MAP optimization...")

        # Initialize optimization variables
        p_opt_vars = [tf.Variable(state, name=self.model_labels_inv[i]) for i,state in enumerate(p_unconstrained)]
        h_opt_vars = [tf.Variable(state, name=self.h_labels_inv[i]) for i,state in enumerate(h_unconstrained)]

        all_opt_vars = [*p_opt_vars, *h_opt_vars]

        @tf.function
        def neg_log_posterior_fn():
            return -log_posterior_fn(
                tf.stack(all_opt_vars[:self.n_params], axis=-1),
                tf.stack(all_opt_vars[self.n_params:], axis=-1)
                )

        # **** quick test ****
        initial_ll = neg_log_posterior_fn()
        print(f'initial neg ll={initial_ll}')
        if verbose:
            # Check the gradient with respect to each parameter
            with tf.GradientTape() as tape:
                loss = neg_log_posterior_fn()
            self._print_gradients(tape.gradient(loss, all_opt_vars))
        # **** **** **** ****

        track_params_list = []
        track_h_params_list = []
        track_loss_list = []
        track_gradients_list = []
        def step_callback(step, loss, gradients):
            if track_loss:
                track_loss_list.append(loss)
            if track_gradients:
                track_gradients_list.append([grad.numpy() for grad in gradients])
            if track_params:
                current_p = [self.p_bijector_list[i](ov).numpy() for i, ov in enumerate(all_opt_vars[:self.n_params])]
                track_params_list.append(current_p)
                current_h = [self.h_bijector_list[i](ov).numpy() for i, ov in enumerate(all_opt_vars[self.n_params:])]
                track_h_params_list.append(current_h)

        self._run_MAP_optimizer(neg_log_posterior_fn, all_opt_vars, num_steps, 'MAP hier',
                                adam_kwargs=adam_kwargs, step_callback=step_callback, **opt_kwargs)

        self.MAP_track ={}
        if track_params:
            track_params_list = np.array(track_params_list)
            self.MAP_track['params'] = {}
            for k,v in self.model_labels.items():
                self.MAP_track['params'][k] = track_params_list[:,v,:]
            self.MAP_track['h_params'] = {}
            track_h_params_list = np.array(track_h_params_list).squeeze()
            for k,v in self.h_labels.items():
                self.MAP_track['h_params'][k] = track_h_params_list[:,v]
        if track_loss:
            self.MAP_track['loss'] = track_loss_list
        if track_gradients:
            self.MAP_track['gradients'] = track_gradients_list
        # Extract optimized parameters
        # -> apply the fixed values, then transform parameters forward
        p_final = self._bprf_transform_parameters_forward(
            self.fix_update_fn(tf.stack(all_opt_vars[:self.n_params], axis=-1))).numpy()
        h_final = self._h_bprf_transform_parameters_forward(
            self.h_fix_update_fn(tf.stack(all_opt_vars[self.n_params:], axis=-1))).numpy()

        # rows of the fit are the vertices in vx_bool order; others keep their initial values
        full_pars = init_pars.copy()
        full_pars[vx_bool] = p_final
        self.MAP_parameters = pd.DataFrame(full_pars, columns=list(self.model_labels))
        self.h_MAP_parameters = pd.DataFrame(h_final, columns=list(self.h_labels))
        self._restore_gps()
        print('MAP optimization finished.')

    def _create_log_prior_fn(self):
        @tf.function
        def log_prior_fn(parameters, h_parameters):
            # Log-prior function for the model
            p_out = 0.0
            for p in self.priors_to_loop:
                p_out += tf.reduce_sum(self.p_prior[p].prior(parameters[:,self.model_labels[p]]))
            # Also apply the hierarchical priors
            for h in self.h_prior_to_apply.keys():
                if self.h_prior_to_apply[h]=='normal':
                    p_for_prior = parameters[:,self.model_labels[h]]
                    loc_for_prior = h_parameters[:,self.h_labels[h+'_loc']]
                    scale_for_prior = h_parameters[:,self.h_labels[h+'_scale']]
                    p_out += tf.reduce_sum(calculate_log_prob_gauss(
                        data=p_for_prior, loc=loc_for_prior, scale=scale_for_prior
                    ))

                elif self.h_prior_to_apply[h]=='gp':
                    p_out += self._gp_log_prior(h, parameters, h_parameters)

            for h in self.h_priors_to_loop:
                p_out += tf.reduce_sum(self.h_prior[h].prior(h_parameters[:,self.h_labels[h]]))
            return p_out
        return log_prior_fn

    def _create_log_jac_fn(self):
        if not self.include_jacobian:
            return lambda params_unc, h_params_unc: 0.0

        @tf.function
        def log_jac_fn(params_unc, h_params_unc):
            log_det = 0.0

            # 1) “Other” model parameters (baseline, width, etc.)
            for name in self.priors_to_loop:
                x = params_unc[:, self.model_labels[name]]     # shape [n_vx]
                bij = self.p_bijector[name]                   # e.g. Sigmoid
                log_det += tf.reduce_sum(
                    bij.forward_log_det_jacobian(x, event_ndims=0)
                )

            # 2) Vertex-wise parameters with GP priors
            # (the GP log prior is a density over the constrained parameter, also in 'unconstrained' mode)
            for name in self.h_gp_function.keys():
                x = params_unc[:, self.model_labels[name]]     # slice from params, not h_params
                bij = self.p_bijector[name]                   # same bijector you used when transforming
                log_det += tf.reduce_sum(
                    bij.forward_log_det_jacobian(x, event_ndims=0)  # scalar per vertex
                )

            # 3) Hierarchical meta‐parameters (loc, scale, lengthscale hyperpriors, etc.)
            for name in self.h_priors_to_loop:
                x = h_params_unc[:, self.h_labels[name]]
                bij = self.h_bijector[name]
                log_det += tf.reduce_sum(
                    bij.forward_log_det_jacobian(x, event_ndims=0)
                )

            return log_det

        return log_jac_fn

    def _create_residual_ln_likelihood_fn(self):
        # Calculating the likelihood
        if self.noise_method == 'fit_tdist':
            @tf.function
            def residual_ln_likelihood_fn(parameters, residuals):
                resid_ln_likelihood = calculate_log_prob_t(
                    data=residuals,
                    scale=parameters[:,self.model_labels['noise_scale']],
                    dof=parameters[:,self.model_labels['noise_dof']],
                )
                resid_ln_likelihood = tf.reduce_sum(resid_ln_likelihood)
                return resid_ln_likelihood
        elif self.noise_method == 'fit_normal':
            # Assume residuals are normally distributed (loc=0.0)
            # Add the scale as an extra parameters to be fit
            @tf.function
            def residual_ln_likelihood_fn(parameters, residuals):
                # -> rescale based on std...
                resid_ln_likelihood = calculate_log_prob_gauss_loc0(
                    data=residuals,
                    scale=parameters[:,self.model_labels['noise_scale']],
                )
                resid_ln_likelihood = tf.reduce_sum(resid_ln_likelihood)
                return resid_ln_likelihood
        elif self.noise_method == 'fit_ar1' :
            @tf.function
            def residual_ln_likelihood_fn(parameters, residuals):
                noise_scale = parameters[:, self.model_labels['noise_scale']]
                noise_ar1 = parameters[:, self.model_labels['noise_ar1']]

                innov = residuals[1:, ] - tf.expand_dims(noise_ar1, 0) * residuals[:-1,:]
                innov_ln = calculate_log_prob_gauss_loc0(
                    data=innov,
                    scale=parameters[:, self.model_labels['noise_scale']],
                )
                innov_ln_sum = tf.reduce_sum(innov_ln, axis=0)
                denom = 1.0 - noise_ar1 ** 2
                init_scale = noise_scale / tf.sqrt(denom)   # stationary SD of the first sample
                init_ln = calculate_log_prob_gauss_loc0(
                    data=residuals[0,:],
                    scale=init_scale,
                )
                total_ln_per_row = innov_ln_sum + init_ln
                total_ln = tf.reduce_sum(total_ln_per_row)
                return total_ln
        else: # self.noise_method == 'none':
            # Do not fit the noise - assume it is normally distributed
            # -> calculate scale based on the standard deviation of the residuals
            @tf.function
            def residual_ln_likelihood_fn(parameters, residuals):
                # [1] Use N(0, std)
                residuals_std  = tf.math.reduce_std(residuals, axis=0)
                # -> rescale based on std...
                resid_ln_likelihood = calculate_log_prob_gauss_loc0(
                    data=residuals, scale=residuals_std,
                )
                resid_ln_likelihood = tf.reduce_sum(resid_ln_likelihood)
                return resid_ln_likelihood

        return residual_ln_likelihood_fn

    def sort_h_parameters(self, h_parameters):
        # Make sure pd.Dataframe parameters are in order
        h_parameters = reorder_dataframe_columns(pd_to_fix=h_parameters, dict_index=self.h_labels)
        return h_parameters



    def fit_MAP_hier_GP_only(self,
                pid,            # Which parameter is this GP associated with
                pid_pars,       # The values of these parameters (e.g., obtained with a basic grid fit)
                h_init_pars,
                num_steps=100,
                idx = None,
                **kwargs):
        '''
        Maximum a posteriori (MAP) optimization for hierarchical model. Fitting the GP only
        The idea is to:
        [1] Do some basic traditional parameter fitting on the data (e.g., pRF, nCSF)
        [2] You expect that some of these parameters vary spatially
        - so you want to apply a GP prior
        - But you don't know how to set the hyperparameters (lengthscale, variance, etc)
        - So you fit these on the output of [1]
        - There is no interaction with the data
        [3] You fit your hyperparameters **here** and then fix them for subsequent fitting
        - now using your "informed" prior
          (e.g. gp.set_log_prob_fixed(**hyper) -> BPRF_hier holds those h-parameters fixed)

        A sort of "empirical bayesian approach

        pid_pars: DataFrame over all n_voxels (rows idx are used). Vertices where a pid_pars value is
                  non-finite (in the GP's space) are always left out.
        kwargs: h_fixed_pars, adam_kwargs, verbose,
            mask_constant (True): also leave out voxels with constant / non-finite data
            clipnorm (1.0), patience (None), min_rel_improvement (1e-6), restore_best (True), max_nonfinite (20):
                see _run_MAP_optimizer
        This is an alternative to fitting everything at the same time. Which is also an option in
        '''
        optimizer_type = kwargs.get('optimizer', 'adam' )
        verbose = kwargs.pop('verbose', False)
        adam_kwargs = kwargs.pop('adam_kwargs', {})
        opt_kwargs = self._pop_optimizer_kwargs(kwargs)
        if not isinstance(pid, list):
            pid = [pid]
        # values of each parameter in the space of its GP (all vertices)
        pid_vals = {}
        for p in pid:
            if self.h_prior_to_apply[p] != 'gp':
                raise ValueError(f'{p} has no GP prior')
            vals = tf.convert_to_tensor(pid_pars[p].values.astype(np.float32), dtype=tf.float32, name=p)
            if self.h_gp_space[p] == 'unconstrained':
                vals = self.p_bijector[p].inverse(vals)
            pid_vals[p] = vals.numpy()

        valid = np.all([np.isfinite(v) for v in pid_vals.values()], axis=0)
        if kwargs.pop('mask_constant', True):
            valid &= self._valid_voxel_mask()
        idx, _ = self._drop_invalid_idx(self._clean_idx(idx, self.n_voxels), valid)
        self.idx_to_fit = idx
        self._update_gps_for_idx(idx)
        vx_bool = np.zeros(self.n_voxels, dtype=bool)
        vx_bool[idx] = True
        self.n_vx_to_fit = len(idx)
        self.h_fixed_pars = kwargs.pop('h_fixed_pars', {})
        self.n_params = len(self.model_labels)
        self.h_prep_for_fitting(**kwargs)
        self.h_n_params = len(self.h_labels)

        pid_pars_tf = {p: tf.constant(v[vx_bool], dtype=tf.float32) for p, v in pid_vals.items()}
        h_init_pars = self.sort_h_parameters(h_init_pars)
        h_init_pars = format_parameters(h_init_pars)
        h_init_pars = h_init_pars.values.astype(np.float32)

        # Define the log prior function
        @tf.function
        def log_prior_fn(h_parameters):
            p_out = 0.0
            for h in self.h_priors_to_loop:
                p_out += tf.reduce_sum(self.h_prior[h].prior(h_parameters[:,self.h_labels[h]]))
            return p_out
        if self.include_jacobian:
            @tf.function
            def log_jac_fn(h_parameters):
                p_out = 0.0
                for h in self.h_priors_to_loop:
                    p_out += tf.reduce_sum(self.h_bijector[h].forward_log_det_jacobian(
                        h_parameters[:,self.h_labels[h]],
                        event_ndims=0,
                    ))
                return p_out
        else:
            @tf.function
            def log_jac_fn(h_parameters):
                return 0.0
        # Now create the log_posterior_fn
        @tf.function
        def log_posterior_fn(h_parameters):
            # [1] Mask fixed parameters
            h_parameters = self.h_fix_update_fn(h_parameters)
            # [2] Jacobian
            log_jac = log_jac_fn(h_parameters)
            # [3] Bijector (h parameter only, other are fixed)
            h_parameters = self._h_bprf_transform_parameters_forward(h_parameters)
            # [4] Prior (h parameters )
            log_prior = log_prior_fn(h_parameters)
            # [5] Apply gp to parameters
            gp_likelihood = 0.0
            for p in pid:
                gp_likelihood += tf.reduce_sum(self.h_gp_active[p].return_log_prob(
                    parameter=pid_pars_tf[p], **self._gp_kwargs(p, h_parameters),
                ))
            return tf.reduce_sum(log_prior + gp_likelihood + log_jac)

        # -> make sure we are in the correct dtype
        h_state_tensors = [
            tf.convert_to_tensor(h_init_pars[:, i], dtype=tf.float32, name=name)
            for i, name in enumerate(self.h_labels)
        ]

        # Transform initial states using bijectors
        h_unconstrained = [bijector.inverse(state) for bijector, state in zip(self.h_bijector_list, h_state_tensors)]
        print("Starting MAP optimization...")
        # Initialize optimization variables
        h_opt_vars = [tf.Variable(state, name=self.h_labels_inv[i]) for i,state in enumerate(h_unconstrained)]

        @tf.function
        def neg_log_posterior_fn():
            return -log_posterior_fn(tf.stack(h_opt_vars, axis=-1))

        # **** quick test ****
        initial_ll = neg_log_posterior_fn()
        print(f'initial neg ll={initial_ll}')
        if verbose:
            with tf.GradientTape() as tape:
                loss = neg_log_posterior_fn()
            gradients = tape.gradient(loss, h_opt_vars)
            for i, grad in enumerate(gradients):
                print(f' Hierarchical {self.h_labels_inv[i]}: {None if grad is None else grad.numpy()}')
        # **** **** **** ****

        self._run_MAP_optimizer(neg_log_posterior_fn, h_opt_vars, num_steps, 'MAP GP only',
                                adam_kwargs=adam_kwargs, **opt_kwargs)

        # Extract optimized parameters
        # -> apply the fixed values, then transform parameters forward
        h_final = self._h_bprf_transform_parameters_forward(
            self.h_fix_update_fn(tf.stack(h_opt_vars, axis=-1))).numpy()
        self.h_MAP_parameters = pd.DataFrame(h_final, columns=list(self.h_labels))
        print('MAP optimization finished.')
        self._restore_gps()

    def h_default_init_pars(self, pars):
        ''' Data-driven starting values for every h-parameter, from per-vertex estimates pars
        (DataFrame over all n_voxels, e.g. a classical fit; constant / non-finite voxels are ignored).
        normal: loc, scale = mean, std. GP (in its gp_space): mfunc_mean = mean, slopes 0, half the variance
        in the nugget and half spread over the kernels, lengthscales = half the median distance,
        warp weights = 1/std(Xs), spectral parameters 1.
        Returns a 1-row DataFrame with columns h_labels
        '''
        valid = self._valid_voxel_mask()
        h = {}
        for pid, how in self.h_prior_to_apply.items():
            x = pars[pid].values.astype(np.float32)
            if how == 'gp' and self.h_gp_space[pid] == 'unconstrained':
                x = self.p_bijector[pid].inverse(x).numpy()
            x = x[valid & np.isfinite(x)].astype('float64')
            mu = float(np.mean(x)) if len(x) else 0.0
            var = max(float(np.var(x)) if len(x) > 1 else 1.0, 1e-6)
            if how == 'normal':
                h[f'{pid}_loc'], h[f'{pid}_scale'] = mu, np.sqrt(var)
                continue
            gp = self.h_gp_function[pid]
            kernels = gp.stat_kernel_list + gp.warp_kernel_list
            init = {'gpk_nugget': 0.5 * var, 'mfunc_mean': mu}
            init.update({f'gpk{s}_v': np.sqrt(0.5 * var / len(kernels)) for s in kernels})
            init.update({f'gpk{s}_l': gp.median_dX[s] / 2 for s in gp.stat_kernel_list})
            for s in gp.warp_kernel_list:
                sd = np.std(gp.Xs[s].numpy(), axis=0)
                init.update({f'gpk{s}_w{i}': 1.0 / sd[i] if sd[i] > 0 else 1.0 for i in range(len(sd))})
            if isinstance(gp, GPSpec):
                # sum(kappa) / n_vx ~ prior variance per vertex, with an orthonormal basis
                init['gpk_var'] = 0.5 * var * gp.n_vx / gp.n_lbo
            for k in gp.pids_inv:
                # remaining keys: mean-function slopes (0) and spectral kernel parameters (1)
                default = 0.0 if k.startswith('mfunc') else 1.0
                h.setdefault(self._h_label(pid, k), float(init.get(k, default)))
        return pd.DataFrame([h])[list(self.h_labels)].astype('float32')

    def fit_MAP_empirical_bayes(self,
                init_pars,
                gp_pids=None,
                h_init_pars=None,
                idx=None,
                fixed_pars={},
                classical_fit=True,
                fix_hyperparameters=True,
                classical_kwargs=None,
                gp_only_kwargs=None,
                map_kwargs=None):
        '''
        The 3-stage empirical-Bayes recipe in one call:
        [1] classical fit: ParameterFitter from init_pars (e.g. a grid fit). Skip with classical_fit=False.
            Noise parameters are set from the stage-1 residuals (unless given in init_pars and [1] is skipped)
        [2] GP hyperparameters fitted to the stage-1 estimates of gp_pids (fit_MAP_hier_GP_only)
        [3] MAP refit of all parameters under the GP priors (fit_MAP_hier), with the hyperparameters of
            gp_pids held at their stage-2 values (fix_hyperparameters=True; exact GPs are then factorised
            once, not every step) or refined jointly (False)

        init_pars       : DataFrame over all n_voxels; model parameters (+ optionally noise parameters)
        gp_pids         : parameters whose GP hyperparameters are fit in [2] (default: all with a GP prior)
        h_init_pars     : starting h-parameters; missing ones come from h_default_init_pars
        idx             : vertices to fit (others keep init_pars)
        fixed_pars      : {label: value(s)} held fixed in [1] and [3] (per-vertex values follow idx)
        classical_kwargs: to ParameterFitter.fit (e.g. max_n_iterations, learning_rate)
        gp_only_kwargs, map_kwargs: to fit_MAP_hier_GP_only / fit_MAP_hier (defaults: num_steps=1000, patience=100)
        Returns self.MAP_parameters; the output of each stage is kept in self.EB_stages
        '''
        from .optimize import ParameterFitter
        model_cols = list(self.model.parameter_labels)
        idx_ = self._clean_idx(idx, self.n_voxels)
        pars = init_pars.reset_index(drop=True).astype('float32')
        for k, v in fixed_pars.items():
            if k not in pars:
                pars[k] = np.float32(np.nan)
            pars.iloc[idx_, pars.columns.get_loc(k)] = np.asarray(v, dtype='float32')

        # [1] classical fit
        if classical_fit:
            print('*** Stage 1: classical fit ***')
            pf = ParameterFitter(self.model, self.data.iloc[:, idx_], self.paradigm)
            fit_fixed = [k for k in fixed_pars if k in model_cols]
            est = pf.fit(init_pars=pars.iloc[idx_][model_cols].set_axis(pf.data.columns),
                         fixed_pars=fit_fixed or None, **(classical_kwargs or {}))
            pars.iloc[idx_, [pars.columns.get_loc(k) for k in model_cols]] = est[model_cols].values

        # noise parameters from the residuals
        resid = self.data.values - self.model.predict(self.paradigm, pars[model_cols], None).values
        sd = np.nanstd(resid, axis=0)
        ok = np.isfinite(sd) & (sd > 0)
        sd = np.where(ok, sd, np.median(sd[ok]) if ok.any() else 1.0)
        noise_init = {'noise_scale': sd}
        if self.noise_method == 'fit_tdist':
            noise_init['noise_dof'] = np.full(self.n_voxels, 5.0)
        elif self.noise_method == 'fit_ar1':
            r = resid - np.nanmean(resid, axis=0)
            ar1 = np.nansum(r[1:] * r[:-1], axis=0) / np.maximum(np.nansum(r * r, axis=0), 1e-12)
            ar1 = np.clip(np.nan_to_num(ar1), -0.9, 0.9)
            noise_init.update(noise_ar1=ar1, noise_scale=sd * np.sqrt(1 - ar1 ** 2))   # innovation sd
        for k, v in noise_init.items():
            if k in self.model_labels and (classical_fit or k not in pars) and k not in fixed_pars:
                pars[k] = v.astype('float32')
        pars = pars[list(self.model_labels)]

        # [2] GP hyperparameters on the stage-1 estimates
        if gp_pids is None:
            gp_pids = [p for p, how in self.h_prior_to_apply.items() if how == 'gp']
        gp_pids = [gp_pids] if isinstance(gp_pids, str) else list(gp_pids)
        h0 = self.h_default_init_pars(pars)
        if h_init_pars is not None:
            for k in h_init_pars.columns:
                if k in h0:
                    h0[k] = np.float32(h_init_pars[k].values[0])
        print('*** Stage 2: GP hyperparameters ***')
        self.fit_MAP_hier_GP_only(pid=gp_pids, pid_pars=pars, h_init_pars=h0, idx=idx,
                                  **{'num_steps': 1000, 'patience': 100, **(gp_only_kwargs or {})})
        h_stage2 = self.h_MAP_parameters.copy()

        # [3] MAP refit with the GP priors
        print('*** Stage 3: MAP with GP priors ***')
        map_kwargs = {'num_steps': 1000, 'patience': 100, **(map_kwargs or {})}
        h_fixed = dict(map_kwargs.pop('h_fixed_pars', {}))
        originals = dict(self.h_gp_function)
        try:
            if fix_hyperparameters:
                for p in gp_pids:
                    gp = originals[p]
                    hyp = {k: float(h_stage2[self._h_label(p, k)].values[0]) for k in gp.pids_inv}
                    if gp.is_fixed:
                        continue
                    if gp.nystrom or gp._log_prob_mode == 'univariate':
                        # (a fixed dense covariance would change the approximation)
                        h_fixed.update({self._h_label(p, k): v for k, v in hyp.items()})
                    else:
                        # a frozen copy: the user's GP objects are left as they are
                        gp_fixed = copy.copy(gp)
                        gp_fixed.set_log_prob_fixed(**hyp)
                        self.h_gp_function[p] = gp_fixed
            self.fit_MAP_hier(init_pars=pars, h_init_pars=h_stage2, idx=idx, fixed_pars=fixed_pars,
                              h_fixed_pars=h_fixed, **map_kwargs)
        finally:
            self.h_gp_function = originals
            self._restore_gps()

        self.EB_stages = {
            'classical': pars,
            'hyperparameters': h_stage2,
            'MAP': self.MAP_parameters,
            'h_MAP': self.h_MAP_parameters,
        }
        return self.MAP_parameters
