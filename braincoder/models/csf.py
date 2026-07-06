import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import bijectors as tfb
import logging
import pandas as pd
import numpy as np
from ..utils import format_parameters, log10
from ..utils.math import aggressive_softplus, aggressive_softplus_inverse
from ..stimuli import ContrastSensitivityStimulus
from .base import EncodingModel, HRFEncodingModel

Chung_Legge_default = pd.DataFrame({
    'width_r': [1.28,],
    'SFp': [2.5,],  # c/deg
    'CSp': [166,],
    'width_l': [0.68,],
    # MADE UP - > just for model consistency
    'amplitude': [1.0,],
    'baseline': [0,], 
    'crf_exp' : [0,], 
}).astype('float32')

class ContrastSensitivity(EncodingModel):
    
    parameter_labels = [
        'width_r',      # 0
        'SFp',          # 1
        'CSp',          # 2
        'width_l',      # 3
        'crf_exp',      # 4
        'amplitude',    # 5
        'baseline',     # 6
        ]
    stimulus_type = ContrastSensitivityStimulus

    def __init__(self, data=None, parameters=None,  
                    SF_seq=None, CON_seq=None,               
                    weights=None, omega=None, allow_neg_amplitudes=False, bounds=None, 
                    verbosity=logging.INFO, 
                    **kwargs):
        self.data = data
        self.parameters = format_parameters(parameters)
        self.weights = weights
        self.omega = omega
        grid = np.vstack([SF_seq, CON_seq]).T
        self.stim_sequence = pd.DataFrame(
            grid, columns=['SF', 'CON']).astype('float32')
        self._stim_sequence = self.stim_sequence.values
        self.n_SF = len(self.stim_sequence['SF'].unique())
        self.n_CON = len(self.stim_sequence['CON'].unique())

        self.stimulus = self.stimulus_type()
        self.paradigm = self.stimulus.clean_paradigm(self.stim_sequence)

        if omega is not None:
            self.omega_chol = np.linalg.cholesky(omega)
        self.n_params = len(self.parameter_labels)
        self.p_bijector = {
            'width_r' : tfb.Softplus(),
            'SFp'     : tfb.Softplus(),
            'CSp'     : tfb.Softplus(),
            'width_l' : tfb.Softplus(),
            'crf_exp'     : tfb.Softplus(),
            'amplitude'     : tfb.Softplus(),
            'baseline'     : tfb.Identity(),
        }
        self.transformations = [(i.forward,i.inverse) for _,i in self.p_bijector.items()]        
    def update_transformations(self):
        self.transformations = [(i.forward,i.inverse) for _,i in self.p_bijector.items()]

    def get_csf_for_plot(self, SF_grid, CON_grid=np.array([1,1]), parameters=None):
        ''' Get the csf for a grid of SF and CON values '''
        if len(SF_grid.shape) == 1:
            SF_grid, CON_grid = np.meshgrid(SF_grid, CON_grid)
        SF_grid, CON_grid = SF_grid.flatten(), CON_grid.flatten()
        grid = np.vstack([SF_grid, CON_grid]).T
        stim_sequence = pd.DataFrame(
            grid, columns=['SF', 'CON']).astype('float32')
        stim_sequence = stim_sequence.values

        parameters = self._get_parameters(parameters)
        parameters = parameters.values[np.newaxis, ...]

        csf = self._get_csf(stim_sequence, parameters).numpy()[0]
        
        return csf, SF_grid, CON_grid
    
    def get_csf_curve(self, SF_values, parameters=None):
        ''' Get the csf for a grid of SF and CON values '''
        CON_values = np.ones_like(SF_values)
        stim_sequence = pd.DataFrame({
            'SF' : SF_values,
            'CON' : CON_values,
            },
        ).astype('float32')
        stim_sequence = stim_sequence.values
        parameters = self._get_parameters(parameters)
        parameters = parameters.values[np.newaxis, ...]

        csf = self._asymetric_parabola(stim_sequence, parameters).numpy()[0]

        
        return csf


    @tf.function
    def _basis_predictions(self, paradigm, parameters):
        # paradigm: n_batches x n_timepoints x n_stimulus_features
        # parameters:: n_batches x n_voxels x n_parameters

        # norm: n_batches x n_timepoints x n_voxels

        # output: n_batches x n_timepoints x n_voxels
        csf = self._get_csf(paradigm[0], parameters)

        baseline = parameters[:, :, 6, tf.newaxis]
        result = csf + baseline
        # Want the output to be n_batches x n_timepoints x n_voxels
        result = tf.transpose(result, [0, 2, 1])
        return result
    def quick_sfmax(self, parameters, max_sfmax=5000, **kwargs):
        """calculate_sfmax    
        aka high frequency cutoff. Useful summary statistic of whole CSF curve
        set the sensitivity = 1, then solve for the corresponding SF. 
        i.e., what is the highest possible SF we can detect    
        Can be infinte (with low width_r), so we set a max value 
        """
        log10_CSp = np.log10(parameters['CSp'])
        log10_SFp = np.log10(parameters['SFp'])
        sfmax = 10**((np.sqrt(log10_CSp/(parameters['width_r']**2)) + log10_SFp))
        if len(sfmax.shape)>=1:
            sfmax[sfmax>max_sfmax] = max_sfmax
        elif sfmax>max_sfmax:
            sfmax = max_sfmax        
        return sfmax
    
    def quick_aulcsf(self, parameters, **kwargs):
        # parameters = self._get_parameters(parameters=parameters)
        parameters = parameters.values[np.newaxis, ...]  
        SF_levels = kwargs.get('SF_levels', np.array([ 0.5,  1.,   3.,   6.,  12.,  18. ]))
        SF_levels = np.logspace(
                np.log10(1.5), np.log10(18), 50
            )
        # kwargs.get('SF_levels', np.array([ 0.5,  1.,   3.,   6.,  12.,  18. ]))
        normalize_AUC = kwargs.get('normalize_AUC', True)
        stim_sequence = pd.DataFrame({'SF':SF_levels, 'CON':SF_levels*0}).astype('float32')
        log_SF_levels = np.log10(SF_levels)#.reshape(1,-1)
        # Generate grid to make the CSF     
        csf_curve = self._asymetric_parabola(
            stim_sequence = stim_sequence,
            parameters = parameters
            ).numpy()
        logcsf_curve = np.log10(csf_curve)    
        logcsf_curve[logcsf_curve<0] = 0 # Cannot have negative logCSF
        aulcsf = np.trapz(logcsf_curve.T, x=log_SF_levels, axis=0) 
        if not normalize_AUC:
            return aulcsf
        # Chung & Legge, normalized 
        parameters = Chung_Legge_default #self._get_parameters(Chung_Legge_default)
        parameters = parameters.values[np.newaxis, ...]    
        csf_curve = self._asymetric_parabola(
            stim_sequence = stim_sequence,
            parameters = parameters
            ).numpy()            
        logcsf_curve = np.log10(csf_curve)    
        logcsf_curve[logcsf_curve<0] = 0 # Cannot have negative logCSF    
        CL_aulcsf = np.trapz(logcsf_curve.T, x=log_SF_levels, axis=0) 
        norm_aulcsf = 100 * aulcsf / CL_aulcsf
        return norm_aulcsf

    @tf.function
    def _asymetric_parabola(self, stim_sequence, parameters):
        # n_batches x n_populations x  n_grid_spaces
        SF_seq = stim_sequence[:, 0][tf.newaxis, tf.newaxis, :]

        # Unpack parameters with broadcasting                
        # n_batches x n_populations x n_grid_spaces (broadcast)
        width_r = parameters[:, :, 0, tf.newaxis]
        SFp = parameters[:, :, 1, tf.newaxis]
        CSp = parameters[:, :, 2, tf.newaxis]
        width_l = parameters[:, :, 3, tf.newaxis]        
        
        # Safeguard against log of non-positive values
        SF_seq_safe = tf.maximum(SF_seq, 1e-8)
        SFp_safe = tf.maximum(SFp, 1e-8)
        CSp_safe = tf.maximum(CSp, 1e-8)

        # Logarithmic transformations
        log_SF_seq  = log10(SF_seq_safe)
        log_SFp     = log10(SFp_safe)
        log_CSp     = log10(CSp_safe)

        # Create the curves
        log_sf_diff = (log_SF_seq - log_SFp) ** 2
        L_curve = tf.math.pow(10.0, log_CSp - (log_sf_diff) * (width_l ** 2))
        R_curve = tf.math.pow(10.0, log_CSp - (log_sf_diff) * (width_r ** 2))

        # Combine the curves using boolean masks
        
        # # Split stimulus space into left and right
        # is_left = log_SF_seq < log_SFp
        # csf = tf.where(is_left, L_curve, R_curve) # REMOVE THE "WHERE" - makes unstable gradients
        # Smooth transition instead of hard `tf.where`
        alpha = 500.0  # Adjust to control smoothness
        blend_factor = tf.math.sigmoid(alpha * (log_SF_seq - log_SFp))
        csf = (1 - blend_factor) * L_curve + blend_factor * R_curve 
              
        return csf 

    @tf.function
    def _apply_crf(self, stim_sequence, parameters,csf):
        # n_batches x n_populations x  n_grid_spaces
        CON_seq = stim_sequence[:, 1][tf.newaxis, tf.newaxis, :]    
        # n_batches x n_populations x n_grid_spaces (broadcast)        
        crf_exp = parameters[:, :, 4, tf.newaxis]
        amplitude = parameters[:, :, 5, tf.newaxis]        
        # Contrast sensitivity
        cthresh = 100 / tf.clip_by_value(csf, 1e-1, 1e6) # Prevent extremes prev 1e-6...
        ncsf_resp = ((CON_seq ** crf_exp) / (CON_seq ** crf_exp + cthresh ** crf_exp)) * amplitude
        return ncsf_resp    
    
    @tf.function
    def _get_csf(self, stim_sequence, parameters):
        csf = self._asymetric_parabola(stim_sequence, parameters)
        ncsf_resp = self._apply_crf(stim_sequence, parameters, csf)

        return ncsf_resp  
   
    

class ContrastSensitivityWithHRF(HRFEncodingModel, ContrastSensitivity):
    def __init__(
        self, data=None, parameters=None,  
        SF_seq=None, CON_seq=None,               
        weights=None, omega=None, allow_neg_amplitudes=False, bounds=None,
        hrf_model=None, flexible_hrf_parameters=False, verbosity=logging.INFO, **kwargs
        ):

        ContrastSensitivity.__init__(
            self, data=data, parameters=parameters,
            SF_seq=SF_seq, CON_seq=CON_seq,
            weights=weights, omega=omega, allow_neg_amplitudes=allow_neg_amplitudes,
            bounds=bounds, verbosity=verbosity, **kwargs
        )
        HRFEncodingModel.__init__(
            self, hrf_model=hrf_model, flexible_hrf_parameters=flexible_hrf_parameters,             
            verbosity=logging.INFO, **kwargs
        )
        self.flexible_hrf_parameters=flexible_hrf_parameters
        if flexible_hrf_parameters:
            self.transformations = self.transformations + self.hrf_model.transformations
    def update_transformations(self):
        self.transformations = [(i.forward,i.inverse) for _,i in self.p_bijector.items()]
        if self.flexible_hrf_parameters:
            self.transformations = self.transformations + self.hrf_model.transformations



class ContrastSensitivityExp(ContrastSensitivityWithHRF):
    """
    Inherits from ContrastSensitivity but experimental
    """
    def _apply_crf(self, stim_sequence, parameters, csf):
        # stim_sequence: [n_samples, 2] with columns [SF, CON]
        # parameters: [n_batches, n_populations, n_params]
        # csf: contrast sensitivity filter
        # Extract contrast (second column) and reshape for broadcasting
        CON_seq = stim_sequence[:, 1][tf.newaxis, tf.newaxis, :]

        # Sigmoid slope (crf_exp) and scale (amplitude) from parameters
        crf_exp = parameters[:, :, 4, tf.newaxis]
        amplitude = parameters[:, :, 5, tf.newaxis]
        baseline = parameters[:, :, 6, tf.newaxis]

        # Contrast threshold (midpoint of sigmoid)
        cthresh = 100 / tf.clip_by_value(csf, 1e-1, 1e6)

        cs = tf.math.sqrt(CON_seq)
        # cs = tf.clip_by_value(cs, 0, 99999999)
        ncsf_resp = amplitude * (csf * cs) + baseline
        return ncsf_resp

















































class CRF(EncodingModel):
    """N independent Contrast Response Functions for N distinct spatial frequencies.

    Uses ContrastSensitivityStimulus (SF, CON columns).  One CRF is fit per
    unique SF level found in SF_seq; the curves are completely independent.

    Parameterisation
    ----------------
    SF0 (the lowest SF) is the reference:
        amplitude  – SF0 absolute amplitude (Softplus > 0)
        baseline   – SF0 absolute baseline  (unconstrained)
        <shape>_sf0 – SF0 shape parameters (c50, exponent, etc.)

    All other SFs are expressed relative to SF0:
        amplitude_sf{i}  – ratio of SF_i amplitude to SF0 amplitude (Softplus > 0)
        baseline_sf{i}   – additive delta from SF0 baseline (unconstrained)
        <shape>_sf{i}    – independent shape parameters per SF

    The prediction for a timepoint at SF_i is:
        i == 0:  amplitude * f(CON, shape_sf0) + baseline
        i  > 0:  amplitude * amplitude_sf{i} * f(CON, shape_sf{i})
                   + baseline + baseline_sf{i}

    `amplitude` and `baseline` are directly compatible with
    refine_baseline_and_amplitude (the relative baseline deltas introduce a
    small approximation there, which is acceptable for a refinement step).

    Parameter labels: amplitude, <shape>_sf0, baseline, amplitude_sf1,
                      <shape>_sf1, baseline_sf1, …
    """

    CRF_PARAM_NAMES = {
        'naka_rushton': ['amplitude', 'c50', 'exponent', 'baseline'],
        'sqrt':         ['amplitude', 'baseline'],
        'power':        ['amplitude', 'exponent', 'baseline'],
        'linear':       ['amplitude', 'baseline'],
    }

    CRF_BIJECTORS = {
        'naka_rushton': [tfb.Softplus(), tfb.Softplus(), tfb.Softplus(), tfb.Identity()],
        'sqrt':         [tfb.Softplus(), tfb.Identity()],
        'power':        [tfb.Softplus(), tfb.Softplus(), tfb.Identity()],
        'linear':       [tfb.Softplus(), tfb.Identity()],
    }

    stimulus_type = ContrastSensitivityStimulus

    def __init__(self, data=None, parameters=None,
                 SF_seq=None, CON_seq=None,
                 weights=None, omega=None,
                 crf_type='naka_rushton',
                 verbosity=logging.INFO, **kwargs):

        if crf_type not in self.CRF_PARAM_NAMES:
            raise ValueError(f"crf_type must be one of {list(self.CRF_PARAM_NAMES.keys())}")

        self.crf_type = crf_type
        self._crf_param_names = self.CRF_PARAM_NAMES[crf_type]
        self.n_crf_params = len(self._crf_param_names)

        grid = np.vstack([SF_seq, CON_seq]).T
        self.stim_sequence = pd.DataFrame(grid, columns=['SF', 'CON']).astype('float32')
        self._stim_sequence = self.stim_sequence.values

        sf_unique = self.stim_sequence['SF'].dropna().unique()
        self.sf_levels = np.sort(sf_unique).astype('float32')
        self.n_sf = len(self.sf_levels)

        # SF0 uses bare 'amplitude'/'baseline'; later SFs use relative names.
        labels = []
        for i in range(self.n_sf):
            for pname in self._crf_param_names:
                if i == 0 and pname in ('amplitude', 'baseline'):
                    labels.append(pname)
                else:
                    labels.append(f'{pname}_sf{i}')
        self.parameter_labels = labels

        self.data = data
        self.parameters = format_parameters(parameters)
        self.weights = weights
        self.omega = omega

        if omega is not None:
            self.omega_chol = np.linalg.cholesky(omega)

        self.stimulus = self.stimulus_type()
        self.paradigm = self.stimulus.clean_paradigm(self.stim_sequence)
        self.n_params = len(self.parameter_labels)

        crf_bijectors = self.CRF_BIJECTORS[crf_type]
        self.p_bijector = {
            label: crf_bijectors[j % self.n_crf_params]
            for j, label in enumerate(self.parameter_labels)
        }
        self.transformations = [(b.forward, b.inverse) for b in self.p_bijector.values()]

    def update_transformations(self):
        self.transformations = [(b.forward, b.inverse) for b in self.p_bijector.values()]

    def get_crf_curve(self, CON_values, sf_index=0, parameters=None):
        """Predicted responses across CON_values for the sf_index-th SF level.

        Returns an array of shape [n_CON, n_voxels].
        """
        parameters = self._get_parameters(parameters)
        params = parameters.values[np.newaxis, ...]
        stim = np.stack([
            np.full_like(CON_values, self.sf_levels[sf_index], dtype='float32'),
            CON_values.astype('float32'),
        ], axis=1)
        return self._basis_predictions(stim[np.newaxis, ...], params).numpy()[0]

    @tf.function
    def _basis_predictions(self, paradigm, parameters):
        # paradigm   : [1, n_timepoints, 2]   (SF, CON)
        # parameters : [n_batches, n_voxels, n_params]
        # output     : [n_batches, n_timepoints, n_voxels]

        stim    = paradigm[0]                                    # [n_timepoints, 2]
        SF_seq  = stim[:, 0][tf.newaxis, tf.newaxis, :]          # [1, 1, n_timepoints]
        CON_seq = stim[:, 1][tf.newaxis, tf.newaxis, :]          # [1, 1, n_timepoints]

        n = self.n_crf_params
        sf_levels = tf.constant(self.sf_levels, dtype=tf.float32)

        # SF0 reference amplitude and baseline (absolute)
        ref_amplitude = parameters[:, :, 0,     tf.newaxis]  # 'amplitude'
        ref_baseline  = parameters[:, :, n - 1, tf.newaxis]  # 'baseline'

        crf_sum = 0.0
        for i in range(self.n_sf):
            sf_mask = tf.cast(tf.abs(SF_seq - sf_levels[i]) < 1e-3, tf.float32)
            base = i * n

            if i == 0:
                amplitude_i = ref_amplitude
                baseline_i  = ref_baseline
            else:
                # amplitude_sf{i} is a ratio; baseline_sf{i} is an additive delta
                amplitude_i = ref_amplitude * parameters[:, :, base,         tf.newaxis]
                baseline_i  = ref_baseline  + parameters[:, :, base + n - 1, tf.newaxis]

            if self.crf_type == 'naka_rushton':
                c50      = parameters[:, :, base + 1, tf.newaxis]
                exponent = parameters[:, :, base + 2, tf.newaxis]
                con_safe = tf.maximum(CON_seq, 1e-8)
                Cn   = tf.math.pow(con_safe, exponent)
                c50n = tf.math.pow(tf.maximum(c50, 1e-8), exponent)
                resp = amplitude_i * Cn / (Cn + c50n) + baseline_i
            elif self.crf_type == 'sqrt':
                resp = amplitude_i * tf.math.sqrt(tf.maximum(CON_seq, 0.0)) + baseline_i
            elif self.crf_type == 'power':
                exponent = parameters[:, :, base + 1, tf.newaxis]
                con_safe = tf.maximum(CON_seq, 1e-8)
                resp = amplitude_i * tf.math.pow(con_safe, exponent) + baseline_i
            else:  # linear
                resp = amplitude_i * CON_seq + baseline_i

            crf_sum = crf_sum + resp * sf_mask

        return tf.transpose(crf_sum, [0, 2, 1])


class CRFWithHRF(HRFEncodingModel, CRF):
    def __init__(
        self, data=None, parameters=None,
        SF_seq=None, CON_seq=None,
        weights=None, omega=None,
        crf_type='naka_rushton',
        hrf_model=None, flexible_hrf_parameters=False,
        verbosity=logging.INFO, **kwargs
    ):
        CRF.__init__(
            self, data=data, parameters=parameters,
            SF_seq=SF_seq, CON_seq=CON_seq,
            weights=weights, omega=omega,
            crf_type=crf_type,
            verbosity=verbosity, **kwargs
        )
        HRFEncodingModel.__init__(
            self, hrf_model=hrf_model, flexible_hrf_parameters=flexible_hrf_parameters,
            verbosity=verbosity, **kwargs
        )
        self.flexible_hrf_parameters = flexible_hrf_parameters
        if flexible_hrf_parameters:
            self.transformations = self.transformations + self.hrf_model.transformations

    def update_transformations(self):
        self.transformations = [(b.forward, b.inverse) for b in self.p_bijector.values()]
        if self.flexible_hrf_parameters:
            self.transformations = self.transformations + self.hrf_model.transformations


# class ContrastSensitivityTruncatedLogWithHrf(ContrastSensitivityWithHRF):    
    # parameter_labels = [
    #     'width_r',      # 0
    #     'SFp',          # 1
    #     'CSp',          # 2
    #     'lowSFtrunk',    # 3
    #     'crf_exp',      # 4
    #     'amplitude',    # 5
    #     'baseline',     # 6
    #     ]
    # stimulus_type = ContrastSensitivityStimulus

    # def __init__(self, data=None, parameters=None,  
    #                 SF_seq=None, CON_seq=None,               
    #                 weights=None, omega=None, allow_neg_amplitudes=False, bounds=None, 
    #                 verbosity=logging.INFO, 
    #                 **kwargs):
    #     self.data = data
    #     self.parameters = format_parameters(parameters)
    #     self.weights = weights
    #     self.omega = omega
    #     grid = np.vstack([SF_seq, CON_seq]).T
    #     self.stim_sequence = pd.DataFrame(
    #         grid, columns=['SF', 'CON']).astype('float32')
    #     self._stim_sequence = self.stim_sequence.values
    #     self.n_SF = len(self.stim_sequence['SF'].unique())
    #     self.n_CON = len(self.stim_sequence['CON'].unique())

    #     self.stimulus = self.stimulus_type()
    #     self.paradigm = self.stimulus.clean_paradigm(self.stim_sequence)

    #     if omega is not None:
    #         self.omega_chol = np.linalg.cholesky(omega)
    #     self.n_params = len(self.parameter_labels)
    #     self.p_bijector = {
    #         'width_r' : tfb.Softplus(),
    #         'SFp'     : tfb.Softplus(),
    #         'CSp'     : tfb.Softplus(),
    #         'lowSFtrunc' : tfb.Softplus(),
    #         'crf_exp'     : tfb.Softplus(),
    #         'amplitude'     : tfb.Softplus(),
    #         'baseline'     : tfb.Identity(),
    #     }
    #     self.transformations = [(i.forward,i.inverse) for _,i in self.p_bijector.items()]        
    # def update_transformations(self):
    #     self.transformations = [(i.forward,i.inverse) for _,i in self.p_bijector.items()]

    # def get_csf_for_plot(self, SF_grid, CON_grid=np.array([1,1]), parameters=None):
    #     ''' Get the csf for a grid of SF and CON values '''
    #     if len(SF_grid.shape) == 1:
    #         SF_grid, CON_grid = np.meshgrid(SF_grid, CON_grid)
    #     SF_grid, CON_grid = SF_grid.flatten(), CON_grid.flatten()
    #     grid = np.vstack([SF_grid, CON_grid]).T
    #     stim_sequence = pd.DataFrame(
    #         grid, columns=['SF', 'CON']).astype('float32')
    #     stim_sequence = stim_sequence.values

    #     parameters = self._get_parameters(parameters)
    #     parameters = parameters.values[np.newaxis, ...]

    #     csf = self._get_csf(stim_sequence, parameters).numpy()[0]
        
    #     return csf, SF_grid, CON_grid
    
    # def get_csf_curve(self, SF_values, parameters=None):
    #     ''' Get the csf for a grid of SF and CON values '''
    #     CON_values = np.ones_like(SF_values)
    #     stim_sequence = pd.DataFrame({
    #         'SF' : SF_values,
    #         'CON' : CON_values,
    #         },
    #     ).astype('float32')
    #     stim_sequence = stim_sequence.values
    #     parameters = self._get_parameters(parameters)
    #     parameters = parameters.values[np.newaxis, ...]

    #     csf = self._truncparabola(stim_sequence, parameters).numpy()[0]

        
    #     return csf


    # @tf.function
    # def _basis_predictions(self, paradigm, parameters):
    #     # paradigm: n_batches x n_timepoints x n_stimulus_features
    #     # parameters:: n_batches x n_voxels x n_parameters

    #     # norm: n_batches x n_timepoints x n_voxels

    #     # output: n_batches x n_timepoints x n_voxels
    #     csf = self._get_csf(paradigm[0], parameters)

    #     baseline = parameters[:, :, 6, tf.newaxis]
    #     result = csf + baseline
    #     # Want the output to be n_batches x n_timepoints x n_voxels
    #     result = tf.transpose(result, [0, 2, 1])
    #     return result
    # def quick_sfmax(self, parameters, max_sfmax=50, **kwargs):
    #     """calculate_sfmax    
    #     aka high frequency cutoff. Useful summary statistic of whole CSF curve
    #     set the sensitivity = 1, then solve for the corresponding SF. 
    #     i.e., what is the highest possible SF we can detect    
    #     Can be infinte (with low width_r), so we set a max value 
    #     """
    #     log10_CSp = np.log10(parameters['CSp'])
    #     log10_SFp = np.log10(parameters['SFp'])
    #     sfmax = 10**((np.sqrt(log10_CSp/(parameters['width_r']**2)) + log10_SFp))
    #     if len(sfmax.shape)>=1:
    #         sfmax[sfmax>max_sfmax] = max_sfmax
    #     elif sfmax>max_sfmax:
    #         sfmax = max_sfmax        
    #     return sfmax
    
    # def quick_aulcsf(self, parameters, **kwargs):
    #     # parameters = self._get_parameters(parameters=parameters)
    #     parameters = parameters.values[np.newaxis, ...]  
    #     SF_levels = kwargs.get('SF_levels', np.array([ 0.5,  1.,   3.,   6.,  12.,  18. ]))
    #     normalize_AUC = kwargs.get('normalize_AUC', True)
    #     stim_sequence = pd.DataFrame({'SF':SF_levels, 'CON':SF_levels*0}).astype('float32')
    #     log_SF_levels = np.log10(SF_levels)#.reshape(1,-1)
    #     # Generate grid to make the CSF     
    #     csf_curve = self._truncparabola(
    #         stim_sequence = stim_sequence,
    #         parameters = parameters
    #         ).numpy()
    #     logcsf_curve = np.log10(csf_curve)    
    #     logcsf_curve[logcsf_curve<0] = 0 # Cannot have negative logCSF
    #     aulcsf = np.trapz(logcsf_curve.T, x=log_SF_levels, axis=0) 
    #     return aulcsf

    # @tf.function
    # def _truncparabola(self, stim_sequence, parameters):
    #     # n_batches x n_populations x  n_grid_spaces
    #     SF_seq = stim_sequence[:, 0][tf.newaxis, tf.newaxis, :]

    #     # Unpack parameters with broadcasting                
    #     # n_batches x n_populations x n_grid_spaces (broadcast)
    #     width_r = parameters[:, :, 0, tf.newaxis]
    #     SFp = parameters[:, :, 1, tf.newaxis]
    #     CSp = parameters[:, :, 2, tf.newaxis]
    #     lowSFtrunc = parameters[:, :, 3, tf.newaxis]        
        
    #     # Safeguard against log of non-positive values
    #     SF_seq_safe = tf.maximum(SF_seq, 1e-8)
    #     SFp_safe = tf.maximum(SFp, 1e-8)
    #     CSp_safe = tf.maximum(CSp, 1e-8)
    #     lowSFtrunc = tf.maximum(lowSFtrunc, 1e-8) # linear 

    #     # Logarithmic transformations
    #     log_SF_seq  = log10(SF_seq_safe)
    #     log_SFp     = log10(SFp_safe)
    #     log_CSp     = log10(CSp_safe)
        
    #     K = log10(0.5)
    #     logWidth = (10**width_r)*log10(2)/2

    #     logP = log_CSp + K * ((1/logWidth) * (log_SF_seq-log_SFp))
        
    #     truncHalf = CSp-lowSFtrunc

    #     left_condition = tf.math.logical_and(tf.less(logP, truncHalf), tf.less(log_SF_seq, log_SFp))
    #     leftCSF = tf.cast(left_condition, dtype=logP.dtype) * truncHalf

    #     # rightCSF
    #     right_condition = tf.math.logical_or(tf.greater_equal(logP, truncHalf), tf.greater(log_SF_seq, log_SFp))
    #     rightCSF = tf.cast(right_condition, dtype=logP.dtype) * logP

    #     # logCSF
    #     csf = leftCSF + rightCSF              
    #     return csf 

    # @tf.function
    # def _apply_crf(self, stim_sequence, parameters,csf):
    #     # n_batches x n_populations x  n_grid_spaces
    #     CON_seq = stim_sequence[:, 1][tf.newaxis, tf.newaxis, :]    
    #     # n_batches x n_populations x n_grid_spaces (broadcast)        
    #     crf_exp = parameters[:, :, 4, tf.newaxis]
    #     amplitude = parameters[:, :, 5, tf.newaxis]        
    #     # Contrast sensitivity
    #     cthresh = 100 / tf.clip_by_value(csf, 1e-1, 1e6) # Prevent extremes prev 1e-6...
    #     ncsf_resp = ((CON_seq ** crf_exp) / (CON_seq ** crf_exp + cthresh ** crf_exp)) * amplitude
    #     return ncsf_resp    
    
    # @tf.function
    # def _get_csf(self, stim_sequence, parameters):
    #     csf = self._asymetric_parabola(stim_sequence, parameters)
    #     ncsf_resp = self._apply_crf(stim_sequence, parameters, csf)

    #     return ncsf_resp  