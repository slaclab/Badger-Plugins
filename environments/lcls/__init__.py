import time
import numpy as np
from typing import Dict, List
from badger import environment
from badger.stats import percent_80
from badger.errors import BadgerEnvObsError, BadgerInterfaceChannelError
import logging
from .variables import env_vars 


class Environment(environment.Environment):

    name = 'lcls'
    variables = env_vars.variables
    observables = [
        'energy',
        'charge',
        'current',
        'beamrate',
        'beamsize_x',
        'beamsize_y',
        'beamsize_r',
        'beamsize_g',
        'pulse_intensity_p80',
        'pulse_intensity_mean',
        'pulse_intensity_median',
        'pulse_intensity_std',
        'pulse_intensity_std_relative',
        'beam_loss',
        'pulse_id'
    ]

    # Env params
    readonly: bool = False  # set to True if do not want to change the machine state
    points: int = 120
    hxr: bool = True  # if HXR is used

    fel_channel: str = '361'  # FEL channel for the HXR gas detector
    beamsize_monitor: str = '541'  # BPM channel for beam size
    loss_pv: str = 'LBLM:COL0:862:A:I0_LOSSHSTSCS'  # PV name for loss monitor

    use_check_var: bool = True  # if check var reaches the target value
    check_var_timeout: float = 3.0  # tumeout for the var check
    trim_delay: float = 3.0  # in second
    check_fault_timeout: float = 5.0  # in second

    epsilon: float = 1e-8  # avoid divided by zero in relative FEL jitter

    def get_bounds(self, variable_names):
        assert self.interface, 'Must provide an interface!'

        pvs_low = [v + '.DRVL' for v in variable_names]
        pvs_high = [v + '.DRVH' for v in variable_names]
        bounds_low = self.interface.get_values(pvs_low)
        bounds_high = self.interface.get_values(pvs_high)

        bound_outputs = {}
        for i, v in enumerate(variable_names):
            bound_outputs[v] = [bounds_low[pvs_low[i]], bounds_high[pvs_high[i]]]

        return bound_outputs

    def get_variables(self, variable_names: List[str]) -> Dict:
        assert self.interface, 'Must provide an interface!'

        channel_names = []
        for v in variable_names:
            if v.endswith(':BCTRL'):
                prefix = v[:v.rfind(':')]
                readback = prefix + ':BACT'
            else:
                readback = v
            channel_names.append(readback)

        # Fetch the interface until all values are not None
        time_start = time.time()
        channel_outputs = self.interface.get_values(channel_names)
        while None in channel_outputs.values():
            time.sleep(0.1 * np.random.rand())
            time_elapsed = time.time() - time_start
            if time_elapsed > self.check_var_timeout:
                raise BadgerInterfaceChannelError

            channel_outputs = self.interface.get_values(channel_names)

        variable_outputs = {v: channel_outputs[channel_names[i]]
                            for i, v in enumerate(variable_names)}

        return variable_outputs

    def set_variables(self, variable_inputs: Dict[str, float]):
        assert self.interface, 'Must provide an interface!'

        if self.readonly:
            return

        self.interface.set_values(variable_inputs)

        if not self.use_check_var:
            if self.trim_delay:
                time.sleep(self.trim_delay)  # extra time for stablizing orbits

            return

        variable_ready_flags = [v[:v.rfind(':')] + ':STATCTRLSUB.T'
                                for v in variable_inputs
                                if v.endswith(':BCTRL')]
        variable_status = self.interface.get_values(variable_ready_flags)

        time_start = time.time()
        while np.any(np.array(variable_status.values())):
            time.sleep(0.1 * np.random.rand())

            variable_status = self.interface.get_values(variable_ready_flags)

            time_elapsed = time.time() - time_start
            if time_elapsed > self.check_var_timeout:
                break

    def get_intensity_n_loss(self):
        # At lcls the repetition is 120 Hz and the readout buf size is 2800.
        # The last 120 entries correspond to pulse energies over past 1 second.
        hxr = self.hxr

        points = self.points
        logging.info(f'Get value of {points} points')

        # Sleep for a while to get enough data
        try:
            rate = self.interface.get_value('EVNT:SYS0:1:LCLSBEAMRATE')
            logging.info(f'Beam rate: {rate}')
            nap_time = points / (rate * 1.0)
        except Exception as e:
            nap_time = 1
            logging.warn(
                'Something went wrong with the beam rate calculation. Let\'s sleep 1 second.')
            logging.warn(f'Exception was: {e}')

        time.sleep(nap_time)

        if hxr:
            PV_gas = f'GDET:FEE1:{self.fel_channel}:ENRCHSTCUHBR'
        else:  # SXR
            PV_gas = 'EM1K0:GMD:HPS:milliJoulesPerPulseHSTCUSBR'
        PV_loss = self.loss_pv
        try:
            results_dict = self.interface.get_values([PV_gas, PV_loss])
            intensity_raw = results_dict[PV_gas][-points:]
            loss_raw = results_dict[PV_loss][-points:]
            ind_valid = ~np.logical_or(np.isnan(intensity_raw), np.isnan(loss_raw))
            intensity_valid = intensity_raw[ind_valid]
            loss_valid = loss_raw[ind_valid]

            gas_p80 = percent_80(intensity_valid)
            gas_mean = np.mean(intensity_valid)
            gas_median = np.median(intensity_valid)
            gas_std = np.std(intensity_valid)

            loss_p80 = percent_80(loss_valid)

            return gas_p80, gas_mean, gas_median, gas_std, loss_p80
        except Exception:  # if average fails use the scalar input
            if hxr:  # we don't have scalar input for HXR
                raise BadgerEnvObsError
            else:
                gas = self.interface.get_value('EM1K0:GMD:HPS:milliJoulesPerPulse')

                return gas, gas, gas, 0, 0

    def get_loss(self):  # if only loss is observed
        points = self.points
        logging.info(f'Get value of {points} points')

        try:
            rate = self.interface.get_value('EVNT:SYS0:1:LCLSBEAMRATE')
            logging.info(f'Beam rate: {rate}')
            nap_time = points / (rate * 1.0)
        except Exception as e:
            nap_time = 1
            logging.warn(
                'Something went wrong with the beam rate calculation. Let\'s sleep 1 second.')
            logging.warn(f'Exception was: {e}')

        time.sleep(nap_time)

        PV_loss = self.loss_pv
        try:
            loss_raw = self.interface.get_value(PV_loss)[-points:]
            ind_valid = ~np.isnan(loss_raw)
            loss_valid = loss_raw[ind_valid]
            loss_p80 = percent_80(loss_valid)

            return loss_p80
        except Exception:  # we don't have scalar input for loss
            raise BadgerEnvObsError

    def is_pulse_intensity_observed(self, observable_names):
        return len([name for name in observable_names if
                    name.startswith('pulse_intensity')])

    def is_beam_loss_observed(self, observable_names):
        return 'beam_loss' in observable_names

    def get_observables(self, observable_names: List[str]) -> Dict:
        assert self.interface, 'Must provide an interface!'

        # Make sure machine is not in a fault state
        self.check_fault_status()

        observe_gas = self.is_pulse_intensity_observed(observable_names)
        observe_loss = self.is_beam_loss_observed(observable_names)

        if observe_gas:
            intensity_p80, intensity_mean, intensity_median, intensity_std, \
                loss_p80 = self.get_intensity_n_loss()
        elif observe_loss:
            loss_p80 = self.get_loss()

        observable_outputs = {}
        mid = self.beamsize_monitor
        for obs in observable_names:
            if obs == 'energy':
                value = self.interface.get_value('BEND:DMPH:400:BDES')
            elif obs == 'charge':
                value = self.interface.get_value('SIOC:SYS0:ML00:CALC252')
            elif obs == 'current':
                value = self.interface.get_value('BLEN:LI24:886:BIMAX')
            elif obs == 'beamrate':
                value = self.interface.get_value('EVNT:SYS0:1:LCLSBEAMRATE')
            elif obs == 'beamsize_x':
                value = self.interface.get_value(f'OTRS:IN20:{mid}:XRMS')
            elif obs == 'beamsize_y':
                value = self.interface.get_value(f'OTRS:IN20:{mid}:YRMS')
            elif obs == 'beamsize_r':
                bs_x = self.interface.get_value(f'OTRS:IN20:{mid}:XRMS')
                bs_y = self.interface.get_value(f'OTRS:IN20:{mid}:YRMS')
                value = np.linalg.norm([bs_x, bs_y])
            elif obs == 'beamsize_g':
                bs_x = self.interface.get_value(f'OTRS:IN20:{mid}:XRMS')
                bs_y = self.interface.get_value(f'OTRS:IN20:{mid}:YRMS')
                value = np.sqrt(bs_x * bs_y)
            elif obs == 'beam_loss':
                value = loss_p80
            elif obs == 'pulse_intensity_p80':
                value = intensity_p80
            elif obs == 'pulse_intensity_mean':
                value = intensity_mean
            elif obs == 'pulse_intensity_median':
                value = intensity_median
            elif obs == 'pulse_intensity_std':
                value = intensity_std
            elif obs == 'pulse_intensity_std_relative':
                value = intensity_std / (intensity_mean + self.epsilon)
            elif obs == 'pulse_id':
                value = self.interface.get_value('PATT:SYS0:1:PULSEID')
            else:  # won't happen actually
                value = None

            observable_outputs[obs] = value

        return observable_outputs

    def check_fault_status(self):
        assert self.interface, 'Must provide an interface!'

        ts_start = time.time()
        while True:
            rate_MPS = self.interface.get_value('IOC:BSY0:MP01:PC_RATE',
                                                as_string=True)
            permit_BCS = self.interface.get_value('BCS:MCC0:1:BEAMPMSV',
                                                  as_string=True)

            if (rate_MPS == '120 Hz') and (permit_BCS == 'OK'):
                break
            else:
                ts_curr = time.time()
                dt = ts_curr - ts_start
                if dt > self.check_fault_timeout:
                    raise BadgerEnvObsError

                time.sleep(0.1 * np.random.rand())

    def get_system_states(self):
        assert self.interface, 'Must provide an interface!'

        ignore_small_value = lambda x: x if x > 10 else 0

        general_pvs = [
            'BEND:DMPH:400:BDES',
            'SIOC:SYS0:ML00:AO627',
            'BEND:DMPS:400:BDES',
            'SIOC:SYS0:ML00:AO628',
            # 'IOC:IN20:EV01:RG02_DESRATE',  # this one has to be treated specifically
            'SIOC:SYS0:ML00:CALC038',
            'SIOC:SYS0:ML00:CALC252',
            'BPMS:DMPH:693:TMITCUH1H',
            'BPMS:DMPS:693:TMITCUS1H',
        ]
        matching_quads = [
            'QUAD:IN20:361:BCTRL',
            'QUAD:IN20:371:BCTRL',
            'QUAD:IN20:425:BCTRL',
            'QUAD:IN20:441:BCTRL',
            'QUAD:IN20:511:BCTRL',
            'QUAD:IN20:525:BCTRL',
            'QUAD:LI21:201:BCTRL',
            'QUAD:LI21:211:BCTRL',
            'QUAD:LI21:271:BCTRL',
            'QUAD:LI21:278:BCTRL',
            'QUAD:LI26:201:BCTRL',
            'QUAD:LI26:301:BCTRL',
            'QUAD:LI26:401:BCTRL',
            'QUAD:LI26:501:BCTRL',
            'QUAD:LI26:601:BCTRL',
            'QUAD:LI26:701:BCTRL',
            'QUAD:LI26:801:BCTRL',
            'QUAD:LI26:901:BCTRL',
            'QUAD:LTUH:620:BCTRL',
            'QUAD:LTUH:640:BCTRL',
            'QUAD:LTUH:660:BCTRL',
            'QUAD:LTUH:680:BCTRL',
            'QUAD:LTUS:620:BCTRL',
            'QUAD:LTUS:640:BCTRL',
            'QUAD:LTUS:660:BCTRL',
            'QUAD:LTUS:680:BCTRL',
            'QUAD:LI21:221:BCTRL',
            'QUAD:LI21:251:BCTRL',
            'QUAD:LI24:740:BCTRL',
            'QUAD:LI24:860:BCTRL',
            'QUAD:LTUH:440:BCTRL',
            'QUAD:LTUH:460:BCTRL',
            'QUAD:IN20:121:BCTRL',
            'QUAD:IN20:122:BCTRL',
        ]
        extra_pvs = [
            'CAMR:IN20:186:XRMS',
            'CAMR:IN20:186:YRMS',
            'BLEN:LI21:265:AIMAX1H',
            'BLEN:LI24:886:BIMAX1H',
            'SIOC:SYS0:ML00:CALC038',
            'SIOC:SYS0:ML00:CALC252',
            'BEND:DMPH:400:BACT',
            'SIOC:SYS0:ML00:AO627',
            'ACCL:LI21:1:L1S_S_AV',
            'ACCL:LI21:1:L1S_S_PV',
            'ACCL:LI21:180:L1X_S_AV',
            'ACCL:LI21:180:L1X_S_PV',
            'ACCL:LI22:1:ADES',
            'ACCL:LI22:1:PDES',
            'ACCL:LI25:1:ADES',
            'ACCL:LI25:1:PDES',
        ]

        try:
            states_general = self.interface.get_values(general_pvs)
            states_quads = self.interface.get_values(matching_quads)
            states_extra = self.interface.get_values(extra_pvs)

            system_states = {
                'HXR electron energy [GeV]': states_general['BEND:DMPH:400:BDES'],
                'HXR photon energy [eV]': round(states_general['SIOC:SYS0:ML00:AO627']),
                'SXR electron energy [GeV]': states_general['BEND:DMPS:400:BDES'],
                'SXR photon energy [eV]': round(states_general['SIOC:SYS0:ML00:AO628']),
                'Rate [Hz]': self.interface.get_value('IOC:IN20:EV01:RG02_DESRATE', as_string=True),
                'Charge at gun [pC]': ignore_small_value(states_general['SIOC:SYS0:ML00:CALC038']),
                'Charge after BC1 [pC]': ignore_small_value(states_general['SIOC:SYS0:ML00:CALC252']),
                'Charge at HXR dump [pC]': ignore_small_value(states_general['BPMS:DMPH:693:TMITCUH1H'] * 1.602e-7),
                'Charge at SXR dump [pC]': ignore_small_value(states_general['BPMS:DMPS:693:TMITCUS1H'] * 1.602e-7),
            }
            system_states.update(states_quads)
            system_states.update(states_extra)
        except Exception as e:
            logging.warn(
                'Failed to get system states, will not save the requested system states.')
            logging.warn(f'Exception was: {e}')

            system_states = None

        return system_states
