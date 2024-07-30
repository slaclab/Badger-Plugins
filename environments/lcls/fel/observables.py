
import logging
import time
from badger.stats import percent_80
from badger.errors import BadgerEnvObsError
import numpy as np


def get_intensity_n_loss(hxr, points, loss_pv, fel_channel, interface):
    # At lcls the repetition is 120 Hz and the readout buf size is 2800.
    # The last 120 entries correspond to pulse energies over past 1 second.

    logging.info(f'Get value of {points} points')

    # Sleep for a while to get enough data
    try:
        rate = interface.get_value('EVNT:SYS0:1:LCLSBEAMRATE')
        logging.info(f'Beam rate: {rate}')
        nap_time = points / (rate * 1.0)
    except Exception as e:
        nap_time = 1
        logging.warn(
            'Something went wrong with the beam rate calculation. Let\'s sleep 1 second.')
        logging.warn(f'Exception was: {e}')

    time.sleep(nap_time)

    if hxr:
        PV_gas = f'GDET:FEE1:{fel_channel}:ENRCHSTCUHBR'
    else:  # SXR
        PV_gas = 'EM1K0:GMD:HPS:milliJoulesPerPulseHSTCUSBR'
    try:
        results_dict = self.interface.get_values([PV_gas, loss_pv])
        intensity_raw = results_dict[PV_gas][-points:]
        loss_raw = results_dict[loss_pv][-points:]
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
            gas = interface.get_value('EM1K0:GMD:HPS:milliJoulesPerPulse')

            return gas, gas, gas, 0, 0
    
def get_loss(points, loss_pv, interface):  # if only loss is observed
    logging.info(f'Get value of {points} points')

    try:
        rate = interface.get_value('EVNT:SYS0:1:LCLSBEAMRATE')
        logging.info(f'Beam rate: {rate}')
        nap_time = points / (rate * 1.0)
    except Exception as e:
        nap_time = 1
        logging.warn(
            'Something went wrong with the beam rate calculation. Let\'s sleep 1 second.')
        logging.warn(f'Exception was: {e}')

    time.sleep(nap_time)

    try:
        loss_raw = interface.get_value(loss_pv)[-points:]
        ind_valid = ~np.isnan(loss_raw)
        loss_valid = loss_raw[ind_valid]
        loss_p80 = percent_80(loss_valid)

        return loss_p80
    except Exception:  # we don't have scalar input for loss
        raise BadgerEnvObsError

def is_pulse_intensity_observed(observable_names):
    return len([name for name in observable_names if
                name.startswith('pulse_intensity')])