def singleton(cls):
    instance = cls()
    instance.__call__ = lambda: instance
    return instance

@singleton
class settings:
    #binning and parsing:
    background_subtract_time      = .001   # s region at the beginning of a trace for background subtraction
    background_subtraction_method = ['time', 'pmt'] # choice of 'const', 'time', 'xy const', 'xy time'
    polarization_switching_offset = 9      # how many datapoints at the beginning of a trace with no scope offset before a polarization switching cycle starts.
    constant_inds                 = [6, 20]# these are the indices which indicate the beginning and end of the "aom" on time. this is used when averaging over all of the "aom on" data.
    subbin_choice                 = [1, 25]# this is the choice of time to average over for the measurement
    subbins                       = [[1, 25],[6, 20],[1, 8],[9, 17],[18, 25]]
    asymm_grouping                = 35
    avg_pmts                      = True
    voltage_to_count_rate         = 15 * 10**6

    # determination of tau:
    h_state_g_factor        = .0044
    bohr_magneton           = 8794.1 # rad/s/mG
    h_state_magnetic_moment = bohr_magneton * h_state_g_factor
    current_to_magnetic_field_ratio = 19.7198/15 # mG/mA;
    tau_guess               = 1*10^-3 # s

    # cut parameters
    signal_threshold = .010
    tau_minimum      = .75*10**-3
    tau_maximum      = 1.5*10**-3
    contrast_minimum = .35
    contrast_maximum = 1.25
    number_threshold = 3