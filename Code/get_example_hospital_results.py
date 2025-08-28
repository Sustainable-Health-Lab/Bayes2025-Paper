# find example hospital results and enforce suppression rules
from util import *
import numpy as np
from pathlib import Path

def check_results_include(results):
    required_vars = [
            'n_w',
            'u_w',
            'n_p',
            'u_p',
            'fep', 'fes',
            'mwp', 'mws',
            'day_ind_on_0_199',
            'min_beneficiary_count',
            'min_non_beneficiary_count',
            'diffSignal',
            'cilow',
            'cihigh',
            'dbdfe',
            'dbdfe_stat',
            'wmpcd',
            'pmpcd',
            'w_util',
            'p_util'
            ]
    missing = [var for var in required_vars if var not in results] 
    assert missing == []

def generate_example_hospital_results(path_to_output):
    results = {}
    example_hospital_id = 330154
    function = 'HospiceTreat'
    udata = get_first_utilization_data(example_hospital_id, function)

    results['hospitalID'] = example_hospital_id

    # Basic Utilization Numbers
    n_w = udata.n_w
    n_p = udata.n_p
    results['n_w'] = n_w
    results['n_p'] = n_p
    results['u_w'] = udata.get_u_w()
    results['u_p'] = udata.get_u_p()

    # suppression information and calculations
    day_index = np.arange(0,200)
    min_ben_count = udata.get_diff_min_beneficiary_count()
    ge_eleven_ben = min_ben_count >= 11
    min_non_ben_count = udata.get_diff_min_non_beneficiary_count()
    ge_eleven_non_ben = min_non_ben_count >= 11
    safe_to_export_indicies = ge_eleven_non_ben & ge_eleven_ben
    day_index_sup = day_index[safe_to_export_indicies]

    results['day_ind_on_0_199'] = day_index_sup
    results['min_beneficiary_count'] = min_ben_count[day_index_sup]
    results['min_non_beneficiary_count'] = min_non_ben_count[day_index_sup]
    
    # utilization
    w_util = udata.get_w_utilization_signal()
    p_util = udata.get_p_utilization_signal()
    results['w_util'] = w_util[day_index_sup]
    results['p_util'] = p_util[day_index_sup]


    # diffSignal
    diffSignal = udata.get_diff_signal()
    sup_diffSignal = diffSignal[day_index_sup]
    results['diffSignal'] = sup_diffSignal

    # credible intervals
    cilow, cihigh = udata.get_diff_95_cis()
    results['cilow'] = cilow[day_index_sup]
    results['cihigh'] = cihigh[day_index_sup]

    # dbdfe
    dbdfe_stat, dbdfe = udata.get_dbd_fe()
    sup_dbdfe_stat = dbdfe_stat[day_index_sup]
    sup_dbdfe = dbdfe[day_index_sup]
    results['dbdfe'] = sup_dbdfe
    results['dbdfe_stat'] = sup_dbdfe_stat

    # wmpcd
    wmpcd = udata.get_wm_pcd()
    results['wmpcd'] = wmpcd[day_index_sup]

    # pmpcd
    pmpcd = udata.get_pm_pcd()
    results['pmpcd'] = pmpcd[day_index_sup]

    # Fisher Exact Results
    assert day_index_sup[-1] == 199
    results['fep'] = sup_dbdfe[-1]
    results['fes'] = sup_dbdfe_stat[-1]

    # Mann Whitney Results
    assert n_w + n_p > 11
    results['mws'], results['mwp']= udata.get_mw()
    
    check_results_include(results)

    with open(path_to_output, 'w') as f:
        json.dump(results, f, cls=NumpyEncoder)
    print(f'Saved example hospital results to {path_to_output}')

def main():
    output_dir = Path('/drives/56219-Linux/56219dua/Project2017/4-Analysis/3-Analysis/AidanPaper/Export/')
    save_path = output_dir / 'example_hospital_results.json'
    generate_example_hospital_results(save_path)


if __name__ == '__main__':
    main()
