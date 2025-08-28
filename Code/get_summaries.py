import numpy as np
from util import *
from scipy.stats import fisher_exact, mannwhitneyu
from tqdm import tqdm
import pandas as pd
import json
import matplotlib.pyplot as plt
import ast

def generate_data(save_path, function, gen_figures=False, TEST=False):
    '''
    Runs the analysis for all the hospitals and generates summaries.csv
    This DOES generate the hospital level figures.
    This DOES NOT generate the figures that use data from the summaries file.
    '''
    hl = get_hospitals_list(test=TEST)
    hlist, nWhiteTotals, nPOCTotals = hl['hospitalIDs'], hl['nWhiteTotals'], hl['nPOCTotals']
    
    hospitals_with_wrong_num = []
    failed_hospitals = []
    summaries = []
    for i,h in tqdm(enumerate(hlist)):
        udata = get_first_utilization_data(h, function)
        summary = udata.get_summary()
        if gen_figures:
            fig = udata.get_figure(supress=True)
            save_hospital_figure(udata.hospitalID, function, fig) 

        summaries.append(summary)
        if (summary['n_w'] != nWhiteTotals[i]) or (summary['n_p'] != nPOCTotals[i]):
            hospitals_with_wrong_num.append(h)
            print("WRONG NUM")
            print(f"n_w was {summary['n_w']} nWhiteTotal was {nWhiteTotals[i]}")
            print(f"p_w was {summary['n_p']} nPOCTotal was {nPOCTotals[i]}")
    
    
    print(f"There were {len(hospitals_with_wrong_num)} hospitals with inconsitancies")
    pd.DataFrame({"wrongN": hospitals_with_wrong_num}).to_csv(save_path / "h_wrong_num.csv")
    pd.DataFrame({"failedHospitals": failed_hospitals}).to_csv(save_path / "failed.csv")
    pd.DataFrame(summaries).to_csv(save_path / "summaries.csv", index=False)


def main():
    save_path = Path('/drives/drive1/56219dua/Project2017/4-Analysis/3-Analysis/AidanPaper')
    function = "HospiceTreat"
    generate_data(save_path, function)

if __name__ == '__main__':
    main()


