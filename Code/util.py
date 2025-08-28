from pathlib import Path
import pandas as pd
import numpy as np
from typing import TypedDict, List, Tuple
from scipy.stats import beta, fisher_exact, mannwhitneyu
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize 
import os
import json
import re
from tqdm import tqdm


project = 'Project2017'
OUTPUT_PATH = Path(f'/drives/56219-Linux/56219dua/{project}/3-Output/')
ANALYSIS_PATH = Path(f'/drives/drive1/56219dua/{project}/4-Analysis/3-Analysis/AidanPaper/')

class NumpyEncoder(json.JSONEncoder):
    def default(self,obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float16, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        return json.JSONEncoder.default(self, obj)

class UtilizationData():

    def __init__(self, 
        hospitalID: int,
        function: str,
        n_w: int,
        n_p: int,
        w_first_days: List[int],
        p_first_days: List[int]):

        self.hospitalID = hospitalID
        self.function = function
        self.n_w = n_w
        self.n_p = n_p
        self.w_first_days = w_first_days
        self.p_first_days = p_first_days

        # hidden attributes with getters
        self._w_ut_signals: np.ndarray | None = None
        self._p_ut_signals: np.ndarray | None = None
        self._diff_signal: np.ndarray | None = None

        self._diff_dist: np.ndarray | None = None # approximation of the sampling distribution of the difference signal
        self._diff_cis: Tuple[np.ndarray, np.ndarray] | None = None # diff confidence intervals
        self._wm_pcd: np.ndarray | None = None # probability difference > 0.05 (w more)
        self._pm_pcd: np.ndarray | None = None # probability difference < -0.05 (poc more)
        self._dbd_fe: np.ndarray | None = None # day by day fisher exact

        self._mws: float | None = None # mann whitney u stat
        self._mwp: float | None = None # mann whitney p value

    def get_w_ut_signals(self):
        if self._w_ut_signals is None:
            t = np.arange(0,200)
            self._w_ut_signals = np.zeros((self.n_w, 200))
            for i, d in enumerate(self.w_first_days):
                self._w_ut_signals[i] = (d < t).astype(int)
        return self._w_ut_signals

    def get_w_utilization_signal(self):
        return np.mean(self.get_w_ut_signals(), axis=0)

    def get_p_ut_signals(self):
        if self._p_ut_signals is None:
            t = np.arange(0,200)
            self._p_ut_signals = np.zeros((self.n_p, 200))
            for i, d in enumerate(self.p_first_days):
                self._p_ut_signals[i] = (d < t).astype(int)
        return self._p_ut_signals

    def get_p_utilization_signal(self):
        return np.mean(self.get_p_ut_signals(), axis=0)

    
    def get_diff_signal(self):
        if self._diff_signal is None:
            self._diff_signal = np.mean(self.get_w_ut_signals(), axis=0) - np.mean(self.get_p_ut_signals(), axis=0)
            #self._diff_signal = np.sum(self.get_w_ut_signals(), axis=0)/self.n_w - np.sum(self.get_p_ut_signals(), axis=0)/self.n_p
            assert self.get_w_ut_signals().shape[0] == self.n_w, f"White Signals height ({self.get_w_ut_signals().shape[0]}) doesn't match number of patients ({self.n_w})"

            #print(f'diff signal shape: {self._diff_signal.shape}')
        return self._diff_signal

    def get_diff_dist(self):
        if self._diff_dist is None:
            num_samples = 5000 
            k_w = np.sum(self.get_w_ut_signals(), axis=0)
            w_alphas = 1 + k_w
            w_betas = 1 + self.n_w - k_w

            k_p = np.sum(self.get_p_ut_signals(), axis=0)
            p_alphas = 1 + k_p
            p_betas = 1 + self.n_p - k_p
            
            w_posteriors = beta(w_alphas, w_betas)
            p_posteriors = beta(p_alphas, p_betas)

            w_post_samples = w_posteriors.rvs(size=(num_samples, 200))
            p_post_samples = p_posteriors.rvs(size=(num_samples, 200))

            self._diff_dist = w_post_samples - p_post_samples
        return self._diff_dist

    def get_diff_95_cis(self):
        if self._diff_cis is None:
            #diff_signal = self.get_diff_signal()
            #print(f'diff_dist_shape: {diff_dist.shape}')

            diff_dist = self.get_diff_dist()
            self._diff_cis = np.percentile(diff_dist, [2.5, 97.5], axis=0)

            #print(f'diff_cis shape: {self._diff_cis.shape}')
            # u_w = np.sum(self.get_w_ut_signals(), axis=0)
            # u_p = np.sum(self.get_p_ut_signals(), axis=0)
            #for i, ci in enumerate(self._diff_cis.T):
                #print(f'day: {i+1}; ci: {ci}; ds: {diff_signal[i]}; in ci?: {ci[0] < diff_signal[i] < ci[1]}; u_w: {u_w[i]}; u_p: {u_p[i]}; tds: {(u_w[i]/self.n_w) - (u_p[i]/self.n_p)}')
        return self._diff_cis

    # get the probability that the diff is greater than 0.05 in magnitude 
    # favoring w 
    def get_wm_pcd(self):
        if self._wm_pcd is None:
            self._wm_pcd = np.mean(self.get_diff_dist() > 0.05, axis=0)
        return self._wm_pcd

    # get the probability that the diff is greater than 0.05 in magnitude 
    # favoring poc (poc more probability of clinical(ly singificant)  difference)
    def get_pm_pcd(self):
        if self._pm_pcd is None:
            self._pm_pcd = np.mean(self.get_diff_dist() < -0.05, axis=0)
        return self._pm_pcd

    # get day by day fisher exact results
    def get_dbd_fe(self)-> Tuple[np.ndarray, np.ndarray]:
        w_uts = self.get_w_ut_signals()
        p_uts = self.get_p_ut_signals()
        w_u = np.sum(w_uts, axis=0)
        p_u = np.sum(p_uts, axis=0)
        w_not_u = self.n_w - w_u
        p_not_u = self.n_p - p_u

        fes = np.zeros(200) # fisher exact statistics
        fep = np.zeros(200) # fisher exact p values
        for i in range(200):
            ctable = np.array([
                [w_u[i], w_not_u[i]],
                [p_u[i], p_not_u[i]],
                ])
            fes[i], fep[i] = fisher_exact(ctable, alternative="two-sided")
        return fes, fep

    def get_diff_min_beneficiary_count(self):
        return np.min((np.sum(self.get_p_ut_signals(), axis=0), np.sum(self.get_w_ut_signals(), axis=0)), axis=0)

    def get_diff_max_beneficiary_count(self):
        return np.max((np.sum(self.get_p_ut_signals(), axis=0), np.sum(self.get_w_ut_signals(), axis=0)), axis=0)

    def get_diff_min_non_beneficiary_count(self):
        return np.min((self.n_w - np.sum(self.get_p_ut_signals(), axis=0), self.n_w - np.sum(self.get_w_ut_signals(), axis=0)), axis=0)

    def get_mw(self):
        if (self._mws is not None) and (self._mwp is not None):
            return self._mws, self._mwp
        self._mws, self._mwp = mannwhitneyu(self.w_first_days, self.p_first_days)
        return self._mws, self._mwp
    
    def get_u_w(self):
        return len(self.w_first_days)

    def get_u_p(self):
        return len(self.p_first_days)

    def get_summary(self):

        fes, fep = self.get_dbd_fe()
        mws, mwp = self.get_mw()

        pcd_pos = (self.get_wm_pcd() > 0.95) | (self.get_pm_pcd() > 0.95)
        fep_pos = fep < 0.05
        fep_bp = np.sum(fep_pos & pcd_pos) # count of days dbdfe+ and pcd+
        fep_bn = np.sum(fep_pos & ~pcd_pos)
        fen_bp = np.sum(~fep_pos & pcd_pos)
        fen_bn = np.sum(~fep_pos & ~pcd_pos)

        return {
                "hospitalID": self.hospitalID,
                "function": self.function,
                "n_w": self.n_w,
                "n_p": self.n_p,
                "u_w": len(self.w_first_days),
                "u_p": len(self.p_first_days),
                "count_dbdfe_sig": np.sum(fep < 0.05),
                "dbdfe_sig_days": np.where(fep < 0.05)[0] + 1, 
                "count_wmpcd_sig": np.sum(self.get_wm_pcd() > 0.95),
                "wmpcd_sig_days": np.where(self.get_wm_pcd() > 0.95)[0] + 1,
                "count_pmpcd_sig": np.sum(self.get_pm_pcd() > 0.95),
                "pmpcd_sig_days": np.where(self.get_pm_pcd() > 0.95)[0] + 1,
                "mwp": mwp,
                "qmp": fep[-1],
                "fep_bp":fep_bp,
                "fep_bn":fep_bn,
                "fen_bp":fen_bp,
                "fen_bn":fen_bn,

                }


    
    def get_figure(self, supress=True):
        def plot_with_mask(ax, x, y, mask, **kwargs):
            mask_changes = np.diff(mask.astype(int))
            segment_boundaries = np.where(mask_changes)[0] + 1
            segments = np.concatenate(([0], segment_boundaries, [len(mask)]))
            num_segments = len(segments) - 1
            label = kwargs.pop('label')
            line = None
            for i, (start, end) in enumerate(zip(segments[:-1], segments[1:])):
                if mask[start]:
                    line = ax.plot(x[start:end], y[start:end], **kwargs)
            if line != None:
                line[0].set_label(label)


        t = np.arange(1,201)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5,11), height_ratios=[1,1])

        # plt.rcParams['font.family'] = 'Nimbus Roman'

        cmap = plt.get_cmap('cividis')
        altcmap = plt.get_cmap('viridis')
        norm = Normalize(vmin=0, vmax=1)

        mask = t > 0
        if supress == True:
            mask = ((self.get_diff_min_beneficiary_count() == 0) & (self.get_diff_max_beneficiary_count() >=11)) | (self.get_diff_min_beneficiary_count() >= 11)
            # print(mask)
            

        # First Subplot
        #ax1.plot(t[mask], self.get_diff_signal()[mask], label='Difference Signal', color=cmap(norm(0)))
        plot_with_mask(ax1, t, self.get_diff_signal(), mask, label='Difference Signal', color=cmap(norm(0)))

        ax1.fill_between(t[mask], self.get_diff_95_cis()[0][mask], self.get_diff_95_cis()[1][mask], alpha=0.2, label='95% CI', color=cmap(norm(0.2)))
        ax1.fill_between(t[mask], -1, 1, where=(self.get_wm_pcd()[mask] > 0.95), alpha=0.2, color=altcmap(norm(0.4)), label="P(Difference Signal > 0.05) > 0.95 (W More)")
        ax1.fill_between(t[mask], -1, 1, where=(self.get_pm_pcd()[mask] > 0.95), alpha=0.2, color=cmap(norm(0.8)), label="P(Difference Signal < -0.05) > 0.95 (POC More)")
        ax1.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
        ax1.set_ylim(-1,1)
        ax1.set_title("Difference Signal")
        ax1.set_xlabel("Days")
        ax1.legend()

        # Second Subplot
        # ax2.plot(t[mask], self.get_wm_pcd()[mask], label="P(Difference Signal > 0.05) (W More)", color=altcmap(norm(0)))
        plot_with_mask(ax2, t, self.get_wm_pcd(), mask, label="P(Difference Signal > 0.05) (W More)", color=altcmap(norm(0)))

        # ax2.plot(t[mask], self.get_pm_pcd()[mask], label="P(Difference Signal < -0.05) (POC More)", color=cmap(norm(1)))
        plot_with_mask(ax2, t, self.get_pm_pcd(), mask, label="P(Difference Signal < -0.05) (POC More)", color=cmap(norm(1)))

        ax2.axhline(y=0.95, color='gray', linestyle='-', linewidth=0.5)
        fes, fep = self.get_dbd_fe()
        fe_mask = mask & (fep < 0.05)
        ax2.fill_between(t, 0, 1, where=fe_mask, alpha=0.2, color=cmap(norm(0.2)), label='Fisher Exact p < 0.05')
        # ax2.plot(t[mask], np.mean(self.get_w_ut_signals(), axis=0)[mask], label='White Utilization', color=altcmap(norm(.5)))
        plot_with_mask(ax2, t, np.mean(self.get_w_ut_signals(), axis=0), mask, label='White Utilization', color=altcmap(norm(.5)))

        # ax2.plot(t[mask], np.mean(self.get_p_ut_signals(), axis=0)[mask], label='POC Utilization', color=cmap(norm(.7)))
        plot_with_mask(ax2, t,np.mean(self.get_p_ut_signals(), axis=0), mask, label='POC Utilization', color=cmap(norm(.7)))


        ax2.set_title("Clinical Difference Probabilites and Day by Day Fisher Exact p Value")
        ax2.set_xlabel("Days")
        ax2.set_ylim(0,1)
        ax2.legend()

        # Adjust the layout 
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.25)

        # Include Other Results as Text

        mws, mwp = self.get_mw()

        text_array = [f'HospitalID: {self.hospitalID}', 
                      f'Function: {self.function}',
                      f'# White Patients: {self.n_w}',
                      f'# Patients of Color: {self.n_p}',
                      f'Quality Measures Fisher Exact Statistic {fes[-1]:.3f}, p={fep[-1]:.3f}',
                      f'Mann Whitney U Test Statistic {mws:.3f}, p={mwp:.3f}',
                      f'Supressed: {supress}'
                      ]

        text_content = '\n'.join(text_array)
        fig.text(0.1, 0.1, text_content, ha='left', va='bottom', fontsize=10, 
                 bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})


        return fig


    
    def __str__(self):
        fes, fep = self.get_dbd_fe()
        mws, mwp = self.get_mw()
 
        return "\n".join([
            f"",
            f"First Utilization Data for '{self.function}' at  hospital {self.hospitalID}",
            f"n_w: {self.n_w}",
            f"n_p: {self.n_p}",
            f"u_w: {len(self.w_first_days)}",
            f"u_p: {len(self.p_first_days)}",
            f"count_dbd_fe_sig: {np.sum(fep < 0.05)}",
            f"dbd_fe_sig_days: {np.where(fep < 0.05)[0] + 1}", 
            f"count_wm_pcd_sig: {np.sum(self.get_wm_pcd() > 0.95)}",
            f"wm_pcd_sig_days: {np.where(self.get_wm_pcd() > 0.95)[0] + 1}",
            f"count_pm_pcd_sig: {np.sum(self.get_pm_pcd() > 0.95)}",
            f"pm_pcd_sig_days: {np.where(self.get_pm_pcd() > 0.95)[0] + 1}",
            f"mwp: {mwp}"
        ])


#   __init__(hospitalID, function, n_w, n_p, w_first_days, p_first_days):


    
def save_hospital_figure(hospitalID: int, function: str, fig):
    hospital_figure_path = ANALYSIS_PATH / 'HospitalFigures'

    if not os.path.exists(hospital_figure_path / function):
        os.makedirs(hospital_figure_path / function)

    figure_file_name = f'hospitalID_{hospitalID}_{function}.pdf'
    save_path = hospital_figure_path / function / figure_file_name
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    print(f'Figure saved to {save_path}')
    plt.close()

def generate_n_file(function):
    pattern = re.compile(r'^hospitalID_\d+$')
    matching_dirs = []


    print('Generating n_file...') 
    for item in OUTPUT_PATH.iterdir():
        if item.is_dir() and pattern.match(item.name):
            matching_dirs.append(item.name)

    def get_n_total_n_function(utilization_file_path: Path, function: str = function)   -> Tuple[int, int]:
        ut = pd.read_csv(utilization_file_path)
        num_unique_patients = ut['patientID'].nunique()
        num_unique_patients_with_function = ut[ut['function'] == function]['patientID'].nunique()
        return num_unique_patients, num_unique_patients_with_function 

    print('Counting patients and writing n_file...')
    n_file_path = ANALYSIS_PATH / 'n_files' / f'n_{function}.csv'
    with open(n_file_path, 'w') as f:
        print('hospitalID,nWhiteTotal,nPOCTotal,nWhite,nPOC,Analyze', file=f)
        for dir_name in tqdm(matching_dirs):
            ut_dir = OUTPUT_PATH / dir_name / 'utilizations'
            nWhiteTotal, nWhite = get_n_total_n_function(ut_dir / f'{dir_name}_w_Utilization.csv')
            nPOCTotal, nPOC = get_n_total_n_function(ut_dir / f'{dir_name}_nw_Utilization.csv')
            analyze = int(nWhiteTotal + nPOCTotal >= 11)
            hospitalID = int(dir_name.removeprefix("hospitalID_"))
            print(f'{hospitalID}, {nWhiteTotal}, {nPOCTotal}, {nWhite}, {nPOC}, {analyze}', file=f)

    # for each hospital directory in output folder
    # open the 


def get_hospitals_list(test = False, function='HospiceTreat') -> List[int]:
    counts_path = ANALYSIS_PATH / 'n_files' / f'n_{function}.csv'
    # hospitaID, nWhiteTotal, nPOCTotal
    df = pd.read_csv(counts_path, skipinitialspace=True)
    # filtered_df = df[(df['nWhiteTotal'] >= 11) & (df['nPOCTotal'] >= 11)]
    # filtered_df = df[(df['nWhite'] >= 11) & (df['nPOC'] >= 11)] # we want where >= 11 of both groups RECIEVED HOSPICE
    print(df.head(10))
    print(df.columns)
    filtered_df = df[(df['nWhiteTotal'] + df['nPOCTotal']) >= 11] # we want where >= 11 of both groups RECIEVED HOSPICE



    if test == True:
        return { "hospitalIDs": filtered_df['hospitalID'].tolist()[:7],
                "nWhiteTotals": filtered_df['nWhiteTotal'].tolist()[:7],
                "nPOCTotals": filtered_df['nPOCTotal'].tolist()[:7]
                }

    return { "hospitalIDs": filtered_df['hospitalID'].tolist(),
            "nWhiteTotals": filtered_df['nWhiteTotal'].tolist(),
            "nPOCTotals": filtered_df['nPOCTotal'].tolist()
            }

    hids_df = pd.read_csv(hids_path, header = None)
    return [hid for sublist in hids_df.values.tolist() for hid in sublist]


def get_first_utilization_data(hospitalID:int , function: str, pr = False):
    hospital_string = 'hospitalID_' + str(hospitalID)
    
    white_file_name = hospital_string + '_w_Utilization.csv'
    poc_file_name = hospital_string + '_nw_Utilization.csv'
    
    hospital_util_path = OUTPUT_PATH / hospital_string / 'utilizations'

    white_util_path = hospital_util_path / white_file_name
    poc_util_path =  hospital_util_path / poc_file_name 

    w_df = pd.read_csv(white_util_path)
    p_df = pd.read_csv(poc_util_path)

    n_w = w_df['patientID'].nunique()
    n_p = p_df['patientID'].nunique()

    fun_w_df = w_df[w_df['function'] == function]
    fun_p_df = p_df[p_df['function'] == function]

    w_first_days = fun_w_df.groupby('patientID')['dayNumber'].min().to_numpy()
    p_first_days = fun_p_df.groupby('patientID')['dayNumber'].min().to_numpy()

    # out of range values are set to 1 or 200 respectively
    w_first_days[w_first_days < 1] = 1
    p_first_days[p_first_days < 1] = 1

    w_first_days[w_first_days > 200] = 200
    p_first_days[p_first_days > 200] = 200
    
    if pr == True: 
        print(f"Retrieving population sizes and first days of utilization \
of {function} for hosital {str(hospitalID)}.")
        print(f'n_w: {n_w}')
        print(f'n_p: {n_p}')
        print(f'w_first_days: {w_first_days} ({len(w_first_days)})')
        print(f'p_first_days: {p_first_days} ({len(p_first_days)})')

    return UtilizationData(hospitalID, 
                           function, 
                           n_w, n_p, 
                           w_first_days, 
                           p_first_days)

