import os, glob, pickle
import numpy as np 
from scipy.ndimage import gaussian_filter1d
from scipy.io import savemat
from itertools import groupby
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import GraphicalLassoCV
from sklearn.cluster import KMeans
from nilearn.maskers import NiftiMasker
from nilearn.connectome import cov_to_corr
import nitime.timeseries as nts 

class MatrixInWindow:
    def __init__(self, input_img, ROI, TR, f_ub, f_lb, window_size, max_iter, tol):
        self.input_img = input_img
        self.ROI = ROI
        self.TR = TR
        self.f_ub = f_ub
        self.f_lb = f_lb
        self.window_size = window_size
        self.max_iter = max_iter
        self.tol = tol

    def TsROI(self):
        ROIlist = sorted(glob.glob(os.path.join(self.ROI, "*.nii")))
        ts_ROI_ = []
        for i in range(len(ROIlist)):
            masker = NiftiMasker(mask_img=ROIlist[i], standardize=True, low_pass=self.f_ub, high_pass=self.f_lb, t_r=self.TR, verbose=1)
            ts_ = masker.fit_transform(self.input_img)
            ts_ROI_.append(ts_.mean(axis=1))
        ts_ROI = np.squeeze(np.array(ts_ROI_))
        self.ts_ROI = ts_ROI
    
    def GaussianFilter(self):
        step_window = 1

        rsfmri_vol = self.ts_ROI.shape[1]
        total_window = int(np.floor((rsfmri_vol - self.window_size) / step_window + 1))

        self.step_window = 1
        self.total_window = total_window

        gf_ = [0.0] * self.window_size
        if self.window_size % 2 == 0:
            mid1_ = int(self.window_size / 2) - 1
            mid2_ = int(self.window_size / 2)
            gf_[mid1_] = 1.0
            gf_[mid2_] = 1.0
            gf = gaussian_filter1d(gf_, sigma=3)
        else:
            mid_ = self.window_size // 2
            gf_[mid_] = 1.0
            gf = gaussian_filter1d(gf_, sigma=3)        
        self.gf = gf

    def DFCMatrix(self):
        corr_list = []

        for i in range(self.total_window):
            start_window_ = self.step_window * i
            end_window_ = start_window_ + self.window_size
            ts_window = nts.TimeSeries(self.ts_ROI[:,start_window_:end_window_], sampling_interval=self.TR)

            ts_window_g_ = []
            for j in range(len(self.gf)):
                ts_window_g_.append(ts_window.data[:,j] * self.gf[j])
            ts_window_g = np.array(ts_window_g_, dtype=np.float64).T

            scaler = StandardScaler()
            ts_window_g_standard = scaler.fit_transform(ts_window_g.T)

            estimator = GraphicalLassoCV(max_iter=self.max_iter, tol=self.tol, verbose=True)
            with np.errstate(invalid='ignore'):
                cov_ = estimator.fit(ts_window_g_standard).covariance_
            corr_mat = cov_to_corr(cov_)
            corr_flat = corr_mat.ravel()
            corr_list.append(corr_flat)

        self.corr_list = corr_list

class DFCState:
    def __init__(self, DFCMat_list, n_clusters, n_init, max_iter, tol):
        self.n_clusters = n_clusters
        self.DFCMat_list = DFCMat_list
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol

    def Kmeans(self):
        # Before KMeans
        flaten_ = []
        for i in range(len(self.DFCMat_list)):
            with open(self.DFCMat_list[i], 'rb') as f:
                DFCMat = pickle.load(f)
                for j in range(int(DFCMat.total_window)):
                    flaten_.append(DFCMat.corr_list[j])
                
        flaten_forKMeans_withdiag = np.array(flaten_)
        diag_col_idx = np.unique(np.where(flaten_forKMeans_withdiag == 1)[1])
        self.diag_col_idx = diag_col_idx
        flaten_forKMeans = np.delete(flaten_forKMeans_withdiag, self.diag_col_idx, axis=1)

        # KMeans
        model = KMeans(n_clusters=self.n_clusters, init='k-means++', n_init=self.n_init, max_iter=self.max_iter, tol=self.tol, random_state=0)
        distance = model.fit_transform(flaten_forKMeans)
        state = model.fit_predict(flaten_forKMeans)

        self.distance = distance
        self.state = state

        # After KMeans
        all_window = flaten_forKMeans.shape[0]
        non_diag_num = flaten_forKMeans.shape[1]

        M_before_insertnan = np.array([state for i in range(non_diag_num)], dtype=np.float64).T
        nanlist = np.array([np.nan for i in range(all_window)])

        for i in range(len(diag_col_idx)):
            M_before_insertnan = np.insert(M_before_insertnan, diag_col_idx[i], nanlist, axis=1)
        M_after_insertnan = np.array(M_before_insertnan)
        self.M_after_insertnan = M_after_insertnan

        # State
        State_Mat_list = []
        ROI_num = len(self.diag_col_idx)
        
        for i in range(self.n_clusters):
            State_Mat_ = np.mean(flaten_forKMeans_withdiag[np.where(state == i)], axis=0)
            State_Mat = State_Mat_.reshape(ROI_num, ROI_num)
            State_Mat_list.append(State_Mat)

        self.StateMat_List = State_Mat_list

    def VisState(self):
        width = 4 * self.n_clusters
        fig, axes = plt.subplots(1, self.n_clusters, figsize=(width, 3.15), tight_layout=True)

        for i in range(len(self.StateMat_List)):
            sns.heatmap(self.StateMat_List[i], cmap='bwr', ax=axes[i], vmax=0.8, vmin=-0.8, square=True)
            fname = f'State {i}'
            axes[i].set_title(fname)

        plt.plot()
        plt.subplots_adjust(wspace=0.05)
        fname = "FC_State"
        resdir = "./Result"
        os.makedirs(resdir, exist_ok=True)
        fpath = os.path.join(resdir, fname)
        plt.savefig(fpath)

    def TemporalProperties(self):
        # Calculating fractional window
        fw_list = []
        for i in range(self.n_clusters):
            fw_list.append([])

        subj_num = len(self.DFCMat_list)
        state_persubj = np.split(self.state, subj_num)

        for i in range(subj_num):
            for j in range(self.n_clusters):    
                fw_sub = sum(state_persubj[i] == j) / 221 * 100
                fw_list[j].append(fw_sub)
        fw_arr = np.array(fw_list)
        self.fractional_window = fw_arr

        # Caclurating dwell time and number of transition
        dwell_time_list = []
        num_trans_list = []
        for i in range(self.n_clusters):
            dwell_time_list.append([])

        for i in range(subj_num):
            seq_window_same_state = [len([*group]) for k, group in groupby(state_persubj[i])]
            seq_window_same_state_arr = np.array(seq_window_same_state)
            seq_window_state_ = [np.unique([*group]) for k, group in groupby(state_persubj[i])]
            seq_window_state = np.array([int(k[0]) for k in seq_window_state_])

            for j in range(self.n_clusters):
                dwell_time_all = seq_window_same_state_arr[np.where(seq_window_state == j)]
                if len(dwell_time_all) != 0:
                    dwell_time_list[j].append(dwell_time_all.mean())
                else:
                    dwell_time_list[j].append(0)

            num_trans = len(seq_window_state) -1 
            num_trans_list.append(num_trans)

        dwell_time_arr = np.array(dwell_time_list)
        num_trans_arr = np.array(num_trans_list)

        self.dwell_time = dwell_time_arr
        self.num_trans = num_trans_arr

if __name__ == "__main__":
    """ Maxrix for each window """
    # folder path containing nifti (fMRI)
    subj_folder = "" 
    subj_list = glob.glob(os.path.join(subj_folder, "*.nii"))
    subj_list = sorted(subj_list)

    # folder path containing nifti (ROI)
    ROI = "" 

    # The following parameters to be set 
    TR = 2.5
    f_ub = 0.1
    f_lb = 0.01
    window_size = 22
    max_iter_window = 100
    tol_window = 1e-4

    # Main script
    for i in range(len(subj_list)):
        input_img = subj_list[i]
        model = MatrixInWindow(input_img=input_img, ROI=ROI, TR=TR, f_ub=f_ub, f_lb=f_lb, window_size=window_size, max_iter=max_iter_window, tol=tol_window)
        model.TsROI()
        model.GaussianFilter()
        model.DFCMatrix()

        fname = f"sub-{i+1:03}.pickle"
        fdir = "./DFCMatPerSubj"
        os.makedirs(fdir, exist_ok=True)
        fpath = os.path.join(fdir, fname)

        with open(fpath, mode='wb') as f:
            pickle.dump(model, f)

    """ DFC state and temporal properties """
    # Loading FC matrix per window created after executing DFCwindow_main.py
    DFCMat_list = glob.glob(os.path.join("./DFCMatPerSubj", "*.pickle"))
    DFCMat_list = sorted(DFCMat_list)

    # The following parameters to be set 
    n_clusters = 2
    n_init = 10
    max_iter_state = 300
    tol_state = 0.0001

    # Main script
    model_DFCstate = DFCState(DFCMat_list=DFCMat_list, n_clusters=n_clusters, n_init=n_init, max_iter=max_iter_state, tol=tol_state)
    model_DFCstate.Kmeans()
    model_DFCstate.VisState()
    model_DFCstate.TemporalProperties()

    Dic = {
        'Matrix': model_DFCstate.StateMat_List, 
        'Fractonal window': model_DFCstate.fractional_window, 
        'Dwell time': model_DFCstate.dwell_time, 
        'Number of trans': model_DFCstate.num_trans
    }
    resdir = "./Result"
    fname = 'DFCres.mat'
    fpath = os.path.join(resdir, fname)
    savemat(fpath, Dic)