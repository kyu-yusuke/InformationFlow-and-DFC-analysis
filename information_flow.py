import os, glob
import numpy as np 
from nilearn.maskers import NiftiMasker
import nitime.analysis as nta 
import nitime.timeseries as nts 

def information_flow(input_img, ROI, TR, f_ub, f_lb, order):
    ROIlist = sorted(glob.glob(os.path.join(ROI, "*.nii")))
    ts_ROI_ = []
    for i in range(len(ROIlist)):
        masker = NiftiMasker(mask_img=ROIlist[i], standardize=True, low_pass=f_ub, high_pass=f_lb, t_r=TR)
        ts_ = masker.fit_transform(input_img)
        ts_ROI_.append(ts_.mean(axis=1))
    ts_ROI = np.squeeze(np.array(ts_ROI_))

    ts_obj = nts.TimeSeries(ts_ROI, sampling_interval=TR)
    G = nta.GrangerAnalyzer(ts_obj, order=order)
    freq_idx_G = np.where((G.frequencies > f_lb) * (G.frequencies < f_ub))[0]
    g_xy = np.mean(G.causality_xy[:, :, freq_idx_G], axis=-1)
    g_yx = np.mean(G.causality_yx[:, :, freq_idx_G], axis=-1)

    GC_xy = np.mean(G.causality_xy[:, :, freq_idx_G] - G.causality_yx[:, :, freq_idx_G], axis=-1)
    dGCA_xy = g_xy / (g_xy + g_yx)

    return GC_xy, dGCA_xy

if __name__ == "__main__":
    # fMRI img (nifti format)
    input_img = "" 

    # folder containing nifti format ROI
    ROI = "" 

    # The following parameters to be set 
    TR = 2.5
    f_ub = 0.1
    f_lb = 0.01
    order = 1

    information_flow(input_img=input_img, ROI=ROI, TR=TR, f_ub=f_ub, f_lb=f_lb, order=order)