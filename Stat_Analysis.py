import numpy as np
from scipy.io import loadmat
import scipy.stats
import pandas as pd
import sys
import scipy.stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pingouin as pg
# Redirect all output to a text file
pd.set_option('display.max_rows', None)  # Show all rows (use with caution)
data_type = "x1acrossall"  # Can be set to 'audio', 'eeg', etc.
test = ["TR4","TR5","TR6"]
tolerance = 1e-6
time_point = [1, 2, 3]
diff_found_audio = False
tests = [0,1,2]
operation = "compare"
combined = []
diff = []
def describe_complex_array(arr):
    arr = np.array(arr, dtype=np.complex128)
    magnitudes = np.abs(arr)
    phases = np.angle(arr)

    print("  Magnitude:")
    print(f"    Min: {magnitudes.min():.4g}")
    print(f"    Max: {magnitudes.max():.4g}")
    print(f"    Mean: {magnitudes.mean():.4g}")
    print(f"    Std: {magnitudes.std():.4g}")
                    
    print("  Phase (radians):")
    print(f"    Min: {phases.min():.4g}")
    print(f"    Max: {phases.max():.4g}")
    print(f"    Std: {phases.std():.4g}")


def real_imag_summary(arr):
    arr = np.array(arr, dtype=np.complex128)
    real_part = arr.real
    imag_part = arr.imag
    print(f"  Real: min={real_part.min():.4g}, max={real_part.max():.4g}, std={real_part.std():.4g}")
    print(f"  Imag: min={imag_part.min():.4g}, max={imag_part.max():.4g}, std={imag_part.std():.4g}")

test = [0,1,2]
time_point = [0,1,2]
# Restore normal output to console
sys.stdout.close()
sys.stdout = sys.__stdout__
if operation == "statistics" and data_type == "Cohxy":
    for test in tests:
        for time in time_point:

            py_data = loadmat(f'E:/psymsc5/Cohxy_py.mat')['Cohxy']
            mat_data = loadmat(f'E:/psymsc5//Cohxy_data.mat')['Cohxy']
            py = py_data[test,time,:,:].flatten()
            mat = mat_data[test,time,:,:].flatten()
            print(py.shape)
            py_normal = scipy.stats.shapiro(py)
            mat_normal = scipy.stats.shapiro(mat)
            count_py_greater = np.sum(py > mat)
            print(count_py_greater)
            # Anzahl Fälle, bei denen py < mat
            count_py_less = np.sum(py < mat)
            print(count_py_less)

            # Anzahl Fälle, bei denen py == mat
            count_equal = np.sum(py == mat)
            print(count_equal)


            print("Track TR{}, time point {} comparison".format(test+4,time))
            print("Shapiro pval python {}".format(py_normal.pvalue))
            print("Shapiro pval matlab {}".format(mat_normal.pvalue))

            if py_normal.pvalue < 0.05 and mat_normal.pvalue < 0.05:
                wilcox_coh = scipy.stats.wilcoxon(py, mat)
                print("Wilcoxon result: {}, stat {}".format(wilcox_coh.pvalue,wilcox_coh.statistic))

            else:
                ttest_coh = scipy.stats.ttest_rel(py,mat)

                print("dependent ttest result: {}, stat {}".format(ttest_coh.pvalue,ttest_coh.statistic))

    
    if data_type == "range":
        for test_case in test:
            for time in time_point:
                chan_freq_all = loadmat(f"chan_freq_all_{test_case}_py_{time-1}.mat")["chan_freq_all"].flatten()
                x1across_all = loadmat(f"x1across_all_{test_case}_py_{time-1}.mat")["x1across_all"].flatten()
                y1across_all = loadmat(f"y1across_all_{test_case}_py_{time-1}.mat")["y1across_all"].flatten()

                print("Track {}, time {}".format(test_case,time))
                print("chan_freq_all")
                describe_complex_array(chan_freq_all)
                real_imag_summary(chan_freq_all)
                #compare_means(chan_freq_all)
                
                print("x1across_all")
                describe_complex_array(x1across_all)
                real_imag_summary(x1across_all)
                #compare_means(x1across_all)

                print("y1across_all")
                describe_complex_array(y1across_all)
                real_imag_summary(y1across_all)
                #compare_means(y1across_all)

