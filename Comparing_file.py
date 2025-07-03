import scipy.io
import numpy as np
import os
from scipy.io import loadmat

import pandas as pd
import csv

def all_comparison(py_data, mat_data, tolerance = 1e-6):
    results = []
    match = []
    combined = []
    diff_found = False
    print(f"shape pydats: {py_data.shape} and matdata {mat_data.shape}")
    for k in range(py_data.shape[0]):
        for j in range(py_data.shape[1]):
            for l in range(py_data.shape[2]):
                a = py_data[k, j,l]
                b = mat_data[k, j,l]
                index = f"{k},{j},{l}"
                if not np.isclose(a, b, atol=tolerance):
                    #print(f"❌ Mismatch at [{k},{j},{l}]: Python={a}, MATLAB={b}")
                    #print(f"Difference: {abs(a - b)}, python: {a}, matlab: {b}")
                    diff_found = True
                    combined.append([index,a, b,(a-b),"mismatch"])
                #else:
                    #print(f"✅ Match [{k},{j},{l}]: Python={a}, MATLAB={b}")
    if diff_found == False:
        combined.append(["-","-","-","-","all within tolerance"])
    df = pd.DataFrame(combined, columns=['index','py_value', 'mat_value','diff_py_mat','match'])
    return df

def Cohxy_comparison(py_data,mat_data,tolerance, t, index):
    results = []
    match = []
    t = t -1
    combined = []
    diff_found = False
    print(f"Cohxy shape {py_data.shape} mat cohxy shape {mat_data.shape}")
    for k in range(py_data.shape[2]):
        for l in range(py_data.shape[3]):
            a = py_data[index,t, k, l]
            b = mat_data[index,t, k, l]
            index_cohxy = f"[{index},{t},{k},{l}]"
            print(index_cohxy)
            if not np.isclose(index, a, b, atol=tolerance):
                #print(f"❌ Mismatch  [{test}, {(time-1)}, {k}, {l}]: Python={a}, MATLAB={b}")
                print(f"Difference: {abs(a - b)}")
                diff_found = True
                match.append("mismatch")
                combined.append([index,a, b,(a-b),match])
            #else:
                #print(f"✅ Match [{test}, {time-1}, {k}, {l}]: Python={a}, MATLAB={b}")
        if not diff_found:
            combined.append(["-","-","-","-","all within tolerance"])
        df = pd.DataFrame(combined, columns=['index_cohxy','py_value', 'mat_value','diff_py_mat','match'])
    return df    

def mean_comparison(py_data, mat_data, tolerance = 1e-6):
    print(f"shape pydats: {py_data.shape} and matdata {mat_data.shape}")
    diff = []
    results = []
    match = []
    combined = []
    diff_found = False
    for k in range(py_data.shape[0]):
        for j in range(py_data.shape[1]):
            a = py_data[k, j]
            b = mat_data[k, j]
            index = f"{k},{j}"
            index = f"[{k},{j}]"  
            if not np.isclose(a, b, atol=tolerance):
            #    print(f"❌ Mismatch at [{k},{j}]: Python={a}, MATLAB={b}")
            #    print(f"Difference: {abs(a - b)}, python: {a}, matlab: {b}")
                diff_found = True
                combined.append([index,a, b,(a-b),"mismatch"])
            #else:
            #    print(f"✅ Match [{k},{j}]: Python={a}, MATLAB={b}")
    if diff_found == False:
        combined.append(["-","-","-","-","all within tolerance"])
    df = pd.DataFrame(combined, columns=['index','py_value', 'mat_value','diff_py_mat','match'])
    return df


def load_data(data_type, test_case, time_point, abs1):
    # This function returns the loaded Python and MATLAB data arrays for given params
    base_path = 'E:/psymsc5/raw_data'
    print(data_type)
    print(f"Time point {time_point}, test case {test_case}")
    if data_type == "Cohxy":
        py_file = f'{base_path}{data_type}_py.mat'
        mat_file = f'{base_path}{data_type}_data.mat'
        key = data_type
    elif data_type.endswith('all') or data_type.endswith('all_abs'):
        py_file = f"{base_path}{data_type}_{test_case}_py_{time_point-1}{abs1}.mat"
        mat_file = f"{base_path}{data_type}_{test_case}_{abs1}{time_point}.mat"
        key = data_type
    elif data_type.endswith('mean') or data_type.endswith('mean_abs'):
        py_file = f"{base_path}{data_type}_{test_case}_py_{abs1}{time_point-1}.mat"
        mat_file = f"{base_path}{data_type}_{test_case}_{abs1}{time_point}.mat"
        key = data_type
    elif data_type == 'audio':
        py_file = f"{base_path}All_sound_cochl_stft_Py_{test_case}_{time_point-1}.mat"
        mat_file = f"{base_path}All_sound_cochl-stft_{test_case}_{time_point}.mat"
        key = 'All_sound_cochl_stft'
    elif data_type == 'eeg':
        py_file = f"{base_path}EEG_fft_Py_{test_case}_{time_point-1}.mat"
        mat_file = f"{base_path}EEG_fft_{test_case}_{time_point}.mat"
        key_py = 'EEG_fft'
        key_mat = 'EEG_fft'
        py_data = loadmat(py_file)[key_py]
        mat_data = loadmat(mat_file)[key_mat]
        return py_data, mat_data
    else:
        raise ValueError(f"Unknown data_type {data_type}")

    py_data = loadmat(py_file)[key]
    mat_data = loadmat(mat_file)[key]
    #print("pydata shape {}".format(py_data.shape))
    #print("mat_data {}".format(mat_data.shape))
 
    return py_data, mat_data


def run_comparisons(data_type, tests, time_points, abs1='', tolerance=1e-6):
    for index, test_case in enumerate(tests):
        for t in time_points:
            print(f"Time point: {t}")
            py_data, mat_data = load_data(data_type, test_case, t, abs1)
            if data_type.endswith('all') or data_type.endswith('all_abs') or data_type == 'eeg':
                df = all_comparison(py_data, mat_data, tolerance)
            elif data_type == "Cohxy":
                df = Cohxy_comparison(py_data,mat_data,tolerance,t,index)
            else:
                df = mean_comparison(py_data, mat_data, tolerance)
            
            # Save the results CSV
            filename = f"py_vs_mat_{data_type}{abs1}_{test_case}_{t}.csv"
            df.to_csv(filename, index=False)
            print(f"Saved comparison results to {filename}\n")



# y1across(abs),x1across(abs), chan_freq_all
abs1 = ""
data_type = "Cohxy"  # or 'audio', 'eeg', 'y1acrossmean', etc.
tests = ["TR4", "TR5", "TR6"]
time_points = [1, 2, 3]
tolerance = 1e-6

run_comparisons(data_type, tests, time_points, abs1, tolerance)
