
from scipy.io import savemat
import os
import soundfile as sf
import sys
import gc
import pandas as pd
import mne
import numpy as np
import scipy
import scipy.fft
from scipy import signal
from matplotlib import mlab
import matplotlib.pyplot as plt
import scipy.signal as signal  # Ensure this is correct
from scipy.signal import get_window
import librosa
from scipy.signal import resample_poly, filtfilt, butter, hilbert, lfilter, detrend
from fractions import Fraction
chan_freq_mean = []
y1across_mean = []
x1across_mean = []


Cohxy_ini = False
#Precision for accurate calculation
np.set_printoptions(precision=16
, suppress=False)

def detrend_poly(channel_data, degree=2):
    # Create time vector
    tim = np.arange(len(channel_data))
    
    # Fit a polynomial of the given degree
    p = np.polyfit(tim, channel_data, deg=degree)
    
    # Evaluate the polynomial at each time point
    trend = np.polyval(p, tim)
    
    return channel_data - trend


Test = ['TR4','TR5','TR6']
time_points = np.array([
[1097.776, 1140.6],
[1184.0, 1221.844],
[1246.504, 1295.0]
])

#transform cochlear position back into Hz
def inv_cochlear_map(x, Fmax=2e4):
    f = 456 * (10**(0.021 * x) - 0.8)
    
    # Normalize based on the given function
    if Fmax != 57e3:
        f = f * Fmax / inv_cochlear_map(100, 57e3)  #
    return f


#transform Hz into cochleear position
def cochlear_map(f, Fmax=2e4):
    if Fmax != 57e3:
        f = f * inv_cochlear_map(100, 57e3) / Fmax
    x = np.log10(f / 456 + 0.8) / 0.021
    return x    

###General Note
#Usage of "[..] = None", Always delete variables for efficient memory usage whenever its not needed anymore

for index_test, iTest in enumerate(Test): 
    filename = f'A080721_001_{iTest}_raw.fif'
    base_path = 'E:/Psymsc5'
    dataset = os.path.join(base_path, filename)
    # Load EEG data, define as single continous trial and preprocess
    raw = mne.io.read_raw_fif(dataset, preload=True)  # ✅ CORRECT
    fsample = raw.info['sfreq']  # Extracting the sampling frequency from the EEG file
    events = mne.make_fixed_length_events(raw, duration=raw.times[-1])
    epochs = mne.Epochs(raw, events, tmin=0, tmax=raw.times[-1], baseline=None)    
    data = epochs.get_data()  # This will be the equivalent of ft_preprocessing output

    #Loop for timepoints starting
    for index, iTime in enumerate(time_points):

        #each timepoint of interest is converted to the sampling frequency
        enter_time = iTime* 250 / 1

        # Read the audio file
        audio_file = f'E:/Psymsc5/audio_A080721_001_{iTest}.wav'
        Y, FS = sf.read(audio_file, dtype='float64')  

        # Convert to mono if stereo
        if Y.ndim > 1:
            Y_mono = np.mean(Y, axis=1)
        else:
            Y_mono = Y
        
        # Downsample recordings with high sampling rate
        # checks if the Nyquist frequency (which is half the sampling frequency, FS/2) is greater than 20,000 Hz.
        fs = 20000
        if FS / 2 > 20000:
            #Calculate the resampling factors
            ratio = Fraction(fs, FS).limit_denominator()
            P, Q = ratio.numerator, ratio.denominator
            #resample
            YMonores = resample_poly(Y_mono, P, Q)
        else:
            YMonores = Y_mono
            fs = FS     

        #Define high and low edges for the cochlear map
        lo_edge = cochlear_map(100, 20000) 
        hi_edge = cochlear_map(FS / 2, 20000) - 1 

        # Define the number of critical frequency bands
        crit_freqs = 9

        #Define frequencies between high and low for later transformation back into Hz instead cochlear position
        step_size = (hi_edge - lo_edge) / (crit_freqs )
        frequency_points = np.arange(lo_edge, hi_edge + step_size, step_size)
        cochlearMapfs = inv_cochlear_map(frequency_points, 20000)

        #Rounding and multiplying to the nearest 10 for more accurate representation
        cochlearMapfs = np.round(cochlearMapfs / 10) * 10

        #Design L for each sample in the file a multi dimensional matrix for its frequency content
        L = len(YMonores)
        S = np.zeros((9, L))

        #Filter each frequency band derived by cochlearMapfs using the butter filter function
        for ind in range(1, len(cochlearMapfs)):
            order = 3
            fcutlow = cochlearMapfs[ind - 1]
            fcuthigh = cochlearMapfs[ind]
            b, a = butter(order, [fcutlow / (FS / 2), fcuthigh / (FS / 2)], 'bandpass')
            #x shows the decomposed signal dinto its narrowband components
            x = lfilter(b, a, YMonores)
            #put this signal for the first frequency band into the first row of S and so forth
            S[ind - 1, :] = x

        # Compute the amplitude envelope of each band        
        Nenv_temp = np.abs(hilbert(S.T, axis = 0))
        fs = 250  # Target sampling rate
        del S
        # Calculate the resampling factors using a "rational approximation" equal to RAT in MATLAB
        ratio = Fraction(fs, FS).limit_denominator()
        P, Q = ratio.numerator, ratio.denominator

        # Resample the signal
        Nenv = resample_poly(Nenv_temp, P, Q)
        #print(f"Downsampled Nenvshape: {Nenv.shape}, [:5] {Nenv[:5]}")

        # Averaging
        Wenv = np.mean(Nenv, axis=1)
        del Nenv_temp
        del Nenv

        # Normalization of Wenv using max value
        All_sound_cochl = Wenv / np.max(Wenv)
        del Wenv

        # choose first time window of interest
        All_sound_cochl_enter_time = All_sound_cochl[(int(enter_time[0])-1):(int(enter_time[1]))] # directly use start and end as integer indices
        del All_sound_cochl

        #Match parameters to MATLAB
        window_length = 500  # Window length
        overlap = window_length // 2  # Overlap length
        nfft = fs/0.25 #calculate nfft to it matches f in MATLAB [0:0.25:125]
    
        # Create Hanning window
        window = np.hanning(window_length) 

        # Compute raw STFT-values
        All_sound_cochl_freq, t1, All_sound_cochl_stft = scipy.signal.stft(
        All_sound_cochl_enter_time[500:],
        fs=fs,
        window=window,
        nperseg=window_length,
        noverlap=overlap,
        nfft=nfft,
        boundary=None,      # Deaktiviert Padding!
        padded=False,       # Kein zusätzliches Padding
        return_onesided=True,
        scaling='psd'       # Oder 'spectrum', je nach Vergleich
        )

        #De-Normalization procedure
        window_energy = np.sum(window**2)  # Sum of window values (normalization factor)
        All_sound_cochl_stft = All_sound_cochl_stft * np.sqrt(fs*window_energy)  
        filename_alls = f"All_sound_cochl_stft_Py_{iTest}_{index}.mat"

        # Save the data into .mat files using scipy.io.savemat
        scipy.io.savemat(filename_alls, {'All_sound_cochl_stft': All_sound_cochl_stft})
        del All_sound_cochl_enter_time
        

        #Setting for EEG filtering
        order = 3
        nyquist = fs / 2
        lowcut = 0.5
        highcut = 30
        EEG_fft = None
        n_channels = len(raw.info['ch_names'])
        n_freqs = All_sound_cochl_freq.shape[0] # as they will be synchronized via computing, we can take the placeholder from the audio
        n_times = len(t1) # same as above 
        EEG_fft = np.zeros((n_channels, n_freqs, n_times), dtype=np.complex128)

        #https://stackoverflow.com/questions/50839521/scipy-detrend-not-equivalent-to-matlab
        # Loop over EEG channels to preprocess and filter
        for i in range(n_channels):
            channel_data = data[0, i, :] 
            if np.allclose(channel_data, channel_data[0]):
                # signal is constant, just remove the mean
                channel_data = detrend_poly(channel_data,1)
            else:
                channel_data = detrend(channel_data)
                channel_data_function = detrend_poly(channel_data,1)
            channel_data_enter_time = channel_data[(int(enter_time[0])-1):(int(enter_time[1]))] # use start and stop indices
            signal_length = len(channel_data_enter_time)

            # Amplitude envelope of the EEG signal of that channel
            EEGUPPER = np.abs(hilbert(channel_data_enter_time))

            # Bandpass filter the EEG envelope
            b, a = butter(order, [lowcut / nyquist, highcut / nyquist], btype='band')

            #Filtering, also apply padding as in the audio processing
            EEG_env=filtfilt(b, a, EEGUPPER, axis=0, padtype = 'odd', padlen=3*(max(len(b),len(a))-1))                

            # Scale/Normalize the EEG envelope
            EEG_env_norm = EEG_env / np.max(EEG_env)

            #same parameters as before
            All_freqs, t1, eeg1 = scipy.signal.stft(
            EEG_env_norm[500:],
            fs=fs,
            window=window,
            nperseg=window_length,
            noverlap=overlap,
            nfft=nfft,
            boundary=None,      # Deaktiviert Padding!
            padded=False,       # Kein zusätzliches Padding
            return_onesided=True,
            scaling='psd'       # Oder 'spectrum', je nach Vergleich
            )

            window_energy = np.sum(window**2)  # Sum of window values (normalization factor)
            eeg1 = eeg1 * np.sqrt(fs*window_energy)  
            EEG_fft[i,:,:] = eeg1
        filename_eegfft = f"EEG_fft_Py_{iTest}_{index}.mat"

        # Save the data into .mat files using scipy.io.savemat
        scipy.io.savemat(filename_eegfft, {'EEG_fft': EEG_fft})
        del eeg1, t1, EEG_env, EEGUPPER, EEG_env_norm, b, a, channel_data, channel_data_enter_time

        # Assign the values
        X1 = EEG_fft
        del EEG_fft
        #Transpose so it matches computing of coherence
        Y1 = All_sound_cochl_stft.conj().T
        del All_sound_cochl_stft

        #define Cohxy as 4D matrix for Participants, time points, channels and frequency bands
        n_timepoints = len(time_points)
        n_times = X1.shape[2]
        n_channels = X1.shape[0]
        n_freqs = X1.shape[1]
        n_tests = len(Test)
        if Cohxy_ini == False:
            Cohxy = np.zeros(((n_tests), ((n_timepoints)), n_channels, n_freqs), dtype=np.float64)
            Cohxy_ini = True
        gc.collect()

        #for checking
        chan_freq_mean = np.zeros((n_channels, n_freqs), dtype=np.complex128)  # Channel frequency array
        chan_freq_mean_abs= np.zeros((n_channels, n_freqs), dtype=np.complex128)  # Channel frequency array
        x1across_mean_abs= np.zeros((n_channels, n_freqs), dtype=np.complex128)  # X1 across segments
        y1across_mean_abs = np.zeros((n_channels, n_freqs), dtype=np.complex128)  # Y1 across segments
        x1across_mean = np.zeros((n_channels, n_freqs), dtype=np.complex128)  # X1 across segments
        y1across_mean = np.zeros((n_channels, n_freqs), dtype=np.complex128)  # Y1 across segments
        #all
        chan_freq_all = np.zeros((n_channels, n_freqs, n_times), dtype=np.complex128)  # Channel frequency array
        x1across_all = np.zeros((n_channels, n_freqs, n_times), dtype=np.complex128)  # X1 across segments
        y1across_all = np.zeros((n_channels, n_freqs, n_times), dtype=np.complex128)  # Y1 across segments
        x1across_all_abs = np.zeros((n_channels, n_freqs, n_times), dtype=np.complex128)  # X1 across segments
        y1across_all_abs = np.zeros((n_channels, n_freqs, n_times), dtype=np.complex128)  # Y1 across segments
        print("Test {}, Time {}".format(iTest, index))
        print("Cohxy shape: {}".format(Cohxy.shape))

        for iChan in range(n_channels):
            newx1 = (X1[iChan, :, :].conj()).T
            for iFreq in range(n_freqs):
                chan_freq = np.zeros((n_times,), dtype = np.complex128)
                x1across = np.zeros((n_times,), dtype = np.complex128)
                y1across = np.zeros((n_times,), dtype = np.complex128)

                for iSegment in range(n_times):
                    if iSegment < newx1.shape[0] and iFreq < newx1.shape[1] and iSegment < Y1.shape[0] and iFreq < Y1.shape[1]:
                        chan_freq[iSegment] = np.abs(newx1[iSegment, iFreq]) * np.abs(Y1[iSegment, iFreq]) * \
                                                        np.exp(1j * (np.angle(newx1[iSegment, iFreq]) - np.angle(Y1[iSegment, iFreq])))
                        x1across[iSegment] = newx1[iSegment, iFreq]
                        y1across[iSegment] = Y1[iSegment, iFreq]    
                        chan_freq_all[iChan, iFreq, iSegment] = chan_freq[iSegment]  # Store for all time segments
                        y1across_all[iChan, iFreq, iSegment] = y1across[iSegment]# Store for all time segments
                        x1across_all[iChan, iFreq, iSegment] = x1across[iSegment] # Store for all time segments
                        y1across_all_abs[iChan, iFreq, iSegment] = np.abs(y1across[iSegment])# Store for all time segments
                        x1across_all_abs[iChan, iFreq, iSegment] = np.abs(x1across[iSegment]) # Store for all time segments
                chan_freq_mean_abs[iChan,iFreq] = np.abs(np.mean(chan_freq,dtype=np.complex128))
                chan_freq_mean[iChan,iFreq] = np.mean(chan_freq,dtype=np.complex128)
                x1across_mean_abs[iChan,iFreq] = np.mean(abs(x1across),dtype=np.complex128)
                y1across_mean_abs[iChan,iFreq] = np.mean(abs(y1across),dtype=np.complex128)
                x1across_mean[iChan,iFreq] = np.mean(x1across)
                y1across_mean[iChan,iFreq] = np.mean(y1across)
                Cohxy[index_test, index, iChan, iFreq] = (np.abs(np.mean(chan_freq, dtype=np.complex128)) ** 2) / \
                                                        (np.mean(np.abs(x1across) ** 2, dtype=np.complex128) * np.mean(np.abs(y1across) ** 2, dtype=np.complex128))
                
        print(f"x1across shape {x1across.shape}, y1across shape {y1across.shape}, chan_freq shape {chan_freq.shape}")
        mean_chan_abs= f'chan_freq_mean_abs_{iTest}_py_{index}.mat'
        mean_chan= f'chan_freq_mean_{iTest}_py_{index}.mat'
        all_x1abs = f'x1across_all_abs_{iTest}_py_{index}.mat'
        all_y1abs = f'y1across_all_abs_{iTest}_py_{index}.mat'
        all_x1 = f'x1across_all_{iTest}_py_{index}.mat'
        all_y1 = f'y1across_all_{iTest}_py_{index}.mat'
        mean_x1 = f'x1across_mean_{iTest}_py_{index}.mat'
        mean_y1 = f'y1across_mean_{iTest}_py_{index}.mat'
        mean_x1abs = f'x1across_mean_abs_{iTest}_py_{index}.mat'
        mean_y1abs = f'y1across_mean_abs_{iTest}_py_{index}.mat'
        all_chan = f'chan_freq_all_{iTest}_py_{index}.mat'

        # Save mean
        scipy.io.savemat(mean_chan, {'chan_freq_mean': chan_freq_mean})
        scipy.io.savemat(mean_x1, {'x1across_mean': x1across_mean})
        scipy.io.savemat(mean_y1, {'y1across_mean': y1across_mean})

        #save mean abs
        scipy.io.savemat(mean_chan_abs, {'chan_freq_mean_abs': chan_freq_mean_abs})
        scipy.io.savemat(mean_x1abs, {'x1across_mean_abs': x1across_mean_abs})
        scipy.io.savemat(mean_y1abs, {'y1across_mean_abs': y1across_mean_abs})

        #save all abs
        scipy.io.savemat(all_x1abs, {'x1across_all_abs': x1across_all_abs})
        scipy.io.savemat(all_y1abs, {'y1across_all_abs': y1across_all_abs})

        #save all
        scipy.io.savemat(all_chan, {'chan_freq_all': chan_freq_all})
        scipy.io.savemat(all_y1, {'y1across_all': y1across_all})
        scipy.io.savemat(all_x1, {'x1across_all': x1across_all})
        del chan_freq, y1across, x1across, X1, Y1, newx1
        gc.collect()

filename_chan = f'Cohxy_py.mat'
scipy.io.savemat(filename_chan, {'Cohxy': Cohxy})
print(Cohxy)

print("Head")
#print(Cohxy[:10])
#print("Cohxy shape: {}".format(Cohxy.shape))


print("Cohxy valuees for T1 for TR4, TR5, TR6")
print(np.mean(Cohxy[0, 0, :, :], axis=0))
#print(np.mean(Cohxy[1, 0, :, :], axis=0))
#print(np.mean(Cohxy[2, 0, :, :], axis=0))
print("Cohxy valuees for T2 for TR4, TR5, TR6")
print(np.mean(Cohxy[0, 1, :, :], axis=0))
#print(np.mean(Cohxy[1, 1, :, :], axis=0))
#print(np.mean(Cohxy[2, 1, :, :], axis=0))
print("Cohxy valuees for T3 for TR4, TR5, TR6")
print(np.mean(Cohxy[0, 2, :, :], axis=0))
#print(np.mean(Cohxy[1, 2, :, :], axis=0))
#print(np.mean(Cohxy[2, 2, :, :], axis=0))
print("Cohxy valuees for T4 for TR4, TR5, TR6")


y_limT1 = max(np.nanmax(Cohxy[:, 0, :, :]) * 1.1, 0.05)
print(y_limT1)
y_limT2 = max(np.nanmax(Cohxy[:, 1, :, :]) * 1.1, 0.05)
y_limT3 = max(np.nanmax(Cohxy[:, 2, :, :]) * 1.1, 0.05)

plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(All_sound_cochl_freq, np.mean(Cohxy[0, 0, :, :], axis=0), linewidth=1.5, label=Test[0])
plt.plot(All_sound_cochl_freq, np.mean(Cohxy[1, 0, :, :], axis=0), linewidth=1.5, label=Test[1])
plt.plot(All_sound_cochl_freq, np.mean(Cohxy[2, 0, :, :], axis=0), linewidth=1.5, label=Test[2])
plt.legend()
plt.xlim([0, 9])
plt.ylim([0, 0.10])
plt.xlabel('Frequency')
plt.ylabel('Coherence Value')
plt.title('Time window 1')

plt.subplot(3, 1, 2)
plt.plot(All_sound_cochl_freq, np.mean(Cohxy[0, 1, :, :], axis=0), linewidth=1.5, label=Test[0])
plt.plot(All_sound_cochl_freq, np.mean(Cohxy[1, 1, :, :], axis=0), linewidth=1.5, label=Test[1])
plt.plot(All_sound_cochl_freq, np.mean(Cohxy[2, 1, :, :], axis=0), linewidth=1.5, label=Test[2])
plt.legend()
plt.xlim([0, 9])
plt.ylim([0, 0.10])
plt.xlabel('Frequency')
plt.ylabel('Coherence Value')
plt.title('Time window 2')

plt.subplot(3, 1, 3)
plt.plot(All_sound_cochl_freq, np.mean(Cohxy[0, 2, :, :], axis=0), linewidth=1.5, label=Test[0])
plt.plot(All_sound_cochl_freq, np.mean(Cohxy[1, 2, :, :], axis=0), linewidth=1.5, label=Test[1])
plt.plot(All_sound_cochl_freq, np.mean(Cohxy[2, 2, :, :], axis=0), linewidth=1.5, label=Test[2])
plt.legend()
plt.xlim([0, 9])
plt.ylim([0, 0.05])
plt.xlabel('Frequency')
plt.ylabel('Coherence Value')
plt.title('Time window 3')

plt.tight_layout()
plt.show()