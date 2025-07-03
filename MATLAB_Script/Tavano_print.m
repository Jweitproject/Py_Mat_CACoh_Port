%% cerebroacoustic coherence (CAC)
% (c) alessandro tavano Uni Goethe May 2024

%% import EEG into a fieldtrip format
addpath('C:/Users/jessi/Desktop/FOrschung/Data_Study/fieldtrip-20240603');
format long
TestText = {'TR5'};
for iTest = 1:3
fprintf("Person ID  %s\n", TestText{iTest});
cd('C:/Users/jessi/Desktop/FOrschung/Data_Study');
cfg                 = [];
cfg.dataset         = sprintf('A080721_001_%s_raw.fif', TestText{iTest});
cfg.trialdef.length = Inf;
cfg                 = ft_definetrial(cfg);

cfg.continuous = 'yes';
data           = ft_preprocessing(cfg);

% timepoints specific to this triplet
time_points =[1097.776 1140.6;
    1184.0 1221.844;
    1246.504 1295.0];

% audio envelope
for iTime = 1:3
enter_time = time_points(iTime,:)*250/1;
clear Y_mono YMonores
addpath('C:/Users/jessi/Desktop/FOrschung/Data_Study/ChimeraSoftware/');
[Y, FS] = audioread(sprintf('audio_A080721_001_%s.wav', TestText{iTest}));
%Y_mono = resample(Y, data.fsample,FS);
fprintf("Ich bin Y %s\n", mat2str(Y(1:5)));

Y_mono = Y;
Y_mono = sum(Y_mono, 2) / size(Y_mono, 2);
fprintf("YMono %s\n", mat2str(Y_mono(1:5)));
%Y_mono und 
%fprintf("Y_Mono Shape %s\n", mat2str(size(Y_mono)));
%fprintf("Y_mono %\n", mat2str(Y_mono));
%fprintf('enter time%s\n', mat2str(size(enter_time)));
%disp(enter_time);
%fprintf('Sample Rate: %d Hz\n', FS);
%fprintf('Audio Data Shape: %s\n', mat2str(size(Y)));
if FS/2 > 20000
    [P,Q] = rat(20000/fs);
    %fprintf('P: %d\n', mat2str(P));
    %fprintf('Q: %d\n', mat2str(Q));
    YMonores = resample(Y_mono,P,Q);
    %fprintf('in der if! YMonores %s\n', mat2str(YMonores));
    fs = 20000;
else
    YMonores = Y_mono;
    %fprintf('nicht in der if! YMonores %s\n', mat2str(YMonores));

end
%fprintf('Y: %s\n', mat2str(size(Y)));
%fprintf('Y Mono: %s\n', mat2str(size(Y_mono)));

% Bandpass filter the raw speech waveform into 9 logarithmically spaced bands
lo_edge = cochlear_map(100, 20000);
hi_edge = cochlear_map(FS/2, 20000) - 1;
%fprintf('lo_edge: %d Hz\n', lo_edge);
%fprintf('hi_edge: %d Hz\n', hi_edge);

% Define frequency bins
crit_freqs = 9;
cochlearMapfs = inv_cochlear_map(lo_edge:(hi_edge-lo_edge)/crit_freqs:hi_edge,20000);
cochlearMapfs = [round(cochlearMapfs/10)*10];
%fprintf('cochlearMapfs: %s\n', mat2str(size(cochlfsearMapfs)));
%disp('cochlearMapfs outcome');
%disp(mat2str(cochlearMapfs(1:5), 16));

%fprintf("YMono %s\n", mat2str(Y_mono));
% Design and apply a bandpass filter
L = size(YMonores, 1);
S = zeros(crit_freqs-1, L);
%fprintf("Monores %s\n",mat2str(YMonores(1:5),16));


for ind=2:length(cochlearMapfs)
    order    = 3;
    fcutlow  = cochlearMapfs(ind-1);
    fcuthigh = cochlearMapfs(ind);
    [b,a]    = butter(order,[fcutlow,fcuthigh]/(FS/2), 'bandpass');
    %fprintf('FS: %s\n', mat2str(FS));
    %fprintf('fcutlow, high divided: %s\n', mat2str([fcutlow,fcuthigh]/(FS/2)));


    %fprintf('nr: %s\n', mat2str(ind));
    %fprintf('b: %s\n', mat2str(b,16));
    %fprintf('a:  %s\n', mat2str(a,16));
    x        = filter(b,a,YMonores);
    %fprintf("fcutlow %s \n",mat2str(fcutlow,16));
    %fprintf("fcuthigh %s\n",mat2str(fcuthigh,16));
    %fprintf('x: in loop %s\n', mat2str(x(1:5),16));
    S(ind-1,:) = x;
    %fprintf('S: %s \n for ind: %s\n',mat2str(S(1:5)),mat2str(ind));
end
% Compute the amplitude envelope of each band
Nenv_temp = abs(hilbert(S'));
%fprintf('Nenv_temp: %s\n', mat2str(Nenv_temp(1,1:5),16));

[P, Q] = rat(250 / FS);
%fprintf(" P: %s\n", mat2str(P),16);
%fprintf(" Q: %s\n", mat2str(Q),16);
Nenv = resample(Nenv_temp, P, Q);
%fprintf('Nenv resample first 10 %s\n', mat2str(Nenv(1,1:5),16));

Wenv = mean(Nenv, 2);
clear Nenv
fprintf('Wenv: %s\n', mat2str(Wenv(1:5),16));

% Bandpass the auditory envelope
fs = 250;
[b,a] = butter(order, [0.5/(fs/2) 30/(fs/2)]);
%fprintf('a shape: %s\n', mat2str(size(a)));
%fprintf('b shape: %s\n', mat2str(size(b)));
%fprintf('Wenv shape: %s\n', mat2str(size(Wenv)));

%disp('b first 7');
%disp(mat2str(b(1:7), 16));
%disp('a first 7');
%disp(mat2str(a(1:7), 16));
filtWenv = filtfilt(b,a,Wenv);
%disp('filtwenv first 10 before assigning');
%disp(mat2str(filtWenv(1:5), 16));
%fprintf('filtWenv shape: %s\n', mat2str(size(filtWenv)));
if length(data) < length(filtWenv)
    filtWenv = filtWenv(1:length(data));
end


filtWenv = Wenv;
%disp("filtWenv first 10 after assigning")
%disp(mat2str(filtWenv(1:5), 16));

% Standardize the envelope
Wenv_norm = filtWenv./max(filtWenv);%Wenv_smooth./max(Wenv_smooth);
%fprintf('Wenv_norm shape: %s\n', mat2str(size(Wenv_norm)));
%fprintf('Wenv_norm value: %s\n', mat2str(Wenv_norm(1:5,1),16));
All_sound_cochl = Wenv_norm'; % save wideband envelope in range [0 1]
%fprintf('Wenv_norm transposed shape: %s\n', mat2str(size(All_sound_cochl)));
%fprintf('Wenv_norm transposed value: %s\n', mat2str(All_sound_cochl),16);

All_sound_cochl_enter_time = All_sound_cochl(enter_time(1):enter_time(2));
%fprintf('Wenv_norm transposed: %s\n', mat2str(All_sound_cochl(1:5)));
%fprintf('Wenv_norm not ransposed: %s\n', mat2str(Wenv_norm(1:5)));

%fprintf('All_sound_cochl_enter_time 501 %s\n', mat2str(All_sound_cochl_enter_time(500:505),16));
%fprintf('All_sound_cochl_enter_time shape: %s\n', mat2str(size(All_sound_cochl_enter_time)));
%first and last
%fprintf('First 10 elements: %s\n', mat2str(All_sound_cochl_enter_time(1:10)));
%fprintf('Last 10 elements: %s\n', mat2str(All_sound_cochl_enter_time(end-9:end)));

%stft sound
window = hann(500);
%fprintf('Window hanning.. %s\n', window(1:10),16);
win = hann(500);
% [x1, f1] = stft(Wenv_norm(501:end)',fs,'Window',window,'OverlapLength',200,'FFTLength',1000);
%scaling_factor = 2/(fs*sum(win.^2)); % PSD scaling factor

fprintf("fs bei all sound %s\n", mat2str(fs));
[x1, f1, t1] = spectrogram(All_sound_cochl_enter_time(501:end)',window,[], [0:0.25:125], fs,'psd');
fprintf("Spectrogram for audio");
fprintf("t1 %s\n",mat2str(size(t1)));
fprintf("f1 %s\n",mat2str(size(f1)));
fprintf("x1 %s\n",mat2str(size(x1),16));

All_sound_cochl_stft = x1;
%save(['All_sound_cochl_stft_' TestText{iTest} '_' iTime '.mat'], 'All_sound_cochl_stft');

% Assuming All_sound_cochl_stft is your STFT matrix

% Print the first 5 elements of the first row
%disp('Scaled PSD x1(:, 1:5):');
%disp(All_sound_cochl_stft(1, 1:5));

% Print the first 5 elements of the 6th row (index 6 in MATLAB)
%disp('Scaled PSD x1(6, 1:5):');
%disp(All_sound_cochl_stft(6, 1:5));

% Print the first 20 elements of the 21st row, at every 23rd column
%disp('Scaled PSD x1(21, 1:23:20):');
%disp(All_sound_cochl_stft(21, 21:24));

%fprintf('t... %s\n', t1(1:5));

All_sound_cochl_freq = f1;
fprintf('All_sound_cochl_freq: %s\n', mat2str(All_sound_cochl_freq(1:5),16));

clear Wenv*

% EEG envelope
clear EEG_fft

fprintf("enter time %s und %s\n",mat2str(enter_time(1)), mat2str(enter_time(2)));

for i = 1:size(data.trial{1,1},1)
    channel_data = data.trial{1,1}(i,:);
    if i == 14
        figure;
        plot(channel_data, 'LineWidth', 2);
        title(sprintf("Channel 14 - Zeitpunkt %d", iTime));
        xlabel("Zeit (Samples)");  % Hier wurde das fehlende schließende Klammerzeichen hinzugefügt
        ylabel("Amplitude");
        grid on;
    end

    %fprintf("channel nr %s\n",mat2str(i))
    %fprintf("channel data %s\n",mat2str(channel_data(1:5),16));
    %fprintf("channel i size %s\n", size(channel_data(i)));
    channel_data = detrend(channel_data);
    %fprintf("channel data detrended %s\n",mat2str(channel_data(1:5),16));
    %fprintf('channel_data mean: %f, std: %f, max: %f\n', mean(channel_data), std(channel_data), max(channel_data));

    channel_data_enter_time = channel_data(enter_time(1):enter_time(2));

    if i == 14
        figure;
        plot(channel_data, 'LineWidth', 2);
        title(sprintf("Channel 14 detrended - Zeitpunkt %d", iTime));
        xlabel("Zeit (Samples)");  % Hier wurde das fehlende schließende Klammerzeichen hinzugefügt
        ylabel("Amplitude");
        grid on;
    end

    %fprintf("channel data enter time%s\n",mat2str(channel_data_enter_time(1:5),16));
    EEGUPPER = abs(hilbert(channel_data_enter_time));
    %fprintf("EGGUPPER %s\n", mat2str(EEGUPPER(1:5),16));
    %fprintf('EEGUPPER mean: %f, std: %f, max: %f\n', mean(EEGUPPER), std(EEGUPPER), max(EEGUPPER));


    %[EEGUPPER,EEGLOWER] = envelope(channel_data);
    [b,a] = butter(order, [0.5/(fs/2) 30/(fs/2)]); % bandpass filter
    fprintf("a %s\n", mat2str(a(1:7),16));
    %fprintf("b %s\n", mat2str(b(1:7),16));
    %fprintf("a shape %s\n", mat2str(size(a)));
    %fprintf("b shape %s\n", mat2str(size(b)));
    %fprintf("EEG upper shape %s\n", mat2str(size(EEGUPPER)));
        
    
    EEG_env = filtfilt(b,a,double(EEGUPPER));
    %fprintf("Value eeg_env shape %s\n", mat2str(size(EEG_env),16));

    disp(mat2str(EEG_env(1:5), 16));
    %EEG_env = EEGUPPER;
    % w = gausswin(10)./sum(gausswin(10));
    % EEG_env_smooth = filter(w,1,EEG_env); % operate along matrix columns
    % standardize
    EEG_env_norm = EEG_env./max(EEG_env);%EEG_env_smooth./max(EEG_env_smooth);
    %fprintf("Value eeg_env_norm %s\n", mat2str(EEG_env_norm(1:5),16));
    %fprintf("Value eeg_env norm shape %s\n", mat2str(size(EEG_env_norm),16));
    
    %%

    %stft eeg
    window = hann(500);
    %fprintf(" ich bin fs %s\n",mat2str(fs))
    % [eeg1, f1] = stft(EEG_env_norm(501:end),eeg_fs,'Window',window,'OverlapLength',200,'FFTLength',1000);
    [eeg1, f1, t1] = spectrogram(EEG_env_norm(501:end)',window,[], [0:0.25:125], fs,'psd');
    %fprintf("eeg1 size %s\b",mat2str(size(eeg1)));   
    %fprintf("f1 size %s\b",mat2str(size(f1)));   
    %fprintf("t1 size %s\b",mat2str(size(t1)));   
    % Print the first 5 elements of the first row
    %disp('Scaled PSD eeg1(:, 1:5):');
    %disp(eeg1(1, 1:5));
    
    % Print the first 5 elements of the 6th row (index 6 in MATLAB)
    %disp('Scaled PSD eeg1(6, 1:5):');
    %disp(eeg1(6, 1:5));
    
    % Print the first 20 elements of the 21st row, at every 23rd column
    %disp('Scaled PSD eeg1(21, 21:26):');
    %disp(eeg1(21, 21:24));
    EEG_fft(i,:,:) = eeg1;

end
%fprintf("EEG_fft %s\n", mat2str(EEG_fft(1:5),16));
%fprintf('Number of channels: %d\n', size(data.trial{1,1}, 1));
%fprintf('EEG_fft dimensions:  %s\n', mat2str(size(EEG_fft)));
%save(['EEG_fft' TestText{iTest} '_' time_points(iTime) '.mat'], 'EEG_fft');

All_EEG = EEG_fft;
%fprintf('All_EEG dimensions:  %s\n', mat2str(size(All_EEG)));

All_freqs = f1;
fprintf('All_freqs:  %s\n', mat2str(size(All_freqs)));
% Initialize arrays for storing values across all channels, frequencies, and segments
chan_freq_all = zeros(size(X1,1), size(X1,2), size(X1,3)); % [channels, frequencies, segments]
x1across_all = zeros(size(X1)); % [channels, frequencies, segments]
y1across_all = zeros(size(Y1)); % [channels, frequencies, segments]

% Magnitude-Squared Coherence (Coh)/InterSite Phase Clustering (ISPC)
clear X1 Y1

X1 = EEG_fft(:,:,:);
fprintf("All sound cochl stft %f\n", All_sound_cochl_stft(1,1:5));

Y1 = All_sound_cochl_stft';
fprintf("Y %f\n", Y1(1,1:5));

for iChan = 1:size(X1,1)
    newx1 =  squeeze(X1(iChan,:,:))';
    %av_newx1 = mean(newx1);
    for iFreq = 1:size(newx1,2)
        for iSegment = 1:size(newx1,1)
            %fprintf('av_newx1 %f\n', av_newx1);
            chan_freq(iSegment) = abs(newx1(iSegment,iFreq)).*abs(Y1(iSegment,iFreq)).*exp(1i*(angle(newx1(iSegment,iFreq))-angle(Y1(iSegment,iFreq))));
            x1across(iSegment) = newx1(iSegment,iFreq);
            y1across(iSegment) = Y1(iSegment,iFreq);
            chan_freq_all(iChan, iFreq, iSegment) = abs(newx1(iSegment, iFreq)) * abs(Y1(iSegment, iFreq)) * exp(1i * (angle(newx1(iSegment, iFreq)) - angle(Y1(iSegment, iFreq))));
            x1across_all(iChan, iFreq, iSegment) = newx1(iSegment, iFreq);
            y1across_all(iChan, iFreq, iSegment) = Y1(iSegment, iFreq);
            
        end
        %mean_freq = mean(chan_freq);
        %fprintf('Mean frequency: %f\n', mean_freq);
        %mean_x1 = mean(x1across);
        %mean_y1 = mean(y1across);
        %fprintf('x1 across mean %f\n', mean_x1);
        %fprintf('chan freq mean %f\n', mean_freq);
        %fprintf('y1 across mean%f\n', mean_y1);

        %fprintf("iFreq %s\n",mat2str(iFreq));
        %fprintf("channel frequency %s\n",mat2str(mean(chan_freq)));
        %fprintf("x1across %s\n", mat2str(mean(x1across)));
        %fprintf("y1across %s\n",mat2str(mean(y1across)));
        Cohxy(iTest,iTime,iChan,iFreq) = (abs(mean(chan_freq)).^2)/((mean(abs(x1across).^2))*(mean(abs(y1across).^2)));
        fprintf('Cohxy[%d][%d][%d][%d]: %f\n', iTest, iTime, iChan, iFreq, Cohxy(iTest, iTime, iChan, iFreq));
    end

end
% Save the data to .mat files
filename_chan = sprintf('chan_freq_all_%s_%d.mat', TestText{iTest}, iTime);
filename_x1   = sprintf('x1across_all_%s_%d.mat', TestText{iTest}, iTime);
filename_y1   = sprintf('y1across_all_%s_%d.mat', TestText{iTest}, iTime);

% Save the 3D arrays to .mat files
save(filename_chan, 'chan_freq_all');
save(filename_x1, 'x1across_all');
save(filename_y1, 'y1across_all');

fprintf("Saved chan_freq_all, x1across_all, and y1across_all to .mat files.\n");

fprintf("X1 %s\n",mat2str(size(X1)));
fprintf("Y1 %s\n", mat2str(size(Y1)));



end
end
%save('Cohxy_data.mat', 'Cohxy');

%% plotting

meanCohxy_1_1 = squeeze(mean(Cohxy(1,1,:,:), 3));
meanCohxy_2_1 = squeeze(mean(Cohxy(2,1,:,:), 3));
meanCohxy_3_1 = squeeze(mean(Cohxy(3,1,:,:), 3));

meanCohxy_1_2 = squeeze(mean(Cohxy(1,2,:,:), 3));
meanCohxy_2_2 = squeeze(mean(Cohxy(2,2,:,:), 3));
meanCohxy_3_2 = squeeze(mean(Cohxy(3,2,:,:), 3));

meanCohxy_1_3 = squeeze(mean(Cohxy(1,3,:,:), 3));
meanCohxy_2_3 = squeeze(mean(Cohxy(2,3,:,:), 3));
meanCohxy_3_3 = squeeze(mean(Cohxy(3,3,:,:), 3));

% Print values for debugging
fprintf('Mean Cohxy values for subplot 1:\n');
fprintf('Cohxy(1,1,:) = %s\n', mat2str(meanCohxy_1_1));
fprintf('Cohxy(2,1,:) = %s\n', mat2str(meanCohxy_2_1));
fprintf('Cohxy(3,1,:) = %s\n', mat2str(meanCohxy_3_1));

fprintf('\nMean Cohxy values for subplot 2:\n');
fprintf('Cohxy(1,2,:) = %s\n', mat2str(meanCohxy_1_2));
fprintf('Cohxy(2,2,:) = %s\n', mat2str(meanCohxy_2_2));
fprintf('Cohxy(3,2,:) = %s\n', mat2str(meanCohxy_3_2));

fprintf('\nMean Cohxy values for subplot 3:\n');
fprintf('Cohxy(1,3,:) = %s\n', mat2str(meanCohxy_1_3));
fprintf('Cohxy(2,3,:) = %s\n', mat2str(meanCohxy_2_3));
fprintf('Cohxy(3,3,:) = %s\n', mat2str(meanCohxy_3_3));

%%figure
subplot(3,1,1)
plot(f1,squeeze(mean(Cohxy(1,1,:,:),3)), 'LineWidth',3)
hold on
plot(f1,squeeze(mean(Cohxy(2,1,:,:),3)), 'LineWidth',1.5)
hold on
plot(f1,squeeze(mean(Cohxy(3,1,:,:),3)), 'LineWidth',1.5)
legend(TestText)
xlim([0 9])
subplot(3,1,2)
plot(f1,squeeze(mean(Cohxy(1,2,:,:),3)), 'LineWidth',1.5)
hold on
plot(f1,squeeze(mean(Cohxy(2,2,:,:),3)), 'LineWidth',3)
hold on
plot(f1,squeeze(mean(Cohxy(3,2,:,:),3)), 'LineWidth',1.5)
legend(TestText)
xlim([0 9])
subplot(3,1,3)
plot(f1,squeeze(mean(Cohxy(1,3,:,:),3)), 'LineWidth',1.5)
hold on
plot(f1,squeeze(mean(Cohxy(2,3,:,:),3)), 'LineWidth',1.5)
hold on
plot(f1,squeeze(mean(Cohxy(3,3,:,:),3)), 'LineWidth',3)
legend(TestText)
xlim([0 9])
xlabel('Hz')
print('-vector','Figure1_Nathalie_Alessandro', '-dpdf')
