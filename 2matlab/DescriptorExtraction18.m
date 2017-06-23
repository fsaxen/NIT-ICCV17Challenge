vis = 0;

fps = 100;       % Assumption based on estimated playback speed
cutoff = 1;     % Lowpass cutoff frequency in Hz
order = 1;

%n_sliding_avg = 5;
if train_val_or_test == 0
    fn_in_AUs = [data_folder, 'AUOld_train_nomax.txt'];
    fn_out = [data_folder, 'AUOld_train_descriptor18.mat'];
elseif train_val_or_test == 1
    fn_in_AUs = [data_folder, 'AUOld_val_nomax.txt'];
    fn_out = [data_folder, 'AUOld_val_descriptor18.mat'];
elseif train_val_or_test == 2
    fn_in_AUs = [data_folder, 'AUOld_test_nomax.txt'];
    fn_out = [data_folder, 'AUOld_test_descriptor18.mat'];
end

load(fn_in_AUs);

%% Extract descriptors

n_feat = size(samples(1).data,2);
n_seq = length(samples);
if cutoff > 0
%     fps = 25;
%     N = 3;                 % Order of polynomial fit
%     F = 13;                % Window length
%     [b,a] = sgolay(N,F);   % Calculate S-G coefficients
    [b, a] = butter(order, cutoff/(fps/2));
end

%sliding_avg_wts = ones(n_sliding_avg, 1) ./ n_sliding_avg;

for i = 1:n_seq
    label = samples(i).label;
    features = samples(i).data;
    %features = features(:,end-2:end);
    seq_descr = [];
    for j = 1:size(features,2)
        sig_raw = features(:,j);
        %sig_s = conv(sig_raw,sliding_avg_wts,'valid');
        if cutoff > 0
            sig_s = filtfilt(b, a, sig_raw);
        else
            sig_s = sig_raw;
        end
        sig_v = diff(sig_s);
        if cutoff > 0
            sig_v = filtfilt(b, a, sig_v);
        end
        sig_a = diff(sig_v);
        if cutoff > 0
            sig_a = filtfilt(b, a, sig_a);
        end
        shift = 0;
        %offset = 25;
        %sig_s = sig_s(offset:end-offset);
        %sig_v = sig_v(offset:end-offset);
        %sig_a = sig_a(offset:end-offset);
        if vis
            subplot(311);
            %plot(n_sliding_avg/2+(1:length(sig_s)),sig_s, 'b');
            %hold on; plot(1:length(sig_raw),sig_raw, 'r'); hold off;
            plot(shift + (1:length(sig_s)), sig_s, 'b');
            hold on; plot(sig_raw, 'r'); hold off;
            subplot(312);
            plot(shift + (1:length(sig_v)), sig_v);
            subplot(313);
            plot(shift + (1:length(sig_a)), sig_a);
            title(sprintf('feat %i (label %i)', j, label));
        end
        
        % Extract descriptor vairables (up to constants)
        sigs = { sig_s , sig_v , sig_a };
        descr = zeros(17,3);
        for k = 1:3
            s_mean = mean(sigs{k});
            s_min = min(sigs{k});
            s_max = max(sigs{k});
            s_thresh = 0.5 * (s_min + s_mean);
            % Value
            descr(1,k) = s_mean;
            descr(2,k) = median(sigs{k});
            descr(3,k) = s_min;
            descr(4,k) = s_max;
            % Variability
            descr(5,k) = range(sigs{k});
            descr(6,k) = std(sigs{k});
            descr(7,k) = iqr(sigs{k});
            descr(8,k) = diff(prctile(sigs{k}, [10; 90]));
            descr(9,k) = mad(sigs{k});
            % Time
            [~,descr(10,k)] = max((sigs{k}));
            descr(10,k) = descr(10,k) / fps;
            % Duration
            seg_mean = sigs{k} > s_mean;
            seg_thresh = sigs{k} > s_thresh;
            descr(11,k) = mean(seg_mean);
            descr(12,k) = mean(seg_thresh);
            % Count
            zc_mean = sum(zerocross(seg_mean - 0.5));
            zc_thresh = sum(zerocross(seg_thresh - 0.5));
            descr(13,k) = ceil(zc_mean/2);
            descr(14,k) = ceil(zc_thresh/2);
            % Area
            descr(15,k) = 0.001 * sum(sigs{k} - s_min);
            descr(16,k) = 0.01 * sum(sigs{k} - s_min) / (s_max - s_min);
            % tmax - tmeancross(first)
            mean_cross = find(zerocross(seg_mean - 0.5));
            descr(17,k) = descr(10,k) - mean_cross(1) / fps;
        end
        
        if any(isnan(descr(:)))
            warning('some nan descriptor variables');
        end
        
        
        descr_au = reshape(descr, 1, []);
        descr_au = horzcat(descr_au, descr_au(:,[1:9 15 16]).^2);
        seq_descr = [ seq_descr descr_au ];
    end
    
    samples(i).data = seq_descr;
        
end


save(fn_out, 'samples');
