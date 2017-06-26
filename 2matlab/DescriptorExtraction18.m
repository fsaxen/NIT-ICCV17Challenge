vis = 0;    % visualization?

% filenames
if train_val_or_test == 0
    fn_in_AUs = [data_folder, 'train_AUOld.txt'];
    fn_out = [data_folder, 'train_AUOld_descriptor18.mat'];
elseif train_val_or_test == 1
    fn_in_AUs = [data_folder, 'val_AUOld.txt'];
    fn_out = [data_folder, 'val_AUOld_descriptor18.mat'];
elseif train_val_or_test == 2
    fn_in_AUs = [data_folder, 'test_AUOld.txt'];
    fn_out = [data_folder, 'test_AUOld_descriptor18.mat'];
end


%% Extract descriptors

% load sample table with AU signals
load(fn_in_AUs);

% smoothing parameters
fps = 100;
cutoff = 1;
order = 1;

n_feat = size(samples(1).data,2);
n_seq = length(samples);
if cutoff > 0
    [b, a] = butter(order, cutoff/(fps/2));
end

% for each sample (video)
for i = 1:n_seq
    label = samples(i).label;
    au_signals = samples(i).data;
    seq_descr = [];
    
    % for each AU signal
    for j = 1:size(au_signals,2)
        
        % smooth signal, calulate speed / acceleration signal
        sig_raw = au_signals(:,j);
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
        
        % optional visualization
        if vis
            subplot(311);
            plot((1:length(sig_s)), sig_s, 'b');
            hold on; plot(sig_raw, 'r'); hold off;
            subplot(312);
            plot((1:length(sig_v)), sig_v);
            subplot(313);
            plot((1:length(sig_a)), sig_a);
            title(sprintf('feat %i (label %i)', j, label));
            drawnow;
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
        
        % add squared values of value / variability / area domains
        descr_au = horzcat(descr_au, descr_au(:,[1:9 15 16]).^2);
        
        seq_descr = [ seq_descr descr_au ];
    end
    
    samples(i).data = seq_descr;
        
end

%% save descriptor file
save(fn_out, 'samples');
