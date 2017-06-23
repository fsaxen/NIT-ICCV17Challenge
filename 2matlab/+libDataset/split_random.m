function sets = split_random( data, split_param )
% keep rate of label groups, but only subsampling_rate*length(idx) items

    % Check dataset
    libDataset.util_check_dataset(data);
    
    % Create output datasets
    sets = cell(split_param.k, 1);

    num_samples = size(data.sample_idx, 1);
    
    % Count how many samples have already been selected
    sample_count = 0;
    
    % Number of selected samples for each set
    num_selected_samples = zeros(split_param.k, 1);
    
    % The ratio for k often does not divide num_samples exactly.
    % In this cases at the end some samples would just be left out.
    % We divide out these left over samples at the end and prefer those
    % sets, that almost would have gotten another sample in the first place.
    ratio_left_out = zeros(split_param.k, 1);
    
    for k = 1 : split_param.k
        % Calculate the amount of samples we need for the current dataset.
        num_selected_samples(k) = floor(num_samples * split_param.ratio(k));
        ratio_left_out(k) = rem(num_samples * split_param.ratio(k), 1);
        
        % Prepare for the next set
        sample_count = sample_count + num_selected_samples(k);
    end

    % Handle the left-over samples for split_param.with_repitition = false.
    if ~split_param.with_repetition
        % Prefer those classes first that almost would have gotten another
        % sample.
        [~, k_idx] = sort(ratio_left_out, 'descend');
        aftersampling_idx = 1;
        while(sample_count < num_samples && aftersampling_idx <= split_param.k)
            % Choose set that suffered most
            k = k_idx(aftersampling_idx);

            % Update the number of selected samples
            num_selected_samples(k) = num_selected_samples(k) + 1;

            % Prepare for next iteration
            sample_count = sample_count + 1;
            aftersampling_idx = aftersampling_idx + 1;
        end
    end
    
    % Generate a scrambled ordering of items
    if split_param.with_repetition
        sample_idx = libUtil.datasample(data.sample_idx, sample_count,'Replace',true);
    else
        sample_idx = libUtil.datasample(data.sample_idx, sample_count,'Replace',false);
%         sample_idx = data.sample_idx(randperm(size(data.sample_idx, 1), sample_count));
    end
    
    
    % Finally copy dataset and select appropriate indices
    sample_count = 1;
    for k = 1 : split_param.k
        % copy dataset
        sets{k} = data;
        
        % Warn if no samples have been selected for this subset
        if num_selected_samples(k) == 0
            warning(strcat('Set k=',num2str(k),' is empty. Please consider changing split_param.ratio(k)'));
        end
       
        % Select new sample indices
        sets{k}.sample_idx = sample_idx(sample_count : sample_count + num_selected_samples(k) - 1);

        % Prepare for next iteration
        sample_count = sample_count + num_selected_samples(k);
    end
end

