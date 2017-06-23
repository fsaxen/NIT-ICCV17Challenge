function sets = split_stratified( data, split_param)
% keep rate of label groups, but only subsampling_rate*length(idx) items

    % Check dataset
    libDataset.util_check_dataset(data, 'supervised', 'discrete_predictors', 'one_predictor_idx');

    % Create output datasets
    sets = cell(split_param.k, 1);

    % Get selected predictors
    y = data.y(data.sample_idx, data.predictor_idx);
    
    % number of distinct labels
    predictors = unique(y);
    num_predictors = length(predictors);
    
    % Create num_predictors random permutations
    sample_idx_p = cell(num_predictors, 1);
    num_samples = zeros(num_predictors, 1);
    for p = 1 : num_predictors
        % Calculate number of samples for each predictor
        sample_idx_p{p} = data.sample_idx(y == predictors(p));
        num_samples(p) = size(sample_idx_p{p},1);
    end

    % Count the number of samples to select for each set and predictor
    num_selected_samples = zeros(split_param.k, num_predictors);

    % Count how many samples have already been selected over all sets
    sample_count = zeros(num_predictors, 1);
    
    % The ratio for k often does not divide num_samples exactly.
    % In this cases at the end some samples would just be left out.
    % We divide out these left over samples at the end and prefer those
    % sets, that almost would have gotten another sample in the first place.
    ratio_left_out = zeros(split_param.k, num_predictors);
    
    % Calculate how many samples are selected for each set
    for k = 1 : split_param.k
        for p = 1 : num_predictors
            % How many samples do we sample for predictors(p)?
            num_selected_samples(k,p) = floor(num_samples(p) * split_param.ratio(k));
            ratio_left_out(k,p) = rem(num_samples(p) * split_param.ratio(k),1);
            
            % Prepare for the next set
            sample_count(p) = sample_count(p) + num_selected_samples(k,p);
        end
    end

    % Handle the left-over samples.
    if ~split_param.with_repetition
        % Prefer those classes that almost would have gotten another sample.
        for p = 1 : num_predictors
            [ratio, k_idx] = sort(ratio_left_out(:,p), 'descend');
            aftersampling_idx = 1;
            while(sample_count(p) < num_samples(p) && aftersampling_idx <= split_param.k && ratio(aftersampling_idx) > 0)
                % Choose set that suffered most
                k = k_idx(aftersampling_idx);
                
                % Add one sample to the selected samples
                num_selected_samples(k,p) = num_selected_samples(k,p) + 1;
                
                % This set now has gotton an aditional sample, even though
                % it shouldn't have gotton an entire sample. We need to
                % subtract the difference from the next predictors because
                % otherwise the subset might get preferred and the final
                % sets skewed (by a tiny fraction, but nevertheless).
                if p < num_predictors
                    not_seen_predictor_idx = p + 1 : num_predictors;
                    ratio_left_out(k, not_seen_predictor_idx) = ratio_left_out(k, not_seen_predictor_idx) - (1 - ratio_left_out(k,p)) / length(not_seen_predictor_idx);
                end

                % Prepare for next iteration
                sample_count(p) = sample_count(p) + 1;
                aftersampling_idx = aftersampling_idx + 1;
            end
        end
    end
    
    % Random permutate the sampling indices
    sample_idx = cell(num_predictors, 1);
    if split_param.with_repetition
        for p = 1 : num_predictors
            sample_idx{p} = libUtil.datasample(sample_idx_p{p}, sample_count(p),'Replace',true);
        end
    else % without repetition
        for p = 1 : num_predictors
            sample_idx{p} = libUtil.datasample(sample_idx_p{p}, sample_count(p),'Replace',false);
        end
    end
    
    
    % Finally select appropriate samples from data
    sample_count = ones(num_predictors, 1);
    for k = 1 : split_param.k
        % Copy dataset
        sets{k} = data;
        % Preallocate memory for sample_idx
        sets{k}.sample_idx = zeros(sum(num_selected_samples(k,:)),1);
        % index for current set
        set_counter = 1;
        for p = 1 : num_predictors
            % Select samples
            sets{k}.sample_idx(set_counter : set_counter + num_selected_samples(k,p) - 1) = sample_idx{p}(sample_count(p) : sample_count(p) + num_selected_samples(k,p) - 1);
            
            % Warn if this set did not get any samples of current predictor
            if num_selected_samples(k,p) == 0
                warning(strcat('Set k=',num2str(k),' does not contain any predictor=',num2str(predictors(p)),'. Please consider changing split_param.ratio(k)'));
            end
            
            % Prepare for next iteration
            set_counter = set_counter + num_selected_samples(k,p);
            sample_count(p) = sample_count(p) + num_selected_samples(k,p);
        end
    end
end

