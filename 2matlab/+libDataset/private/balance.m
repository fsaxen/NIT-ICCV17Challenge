function dataset_out = balance(dataset, threshold, class_hist_sorted, classes_sorted, h_downsample, h_upsample, db_copy_required)
%   Redistribute the dataset by balancing the number of samples for each
%   label.
%   dataset: contains the entire dataset.
%
%   threshold: number of expected samples for each class
%
%   class_hist_sorted: number of samples for each class (sorted in
%                      ascending order). e.g. [3 7 10 20].
%
%   classes_sorted: label value according to class_hist_sorted. e.g. [3 2 0 1]
%
%   h_downsample: Function handle to downsmapling method.
%
%   h_upsample: Function handle to upsampling method.
%

    % Check input
    if length(class_hist_sorted) ~= length(classes_sorted)
        error('class_hist_sorted and classes_sorted must have the same number of elements, because they correspond to eachother.');
    end

    % Get label vector
    label = dataset.y(dataset.sample_idx, dataset.predictor_idx);
    
    
    % Create for each class an own dataset
    num_sets = length(class_hist_sorted);
    set = cell(num_sets, 1);
    
    for i = 1 : num_sets
        
        % Mask all indexed samples that correspond to current class.
        % Caution: y_mask is relative to label and thus contains
        % only the indexed samples from label.
        class_mask = label == classes_sorted(i);
 
        % This copy of dataset will not increase our memory usage :)
        set{i} = dataset;
        set{i}.sample_idx = dataset.sample_idx(class_mask);
        
        % Reduce dataset such that unnecessary data is thrown away. This
        % will result in new memory allocation and thus is only necessary
        % if downsampling or upsampling methods do anything with the
        % features.
        if db_copy_required
            set{i} = libDataset.util_apply_sample_idx(set{i});
            set{i} = libDataset.util_apply_predictor_idx(set{i});
        end
        
        if class_hist_sorted(i) == threshold(i)
            % Take them all, number of class elements matches exactly the
            % expected number.
        elseif class_hist_sorted(i) > threshold(i)
            % downsample
            set{i} = h_downsample(set{i}, threshold(i));
        else %class_hist_sorted(i) < threshold <- implicitly true
            % upsample
            set{i} = h_upsample(set{i}, threshold(i));
        end
    end

    % Combine all sets
    if db_copy_required
        dataset_out = libDataset.util_add_dataset(set(:));
    else
        dataset_out = libDataset.util_add_dataset('NoCopyCheck', set(:));
    end
    
end