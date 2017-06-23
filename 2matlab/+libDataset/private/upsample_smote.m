function data = upsample_smote(data, num_samples)
% upsample_smote upsamples data_in and generates new samples to fill up the
% dataset, such that data_out contains num_samples samples.

    n = size(data.sample_idx, 1);
    x = data.x(data.sample_idx, :);
    
    % We can not work with only one sample. There is no nearest neighbor
    if n == 1
        return;
    end

    % We cannot generate more samples than original samples squared.
    if num_samples > n^2
        num_samples = n^2;
    end
    
    % Calculate k
    % For each k, we can generate n samples
    k = ceil((num_samples - n) / n);
    
    % Calculate the k nearest neighbors (the first nearest neighbor is
    % always the sample itself. We don't count that :)
    nn_idx = knnsearch(x, [], k);
%     nn_idx = nn_idx';

    % Calculate new samples by interpolating between x and its nearest
    % neighbors. The original implementation chooses one of the nearest
    % neighbor randomly. We calculate the sample for each nearest neighbor
    % and randomly subsample afterwards (which is not the same, but I dont
    % think it will make a huge difference).
    [r, ~] = size(nn_idx);
    nn_samples = zeros(k * n, size(x, 2));
%     th=0.3;
    idx = 1;
    for i=1:r
        for j=1:k
            index = nn_idx(i,j);
            th = rand; % set it to a static value = 0.3 might perform better.
            nn_samples(idx, :)=(1-th).*x(i,:) + th.*x(index,:);
            idx = idx + 1;
        end
    end
    
    % Randomly subsample from new_samples to finally get num_samples
    % samples.
    nn_samples = libUtil.datasample(nn_samples, num_samples - n, 'Replace', false);
    nn_labels = ones(size(nn_samples, 1), 1) * data.y(data.sample_idx(1), data.predictor_idx);

    % Create new dataset
    nn_dataset = libDataset.create_dataset(nn_samples, nn_labels);
    
    if isfield(data, 'norm_values')
        nn_dataset.norm_values = data.norm_values;
    end
    if isfield(data, 'subject')
        nn_dataset.subject = -ones(size(nn_samples, 1), 1);
    end
    if isfield(data, 'filename')
        nn_dataset.filename = cell(size(nn_samples, 1), 1);
        for i = 1 : size(nn_samples, 1)
            nn_dataset.filename{i} = 'smote sample without filename.';
        end
    end

    % Append them to the dataset and provide some labels
    data = libDataset.util_add_dataset(data, nn_dataset);
end