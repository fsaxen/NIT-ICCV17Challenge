function data_o = downsample_kmeans( data, num_samples)
% DOWNSAMPLE_KMEANS downsamples data and creates new samples based on the
% nearest neighbors of data

    % Get samples
    x = data.x(data.sample_idx, :);
    % Calculate num_samples clusters and get the cluster centers
    [~, c] = kmeans(x, num_samples);
    % Create labels
    label = data.y(data.sample_idx(1), data.predictor_idx) * ones(num_samples, 1);
    
    % Create dataset from cluster centers with labels
    data_o = libDataset.create_dataset(c, label);
    
    if isfield(data, 'norm_values')
        data_o.norm_values = data.norm_values;
    end
    if isfield(data, 'subject')
        data_o.subject = -ones(size(c, 1), 1);
    end
    if isfield(data, 'filename')
        data_o.filename = cell(size(c, 1), 1);
        for i = 1 : size(nn_samples, 1)
            data_o.filename{i} = 'kmeans sample without filename.';
        end
    end
end

