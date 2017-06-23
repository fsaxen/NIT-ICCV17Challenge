function dataset_redistributed = redistribute(data, redistribute_param)
% Redistribute the dataset, such that the label frequency matches a certain
% distribution, which is given by redistribute_param.
    t0 = tic;
    
    import dataset.*
    
    % Check input parameters
    if nargin < 2
        error('Dataset and redistribute_param must be provided');
    end
    
    % Check if dataset is a correct struct.
    % Additionally, check if there are multiple predictors selected,
    % because we can only deal with one predictor to redistribute the
    % dataset.
    % Also check, if only discrete (classification labels) predictors are
    % chosen, because we cannot handle continuous (regression) values.
    libDataset.util_check_dataset(data, 'supervised', 'one_predictor_idx', 'discrete_predictors');
    
    % Set standard parameters
    if ~isfield(redistribute_param, 'downsampling_type')
        redistribute_param.downsampling_type = 'random';
    end
    if ~isfield(redistribute_param, 'upsampling_type')
        redistribute_param.upsampling_type = 'none';
    end
    if ~isfield(redistribute_param, 'type')
        redistribute_param.type = 'min';
    end
    if ~isfield(redistribute_param, 'factor')
        factor = 1.0;
    else
        factor = redistribute_param.factor;
    end
    if ~isfield(redistribute_param, 'visualize')
        redistribute_param.visualize = false;
    end
    
    db_copy_required = false;
    
    % Set handle to downsampling method
    switch redistribute_param.downsampling_type
        case 'none'
            h_downsample = @(x, t) x;
        case 'random'
            h_downsample = @downsample_random;
        case 'kmeans'
            db_copy_required = true;
            h_downsample = @downsample_kmeans;
        otherwise
            error(['Downsampling technique ', redistribute_param.downsampling_type, ' not yet implemented.']);
    end
    
    % Set handle to upsampling method
    switch redistribute_param.upsampling_type
        case 'none'
            h_upsample = @(x, t) x;
        case 'random'
            h_upsample = @upsample_random;
        case 'smote'
            db_copy_required = true;
            h_upsample = @upsample_smote;
        otherwise
            error(['Upsampling technique ', redistribute_param.downsampling_type, ' not yet implemented.']);
    end
    
    % Calculate histogram of labels
    label = data.y(data.sample_idx, data.predictor_idx);
    classes = unique(label); % e.g.  [0 1 2 3]
    num_classes = length(classes);
    class_hist = hist(label, classes); % e.g. [10 20 7 3]
    [class_hist_sorted, class_hist_sorted_idx] = sort(class_hist, 'ascend'); % e.g. [3 7 10 20], [4 3 1 2]
    classes_sorted = classes(class_hist_sorted_idx); % e.g. [3 2 0 1]
    
    % Check ith_order parameter
    if strcmp(redistribute_param.type, 'ith_order')
        if ~isfield(redistribute_param, 'i')
            error('For type ith_order you must provide redistribute_param.i.');
        elseif rem(redistribute_param.i, 1) ~= 0 || redistribute_param.i < 1 || redistribute_param.i > num_classes
            error('redistribute_param.i must be a positive integer in the range 1 <= i <= num_classes.');
        end     
    end
    
    % Calculate threshold
    switch redistribute_param.type
        case 'min'
            threshold = ones(1, num_classes) * class_hist_sorted(1) .* factor;
        case 'max'
            threshold = ones(1, num_classes) * class_hist_sorted(end) .* factor;
        case 'mean'
            threshold = ones(1, num_classes) * round(mean(class_hist) .* factor);
        case 'median'
            threshold = ones(1, num_classes) * median(class_hist) .* factor;
        case 'ith_order'
            threshold = ones(1, num_classes) * class_hist_sorted(redistribute_param.i) .* factor;
        case 'damping'
            if redistribute_param.damping.k > num_classes
                warning('redistribute_param.damping.k > number of predictor classes! setting k to that number.');
                redistribute_param.damping.k = nun_classes;
            end
            nk = class_hist_sorted(num_classes - redistribute_param.damping.k + 1);
            s = redistribute_param.damping.beta * nk / nk.^(1-redistribute_param.damping.alpha);
            threshold = ceil(s * class_hist_sorted.^(1-redistribute_param.damping.alpha)) .* factor;
        otherwise
            error(message('Redistribution type not yet implemented.'));
    end

    % Warn if threshold exceeds most dominant class or falls below
    % least dominant class.
    if any(threshold > class_hist_sorted(end))
        warning('Maybe unwanted? Redistribution with more samples than most frequent class.');
    end
    if any(threshold < class_hist_sorted(1))
        warning('Maybe unwanted? Redistribution with less samples than least fequent class.');
    end
            
    % Redistribute
    dataset_redistributed = balance(data, round(threshold), class_hist_sorted, classes_sorted, h_downsample, h_upsample, db_copy_required);

    % Measure elapsed time to calculate redistribution
    t = toc(t0);
    
    % Visualize input and output distribution
    if redistribute_param.visualize
        % Plot input histogram
        if isfield(redistribute_param, 'figure')
            figure(redistribute_param.figure);
        else
            figure();
        end
        subplot(211);
        bar(class_hist)
        ylim([0 sum(class_hist)]);
        set(gca, 'XTickLabel', double(classes));
        grid on;
        xlabel('class label');
        ylabel('frequency');
        title('Original distribution');
        
        % Plot redistributed histogram
        subplot(212);
        y = dataset_redistributed.y(dataset_redistributed.sample_idx, dataset_redistributed.predictor_idx);
        samples_desired(class_hist_sorted_idx) = threshold;
        out_hist = hist(y, num_classes);
        h = bar([out_hist; samples_desired]');
        legend(h, {'Output', 'Desired'});
%         ylim([0 sum(class_hist)]);
        set(gca, 'XTickLabel', double(unique(y)));
        grid on;
        xlabel('class label');
        ylabel('frequency');
        title(sprintf('Output distribution in %.2f seconds', t));
    end
end