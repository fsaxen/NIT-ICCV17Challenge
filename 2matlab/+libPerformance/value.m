function [ perf_value, multiplier ] = value( ground_truth, prediction, measure )
% libPerformance.value - Calculate performance value of predicted outcome
% against the ground truth. 
%
% Input:
%
%   ground_truth:  Ground truth data. One sample per row. One predictor per
%                  column.
%
%   prediction:    Predicted outcome with the exact same size of
%                  ground_truth. 
%
%   measure:       String or cell array of strings with the performance
%                  measures. 
%
% Output:
%   
%   perf_value:    Matrix of performance values. Each performance measure
%                  gets one row. Each predictor one column.
%                  See switch case statement below for a list of measures.
%                  If you add '_d' behind any performanc measure, the
%                  predictions will be discretized (rounded to integer
%                  values) first. 
%
%   multiplier:    Matrix of multipliers corresponding to perf_value. Each 
%                  value is only -1 or +1. If +1, the performance value is
%                  better the higher it is. If -1, the performance is
%                  better the lower it is.
%

    import libPerformance.*
    
    % How many performance measures do we have?
    if iscell(measure)
        num_perf = length(measure);
    else
        % Convert performance measure into cell
        num_perf = 1;
        perf_n = cell(1,1);
        perf_n{1} = measure;
        measure = perf_n;
    end
    
    % Check input
    if any(size(ground_truth) ~= size(prediction))
        error('Ground truth and predictions have different dimensions.');
    end
    num_predictors = size(ground_truth, 2);
    
    % Allocate output variable
    perf_value = zeros(num_perf, num_predictors);
    
    % Calculate performance for each predictor individually
    for i = 1 : num_predictors
        
        non_nan_idx = ~isnan(ground_truth(:, i)) & ~isnan(prediction(:, i));
    
        gt = double(ground_truth(non_nan_idx, i));

        min_gt = min(gt); max_gt = max(gt);

        % Calculate performance for each measure individually
        for p = 1 : num_perf
            perf = lower(measure{p});
            
            % must be reinitialized for each measure, as it may be discretized for some
            pred = double(prediction(non_nan_idx, i));

            % Calculate weights
            switch perf
                case {'wmae', 'wrmse', 'wmse'}
                    intens_dist = hist(gt, min_gt:max_gt);
                    class_sample_weights = 1 ./ intens_dist;
                    weights = class_sample_weights(gt+1-min_gt)';
            end

            % Discretize?
            if strcmp(perf(end-1:end), '_d')
                pred = round(pred);
                pred(pred < min_gt) = min_gt;
                pred(pred > max_gt) = max_gt;
                perf = perf(1:end-2);
            end

            % Calculate performance
            switch perf
                case 'mae' % mean absolute error
                    perf_value(p,i) = mean(abs(gt-pred));
                    multiplier = -1;
                case 'rmse' % root mean squeared error
                    perf_value(p,i) = sqrt(mean((gt-pred).^2));
                    multiplier = -1;
                case 'mse' % mean squared error
                    perf_value(p,i) = mean((gt-pred).^2);
                    multiplier = -1;
                case 'wmae' % weighted mean absolute error
                    perf_value(p,i) = sum(weights .* abs(gt-pred)) ./ sum(weights);
                    multiplier = -1;
                case 'wrmse' % weighted root mean squared eror
                    perf_value(p,i) = sqrt(sum(weights .* (gt-pred).^2) ./ sum(weights));
                    multiplier = -1;
                case 'wmse' % weighted mean squared error
                    perf_value(p,i) = sum(weights .* (gt-pred).^2) ./ sum(weights);
                    multiplier = -1;
                case 'r2' % R squared, R^2
                    perf_value(p,i) = 1 - (sum((gt-pred).^2)) ./ (sum((gt-mean(gt)).^2));
                    multiplier = 1;
                case 'corr' % Correlation
                    perf_value(p,i) = corr(gt, pred);
                    multiplier = 1;
                case 'icc11' % Inter Correlation Coefficient ICC(1,1)
                    perf_value(p,i) = performance_ICC(1,'single',[gt pred]);
                    multiplier = 1;
                case 'icc31' % Inter Correlation Coefficient ICC(3,1)
                    perf_value(p,i) = performance_ICC(3,'single',[gt pred]);
                    multiplier = 1;
                case 'accuracy' % Accuracy
                    multiplier = 1;
        %            classes = 0:5;
                    classes = unique(vertcat(unique(gt), unique(pred)));
                    confus = confusionmat(gt, pred, 'order', classes);
                    correct = diag(confus);
                    total = length(gt);
                    perf_value(p,i) = sum(correct) / total;
                case 'mf1' % Macro F1 Measure (default F1)
                    multiplier = 1;
                    classes = unique(vertcat(unique(gt), unique(pred)));
                    n_classes = length(classes);
                    confus = confusionmat(gt, pred, 'order', classes);
                    f1 = zeros(1, n_classes);
                    for c = 1:n_classes
                        re = recall(confus, c);
                        pr = precision(confus, c);
                        f1(c) = 2 * re * pr;
                        if f1(c) > 1e-4
                            f1(c) = f1(c) / (re + pr);
                        end
                    end
                    perf_value(p,i) = mean(f1);
                case 'mif1' % Micro F1 Measure = Accuracy
                    multiplier = 1;
                    classes = unique(vertcat(unique(gt), unique(pred)));
                    n_classes = length(classes);
                    confus = confusionmat(gt, pred, 'order', classes);
                    confus_micro_av = zeros(2,2);
                    for c = 1:n_classes
                        % TP
                        confus_micro_av(1,1) = confus_micro_av(1,1) + confus(c, c);
                        % FN
                        confus_micro_av(1,2) = confus_micro_av(1,2) + ...
                            sum(confus(c,:)) - confus(c, c);
                        % FP
                        confus_micro_av(2,1) = confus_micro_av(2,1) + ...
                            sum(confus(:,c)) - confus(c, c);
                    end
                    re = recall(confus_micro_av, 1);
                    pr = precision(confus_micro_av, 1);
                    perf_value(p,i) = 2 * re * pr;
                    if perf_value(p,i) > 1e-4
                        perf_value(p,i) = perf_value(p,i) / (re + pr);
                    end
                case 'mrecall' % Mean Recall
                    multiplier = 1;
        %            classes = 0:5;
                    classes = unique(vertcat(unique(gt), unique(pred)));
                    n_classes = length(classes);
                    confus = confusionmat(gt, pred, 'order', classes);
                    re = zeros(1, n_classes);
                    for c = 1:n_classes
                        re(c) = recall(confus, c);
                    end
                    perf_value(p,i) = mean(re);
                otherwise
                    error('unknown measure');
            end
        end
    end

    
    function p = precision(confus, cls)
        tp = confus(cls, cls);
        s = sum(confus(:,cls));
        if (s)
            p = tp / s;
        else
            p = 0;
        end
    end

    function r = recall(confus, cls)
        tp = confus(cls, cls);
        s = sum(confus(cls,:));
        if (s)
            r = tp / s;
        else
            r = 0;
        end
    end

end

