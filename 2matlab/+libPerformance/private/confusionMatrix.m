function [cm, threshold] = confusionMatrix(p, gt)
%CONFUSIONMATRIX calculates the confusion matrix for each possible
%threshold.
%   the confusion matrix is computed, such that gt is compared to (p <
%   threshold) for each possible threshold and thus provides size(p,1)
%   confusion matrices.
%   Positive classes must have the value 1, negative classes can have the
%   value 0 or -1.
%   Only for two class problems.
%   The output confusion matrix provides TP, FP, TN, FN in each row for the
%   corresponding threshold value.

    % Number of samples
    N = size(p, 1);

    % Initialize confusion matrix
    cm = [N.*ones(N, 1), zeros(N, 3)]; % TP, FP, TN, FN

    % Check input
    if size(gt, 1) ~= N
        error('Ground truth must have the same number of samples as p (one sample per row).');
    end
    
    if size(p, 2) ~= 1 || size(gt, 2) ~= 1
        error('Ground truth and p must have only one column.');
    end
    
    % Let's find out if we have to sort in ascending or descending order.
    % This depends on the classifier, since the svm e.g. sometimes
    % threshold by p < threshold and sometimes by p > threshold.
%     if abs(sum(p > 0) - sum(gt)) < abs(sum(p < 0) - sum(gt))
        % sort prediction probability
        [threshold, idxs] = sort(p, 'ascend');
%     else
        % sort prediction probability
%         [threshold, idxs] = sort(p, 'descend');
%     end
    
    % get ground truth accordingly
    gts = gt(idxs,:);

    % usually 2 unique lables, 0 and 1 or -1 and 1
    labels = unique(gt);
    
    switch length(labels)
        case 1
            warning('Only one class provided in the ground truth.');
            if labels(1) == 1
                nlabel = [0 N];
            else
                nlabel = [N 0];
            end
        case 2
            % Measure the frequency of each label
            nlabel = hist(gt, length(labels), labels);
        otherwise
            error('Calculating the confusion matrices does only work for 2-class problems.');
    end

    ispositive = gts(1) == 1; %labels(2);
    cm(1, 1) = nlabel(2) - double(ispositive);
    cm(1, 2) = nlabel(1) - double(~ispositive);
    cm(1, 3) = double(~ispositive);
    cm(1, 4) = double(ispositive);
    for n = 2 : N
        ispositive = gts(n) == 1;%labels(2);
        cm(n, 1) = cm(n-1, 1) - double(ispositive);
        cm(n, 2) = cm(n-1, 2) - double(~ispositive);
        cm(n, 3) = cm(n-1, 3) + double(~ispositive);
        cm(n, 4) = cm(n-1, 4) + double(ispositive);
    end
end

