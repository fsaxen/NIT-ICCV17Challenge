function varargout = vector(varargin)
%VECTOR Summary of this function goes here
%   Detailed explanation goes here

    % Import library to access private functions
    import libPerformance.*
   
    % Check input parameters
    if nargin < 3
        error('You must provide the prediction, ground truth and at least one measure.');
    end
    
    p = varargin{1};
    gt = varargin{2};
    
    % Get confusion matrix (which also checks the input parameters p and gt)
    [cm, threshold] = confusionMatrix(p, gt);
    TP = cm(:, 1);
    FP = cm(:, 2);
    TN = cm(:, 3);
    FN = cm(:, 4);
    
    nmeasures = nargout;
    if nargout > (nargin - 2)
        error('Too many output parameters.');
    elseif nargout < (nargin - 2)
        error('Not enough output parameters.');
    end
    
    varargout = cell(nmeasures, 1);
    for i = 1 : nmeasures
        measure = lower(varargin{i + 2});
        switch measure
            case 'confusionmatrix'
                varargout{i} = cm;
            case 'threshold'
                varargout{i} = threshold;

            case {'tpr', 'recall', 'sensitivity'} % true positive rate
                tpr = TP ./ (TP + FN);
                varargout{i} = tpr;
            case {'fnr', 'missrate'} %  False negative rate
                fnr = FN ./ (TP + FN);
                varargout{i} = fnr;
            case 'fpr' % false positive rate
                fpr = FP ./ (FP + TN);
                varargout{i} = fpr;
            case {'tnr', 'specificity'} % true negative rate
                tnr = TN ./ (FP + TN);
                varargout{i} = tnr;
            case 'precision'
                precision = TP ./ (TP + FP);
                varargout{i} = precision;
            case {'accuracy', 'mif1'} % Accuracy == micro f1
                acc = (TP + TN) ./ (TP + TN + FP + FN);
                varargout{i} = acc;
            case 'mf1' % macro f1
                mf1 = 2 .* TP ./ (2 .* TP + FN + FP);
                varargout{i} = mf1;
            
            otherwise
                error(['Unkown measure: ', measure]);
        end
        
    end
    
    
end

