function [y, e, p] = predict_fast(x, model)

svm_type = model.Parameters(1); 
kernel_type = model.Parameters(2);
nr_class = model.nr_class;
sv = full(model.SVs); % support vectors (not sparse)
num_svs = size(sv, 1); % Number of support vectors
sv_coef = model.sv_coef; % lagrange multipliers
label = model.Label; % class labels (empty for regression)

% special case: constant output
if num_svs == 0
    e = [];
    switch nr_class
        case 1
            p = zeros(size(x, 1), 1);
            y = repmat(model.Label, size(x, 1), 1);
        case 2
            p = repmat(-model.rho, size(x, 1), 1);
            if isempty(label)
                y = p;
            else
                y = label(2-(p > 0));
            end
        otherwise
            error('unsupported case');
    end
    return;    
end


% We only want to use max. x GB of memory in this prediction step.
max_elems = 2 * 1024^3 / 8; % = x GB (8 Byte per element)

% Estimate how many columns we need to store
% The kernel product is the most crucial step
switch(kernel_type)
    case {0, 1, 3, 5}
        max_columns = num_svs; 
        % num_svs    = size(    x * sv'   ,2)
    case 2
        max_columns = 1 + num_svs + 1 + num_svs + size(x, 2) + num_svs;
        % 1          = size(   x2(idx_samples,:)   , 2)
        % num_svs    = size(   x2(idx_samples,:)*ones(1,num_svs)   ,2)
        % 1          = size(   ones(numel(idx_samples), 1)   , 2)
        % num_svs    = size(   ones(numel(idx_samples), 1) * sv2   , 2)
        % size(x, 2) = size(   x(idx_samples,:)   , 2)
        % num_svs    = size(   x(idx_samples,:) * sv'   , 2)
        % => RBF is expensive
    otherwise
        error('Kernel not supported');
end

num_samples = size(x, 1);
max_samples = floor(max_elems / max_columns); % Maximum number of samples computed at one time
num_elems = size(x,1) * max_columns; % Highest possible number of elemens in one matrix if computed entirely

% output initialization        
y = zeros(num_samples, 1);
e = [];
if(nr_class > 2)
    p = zeros(num_samples, nr_class * (nr_class - 1) / 2);
else
    p = y;
end

% Preliminary calculations for rbf kernel
if(kernel_type == 2)
    sv2 = sum(sv .* sv,2)'; % = norm(sv).^2
    x2 = sum(x .* x,2); % = norm(x).^2
end

% Too much memory usage -> Predict x in chunks
idx_sample = 1;
for sample_set = 1 : ceil(num_elems / max_elems)
    num_samples_new = min(idx_sample + max_samples, num_samples);
    idx_samples = idx_sample : num_samples_new; % indices of samples in x to adress
    idx_sample = num_samples_new + 1; % prepare for the next iteration.

    % Linear kernel can be calculate much faster if the weight vector is
    % combined in advance. For multi-class more difficulty
    if(kernel_type == 0 && nr_class == 2)
        p(idx_samples) = x(idx_samples, :) * (sv' * sv_coef) - model.rho;
        if isempty(label)
            y(idx_samples) = p(idx_samples);
        else
            y(idx_samples) = label(2-(p(idx_samples) > 0));
        end
        continue;
    end
    
    % Calculate kernel product
    switch(kernel_type)
        case 0 % linear kernel
            K = x(idx_samples,:) * sv';
        case 1 % polynomial kernel
            degree = model.Parameters(3);
            coef0 = model.Parameters(5);
            gamma = model.Parameters(4);

            K = (gamma * x(idx_samples,:) * sv' + coef0).^degree;
            % for each sample i and for each support vector j -> y(i) =
            % sum_j(sv_ceof(j) * (gamma * dot(x(i),sv(i))+coef0).^d

        case 2 % RBF kernel
            gamma = model.Parameters(4);

            K = exp(-gamma * (x2(idx_samples,:)*ones(1,num_svs) + ones(numel(idx_samples), 1)*sv2 - 2*x(idx_samples,:)*sv'));
            % for each sample i and for each support vector j -> y(i) = sum_j(sv_ceof(j) * exp(-gamma * norm(x(i)-sv(i)).^2)

        case 3 % Sigmoid kernel
            coef0 = model.Parameters(5);
            gamma = model.Parameters(4);

            K = tanh(gamma * x(idx_samples,:)*sv' + coef0);
            % for each sample i and for each support vector j -> y(i) =
            % sum_j(sv_ceof(j) * tanh(gamma * dot(x(i),sv(i))+coef0)

        case 5 % Intersection kernel
            K = pdist2(x(idx_samples,:), sv, @(x,Y) sum(bsxfun(@min,x,Y),2));

        otherwise % error
            error('Unkown kernel type');
    end
    
    

    switch(nr_class)
        case 1 % one-class SVM
            error('One-Class svm not supported yet');
        case 2 % two-class SVM or SVR
            p(idx_samples) = K * sv_coef - model.rho;
            if svm_type == 3 || svm_type == 4 % SVR
                y = p;
            else % SVM
%                 y(idx_samples) = p(idx_samples) > 0;
                y(idx_samples) = label(2 - (p(idx_samples) > 0));
            end

        otherwise % Multi-Class SVM
            nr_permutations = nr_class * (nr_class-1) / 2;
            nSV = model.nSV;
            coef = zeros(sum(nSV), nr_permutations);
            idxij = 1;
            class_i = [];
            class_j = [];
            o1 = ones(num_samples_new, 1);
            for i = 1 : nr_class
                for j = i + 1 : nr_class
                    idxi = sum(nSV(1:i-1))+1 : sum(nSV(1:i));
                    idxj = sum(nSV(1:j-1))+1 : sum(nSV(1:j));
                    coef(idxi,idxij) = sv_coef(idxi,j-1);
                    coef(idxj,idxij) = sv_coef(idxj,i);
                    class_i = [class_i label(i)];
                    class_j = [class_j label(j)];
                    idxij = idxij + 1;
                end
            end
            class_i = o1 * class_i;
            class_j = o1 * class_j;
            
            p(idx_samples,:) = K * coef - o1 * model.rho';
            probs_thresh = p(idx_samples,:) > 0;
            y(idx_samples) = mode(class_i .* probs_thresh + class_j .* ~probs_thresh, 2);
    end

   
end


end
