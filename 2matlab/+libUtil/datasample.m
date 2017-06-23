function [y, i] = datasample(varargin)
% Reimplementation of datasample function without using the statistics
% toolbox.
% Parameter 'Weights' not yet implemented.

    % Check number of input and output arguments
    narginchk(2,8);
    nargoutchk(1,2);
   
    % Set default input parameters
    p = inputParser;
    defaultSeed = rng;
    defaultDim = 1;
    defaultReplace = true;
    
    % Check input parameters
    addOptional(p,'seed',defaultSeed,@(x) isstruct(x) && isfield(x,'Type') && isfield(x, 'Seed') && isfield(x,'State'));
    addRequired(p,'data',@(x) ~isempty(x));
    addRequired(p,'k',@isnumeric);
    addOptional(p,'dim',defaultDim,@isnumeric);
    addParameter(p,'Replace',defaultReplace,@islogical);
    addParameter(p,'Weights',[]);

    parse(p, varargin{:});

    % Weights not supported yet
    if ~isempty(p.Results.Weights)
        error(message('Weights not supported yet.'));
    end
    
    % Get variables
    replace = p.Results.Replace;
    x = p.Results.data;
    dim = p.Results.dim;
    k = p.Results.k;
    s = p.Results.seed;
    n = size(x,dim);
    defaultStream=any(strcmp('seed', p.UsingDefaults));
    
    % Sample with replacement
    if replace
        if n == 0
            if k == 0
                i = zeros(0,1);
            else
                error(message('stats:datasample:EmptyData'));
            end

        else % unweighted sample
            if defaultStream
                i = randi(n,1,k);
            else
                i = randi(s,n,1,k);
            end
        end        
        % Sample without replacement
    else
        if k > n
            error(message('stats:datasample:SampleTooLarge'));

        else % unweighted sample
            if defaultStream
                i = randperm(n,k);
            else
                i = randperm(s,n,k);
            end
        end
    end

    % Use the index vector to sample from the data.
    if ismatrix(x) % including vectors and including dataset or table
        if dim == 1
            y = x(i,:);
        elseif dim == 2
            y = x(:,i);
        else
            reps = [ones(1,dim-1) k];
            y = repmat(x,reps);
        end
    else % N-D
        subs = repmat({':'},1,max(ndims(x),dim));
        subs{dim} = i;
        y = x(subs{:});
    end
end