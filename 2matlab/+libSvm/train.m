function [ model ] = train( data, svm_param )
% train - Train an SVM/SVR model
%   
%   model = train(data, svm_param) trains an svm model based on the svm
%                                  parameters specified in svm_param.
%
%   data : dataset structure. See libDataset.create_dataset() for details.
%
%   svm_param : Struct element containint the training setup. See
%               svm_param.m for details.
%
    import libSvm.*

    % Set standard parameters
    if ~isfield(svm_param, 'library')
        svm_param.library = 'libsvm';
    end
    if ~isfield(svm_param, 'type')
        warning('No svm_param.type specified. Assuming SVMm (multiclass).');
        svm_param.type = 'SVMm';
    end
    if ~isfield(svm_param, 'predict_fast')
        svm_param.predict_fast = true;
    end
    if ~isfield(svm_param, 'kernel')
        svm_param.kernel = 'linear';
    end
    if ~isfield(svm_param, 'C')
        svm_param.C = 1.0;
    end
    if ~isfield(svm_param, 'epsilon')
        svm_param.epsilon = 0.1;
    end
    if ~isfield(svm_param, 'gamma')
        svm_param.gamma = 1.0 ./ length(data.sample_idx);
    end
    if ~isfield(svm_param, 'degree')
        svm_param.degree = 3.0;
    end
    if ~isfield(svm_param, 'coef0')
        svm_param.coef0 = 0.0;
    end
    if ~isfield(svm_param, 'verbose')
        svm_param.verbose = false;
    end
    if ~isfield(svm_param, 'weights')
        svm_param.weights = [];
    end
    
    % Check dataset
    % Support Vector Machine can only train a single predictor at a time
    % and does only work for supervised problems.
	switch svm_param.type
        case 'SVR'
            libDataset.util_check_dataset(data, 'y', 'supervised', 'one_predictor_idx');
        case {'SVM', 'SVMm', 'SVMb'}
            libDataset.util_check_dataset(data, 'y', 'supervised', 'one_predictor_idx', 'discrete_predictors');
        otherwise
            error('Unsupported Support Vector Machine type');
    end
    
        
    
    % select model params
    if isfield(svm_param, 'param_search') && isfield(svm_param.param_search, 'enable') && svm_param.param_search.enable
        % search for optimal svm parameter
        [svm_param, param, perf, time] = param_search(data, svm_param);
        
        % print results
        if isfield(svm_param.param_search, 'print') && svm_param.param_search.print

            % Create folder if it does not exist already
            folder = fileparts(svm_param.param_search.filename);
            if ~isempty(folder) && ~exist(folder, 'dir')
                mkdir(folder);
            end
            
            % Save parameter plots
            if strcmp(svm_param.kernel, 'rbf')
                C = log(unique(param(:,1)))/log(2);
                gamma = log(unique(param(:,3)))/log(2);
                perf_mat = reshape(mean(perf,2)', length(C), length(gamma))';
                time_mat = reshape(mean(time,2)', length(C), length(gamma))';
                [C, gamma] = meshgrid(C, gamma);
                fig = figure('Visible','off');
                subplot(211);
                %perf_mat = max(perf_mat, 0) .^ 2;
                contourf(C, gamma, perf_mat);
                xlabel('log2(C)'); ylabel('log2(gamma)'); title(svm_param.param_search.performance_measure); colorbar;
                hold on; plot(log(svm_param.C)/log(2), log(svm_param.gamma)/log(2),'mx', 'LineWidth', 1.5, 'MarkerSize', 10); hold off;
                subplot(212);
                contourf(C, gamma, time_mat);
                xlabel('log2(C)'); ylabel('log2(gamma)'); title('training time'); colorbar;
                hold on; plot(log(svm_param.C)/log(2), log(svm_param.gamma)/log(2),'mx', 'LineWidth', 1.5, 'MarkerSize', 10); hold off;
                %drawnow;
                print(fig,'-dpng',sprintf('%s', svm_param.param_search.filename));
                close(fig);
            elseif strcmp(svm_param.kernel, 'linear') || strcmp(svm_param.kernel, 'intersection')
                fig = figure('Visible','off');
                perf = mean(perf,2);
                time = mean(time,2);
                param = param(:,1);
                C = log(param)/log(2);
                subplot(211);
                plot(C, perf); xlabel('log2(C)'); ylabel(svm_param.param_search.performance_measure);
                subplot(212);
                plot(C, time); xlabel('log2(C)'); ylabel('training time');
                print(fig,'-dpng',sprintf('%s', svm_param.param_search.filename));
                close(fig);
            else
                assert(false, ['Plotting SVM param_search results is not yet supported for kernel: ' svm_param.kernel]);
            end
        end
    end
    
    % set parameter string
    param_string = [];
    switch svm_param.library
        case 'libsvm'
            switch svm_param.type
                case 'SVR'
                    param_string = [param_string ' -s 3'];
                    if isfield(svm_param, 'epsilon')
                        param_string = [param_string ' -p ' num2str(svm_param.epsilon)];
                    end
                case {'SVM', 'SVMm', 'SVMb'}
                    param_string = [param_string ' -s 0'];
                otherwise
                    error('Unsupported Support Vector Machine type');                
            end

            % set kernel and kernel parameters
            if strcmp(svm_param.kernel, 'linear')
                param_string = [param_string ' -t 0'];
            elseif strcmp(svm_param.kernel, 'polynomial')
                param_string = [param_string ' -t 1 -d ' num2str(svm_param.degree) ' -g ' num2str(svm_param.gamma) ' -r ' num2str(svm_param.coef0)];
            elseif strcmp(svm_param.kernel, 'rbf')
                param_string = [param_string ' -t 2 -g ' num2str(svm_param.gamma)];
            elseif strcmp(svm_param.kernel, 'sigmoid')
                param_string = [param_string ' -t 3 -g ' num2str(svm_param.gamma) ' -r ' num2str(svm_param.coef0)];
            elseif strcmp(svm_param.kernel, 'intersection')
                param_string = [param_string ' -t 5'];
            else
                error('Unknown kernel type');
            end

            % set cache size parameter (in MB)
            if isfield(svm_param, 'cache')
                param_string = [param_string ' -m ' num2str(svm_param.cache)];
            else
                %param_string = [param_string ' -m 1000'];
                n_samples = length(data.sample_idx);
                cache = ceil(n_samples^2 * 4 / 1048576);
                cache = max(min([cache 4000]), 4);
                param_string = [param_string ' -m ' num2str(cache)];
            end
        case 'liblinear'
            switch svm_param.type
                case 'SVR'
                    param_string = [param_string '-s 11']; % 0, 6, 7, 11, 12, 13 ?
                    if isfield(svm_param, 'epsilon')
                        param_string = [param_string ' -p ' num2str(svm_param.epsilon)];
                    end
                case {'SVM', 'SVMm', 'SVMb'}
                    param_string = [param_string ' -s 2 -B 1']; % 0, 1, 2, 3, 4, 5 ?
                otherwise
                    error('Unsupported Support Vector Machine type');                
            end
        otherwise
            error('No valid svm library.');
    end
    
    % Set cost parameter
    param_string = [param_string ' -c ' num2str(svm_param.C)];
    
    % Set class specific weight for parameter c (weight*c, for C-SVC (default 1))
    if ~isempty(svm_param.weights)
        switch svm_param.library
            case {'libsvm','liblinear'}
                switch svm_param.type
                    case 'SVR'
                        warning('Parameter "weights" is not supported for this SVM type.')
                    case {'SVM', 'SVMm', 'SVMb'}
                        for i = 1:size(svm_param.weights,1)
                            param_string = [param_string, ' -w', num2str(svm_param.weights(i,1)),' ' ,num2str(svm_param.weights(i,2))];
                        end
                    otherwise
                        error('Unsupported Support Vector Machine type');                
                end
            otherwise
                warning('Parameter "weights" is not supported for this SVM library.')
        end
    end
    
    
    % be quite or verbose?
    if ~svm_param.verbose
        param_string = [param_string ' -q'];
    end

    
    
    % binarize classes for SVMb
    if strcmp(svm_param.type, 'SVMb')
        y_model_in = double(data.y(data.sample_idx, data.predictor_idx) ~= 0); %%%%%%%%%%%% TODO: We should check the predictors first because 0 might not always exist %%%%%%
    else
        y_model_in = double(data.y(data.sample_idx, data.predictor_idx));
        
        if strcmp(svm_param.type, 'SVR') && strcmp(svm_param.library, 'liblinear')
            % Liblinear needs normalized predictors for liblinear
            % regression
            mu_y = mean(y_model_in, 1);
            std_y = std(y_model_in, 1);
            y_model_in = bsxfun(@minus, y_model_in, mu_y);
            y_model_in = bsxfun(@rdivide, y_model_in, std_y); 
        end
    end
        
    switch svm_param.library
        case 'libsvm'
            % Build libsvm if mex-file does not exist
            if(exist('libsvmtrain','file') ~= 3)
                makelibsvm()
            end
            % train with LIBSVM
            % ensure that X is double, as libsvm wants it
            model = libsvmtrain(y_model_in, double(data.x(data.sample_idx,:)), param_string);

            % for linear binary case we can reduce the n support vectors to one to
            % save memory and speed up the prediction
            if isfield(svm_param, 'reduce_support_vectors') && svm_param.reduce_support_vectors
                % only pedict_fast supports reduced_support_vectors, the
                % libsvmpredict creashes if model is modified
                if isfield(svm_param, 'predict_fast') && svm_param.predict_fast
                    kernel_type = model.Parameters(2);
                    if kernel_type == 0 && model.nr_class == 2
                        %model.totalSV_before_reduction = model.totalSV;
                        model.SVs = (model.SVs' * model.sv_coef)';
                        model.sv_coef = 1;
                        model.totalSV = 1;
                    end
                end
            end
        case 'liblinear'
            % Build liblinear if mex-file does not exist
            if(exist('liblineartrain','file') ~= 3)
                makeliblinear()
            end

            % train linear model
            % ensure that X is sparse double matrix, as liblinear wants it
            model = liblineartrain(y_model_in, sparse(double(data.x(data.sample_idx,:))), param_string);
            
            if strcmp(svm_param.type, 'SVR')
                model.w = model.w .* std_y;
                model.b = mu_y;
            end
        otherwise
            error('No valid svm library.');
    end
    
    % process pending events (including CTRL+C)
    drawnow;
    
    %%%%%%%%%%%%%%%%%% TODO: Maybe move to +ml library, because also other ml methods, like RF might have a systematic error to correct %%%%%%%%%%% 
    % fit systematic error correction function
    % (especially needed to convert decision value of SVMb to intensity
    % range)
    if isfield(svm_param, 'fit_correction_function') && svm_param.fit_correction_function.enable
        % get decision values of training data
        svm_param.fit_correction_function.enable = false;
        if strcmp(svm_param.type, 'SVMb')
            [~, ~, dec_val] = svm.predict(data, model, svm_param);
        elseif strcmp(svm_param.type, 'SVMm')
            [dec_val] = svm.predict(data, model, svm_param);
        elseif strcmp(svm_param.type, 'SVR')
            error('Doesnt make sense to fit a correction function to a Regressor (SVR).');
        else
            error('Unknown SVM type');
        end
        
        % fit function to map decision value to intensity as
        % well as possible
        if std(dec_val) > 0.01 * std(double(data.y(data.sample_idx, data.predictor_idx)))
            response_correction_poly = polyfit(dec_val, double(data.y(data.sample_idx, data.predictor_idx)), svm_param.fit_correction_function.degree);
        else
            response_correction_poly = [1 0];
        end
        
        % visualize
        if isfield(svm_param.fit_correction_function, 'visualize') && svm_param.fit_correction_function.visualize
            figure(13);
            scatter(dec_val, double(data.y(data.sample_idx, data.predictor_idx)));
            xlabel('SVM output'); ylabel('ground truth');
            hold on;
            x = linspace(min(dec_val)-0.1*range(dec_val), max(dec_val)+0.1*range(dec_val), 50);
            y = polyval(response_correction_poly, x);
            plot(x, y, '-g');
            hold off;
        end
        % wrap model
        model_struct = struct;
        model_struct.svm = model;
        model_struct.response_correction_poly = response_correction_poly;
        model = model_struct;
    end

end

