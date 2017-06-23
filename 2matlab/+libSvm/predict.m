function [y, p] = predict(data, model, svm_param)

    import libSvm.*
    
    if isfield(svm_param, 'fit_correction_function') ...
            && isfield(svm_param.fit_correction_function, 'enable') ...
            && svm_param.fit_correction_function.enable
        
        svm_param.fit_correction_function.enable = false;
        if strcmp(svm_param.type, 'SVMb')
            [~, ~, p] = libSvm.predict(data.x(data.sample_idx, :), model.svm, svm_param);
            y = polyval(model.response_correction_poly, p);
            %y = round(p);
        elseif strcmp(svm_param.type, 'SVMm')
            [dec_val, ~, p] = libSvm.predict(data.x(data.sample_idx, :), model.svm, svm_param);
            y = polyval(model.response_correction_poly, dec_val);
        else
            [dec_val, ~, p] = libSvm.predict(data.x(data.sample_idx, :), model.svm, svm_param);
            y = polyval(model.response_correction_poly, dec_val);
        end
        
    else

        if isfield(svm_param, 'library') && strcmp(svm_param.library, 'liblinear')
            
            % LIBLINEAR
        
            if ~isfield(svm_param, 'predict_fast') || svm_param.predict_fast
                
                p = [];
                if model.bias == 1
                    % Classification
                    switch length(model.Label)
                        case 1
                            y = repmat(model.Label, size(data.sample_idx, 1), 1);
                        case 2
                            p = data.x(data.sample_idx, :) * model.w(1:end-1)';
                            p = p + model.w(end);
                            y = model.Label(2 - (p > 0));
                        otherwise
                            error('Multiclass liblinear predict_fast not supported yet');
                    end
                else
                    % Regression
                    y = data.x(data.sample_idx, :) * model.w';
                    if isfield(model, 'b')
                        y = y + model.b;
                    end
                end
                
            else
                if(exist('liblinearpredict','file') ~= 3)
                    makeliblinear();
                end

                if isfield(svm_param, 'verbose') && svm_param.verbose
                    verbose_param = '';
                else % (default)
                    verbose_param = '-q';
                end
                
                if isfield(data, 'y')
                    labels = double(data.y(data.sample_idx, data.predictor_idx));
                else
                    labels = zeros(size(data.x, 1), 1);
                end

                [y, ~, p] = liblinearpredict( ...
                    labels, ...
                    sparse(double(data.x(data.sample_idx, :))), model, verbose_param);

                if isfield(model, 'b')
                    y = y + model.b;
                end
                
            end
            
            % Correct svm score
            if model.bias == 1 && length(model.Label) == 2 && model.Label(2) > model.Label(1)
                p = -p;
            end
        else
            
            % LIBSVM
            
            if ~isfield(svm_param, 'predict_fast') || svm_param.predict_fast
                
                % predict_fast is default
                [y, ~, p] = predict_fast(data.x(data.sample_idx, :), model);
                
            else
                
                if(exist('libsvmpredict','file') ~= 3)
                    makelibsvm();
                end
                
                if isfield(data, 'y')
                    labels = double(data.y(data.sample_idx, data.predictor_idx));
                else
                    labels = zeros(size(data.x, 1), 1);
                end
                feat = double(data.x(data.sample_idx, :));
                
                [y, ~, p] = libsvmpredict(labels, feat, model, '-q');
                
            end

            % Correct svm score
            if length(model.Label) == 2 && model.Label(2) > model.Label(1)
                p = -p;
            end
        end
        
    end    
end