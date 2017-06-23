load([data_folder, 'AUOld_train_descriptor18.mat']);

if train_val_or_test == 1
    fn_val = [data_folder, 'AUOld_val_descriptor18.mat'];
else
    fn_val = [data_folder, 'AUOld_test_descriptor18.mat'];
end
folds = 10;


%% SVM config
ml_param = struct();
svr_param = struct;
svr_param.type = 'SVM';
svr_param.library = 'libsvm';
if 1
    svr_param.kernel = 'linear';
    if 1
        %svr_param.C = 2 ^ -2;
        svr_param.C = 10 ^ 0;
    else
        svr_param.param_search.enable = true;
        svr_param.param_search.C_range = 2.^(-2:1:5);
        svr_param.param_search.performance_measure = 'mae';
        svr_param.param_search.print = true;
        svr_param.param_search.filename = 'param_sel.png';
        svr_param.param_search.sampling.name = 'random_subject_partition';
    end
else
    % better, but slower
    svr_param.kernel = 'rbf';
    if 0
        svr_param.C = 2 ^ 15;
        svr_param.gamma = 2 ^ -10;
        svr_param.espilon = 0.5;
    else
        ml_param.param_search.enable = true;
        ml_param.param_search.params = {...
            'svm_param.C', 2.^(-5:10:25),...
            'svm_param.gamma', 2.^(-15:5:-5),...%2.^(-10:5:0),...
            'svm_param.epsilon', [0.1 0.5],...
            };
        ml_param.param_search.max_num_training_samples = 500;
        ml_param.param_search.max_num_testing_samples = 5000;
        %            'svm_param.C', logspace(-2, 5, 5),...
        %    'svm_param.gamma', logspace(-7, -1, 5),...
        %    ml_param.param_search.max_num_training_samples = 100;
        ml_param.param_search.performance_measure = 'mae';
        ml_param.param_search.save_to_file = true;
        ml_param.param_search.filename_prefix = 'PoseEstimationMyWithOpenfacePS/ps';
    end
end

if 1
    nested_ml_param = struct();
    nested_ml_param.type = 'SVM';
    nested_ml_param.svm_param = svr_param;
    ml_param.type = 'Ensemble';
    ml_param.ensemble_num_models = 25;
    ml_param.num_samples = 0.5; % only 80% of samples
    ml_param.ensemble_param = nested_ml_param;
else
    ml_param.type = 'SVM';
    ml_param.svm_param = svr_param;
end

%% Data / xval
X = vertcat(samples(:).data);
Y = vertcat(samples(:).label);
subj_id = vertcat(samples(:).subject_id);
emotion = vertcat(samples(:).emotion);

pred = NaN(size(Y));

for fold = 1:folds
    idx_test = mod(subj_id,folds) == fold-1;
    
    model = train_rank_svm(X(~idx_test,:),Y(~idx_test),subj_id(~idx_test),emotion(~idx_test), ml_param);
    pred(idx_test) = predict_rank_svm(X(idx_test,:),subj_id(idx_test,:),emotion(idx_test,:), model);
end

acc = mean(pred == Y);

fprintf('acc=%.4f\n', acc);
fprintf('acc per emotions: ');
for emo = 1:6
    idx = emotion == emo;
    fprintf('%.2f ', mean(pred(idx) == Y(idx)));
end
fprintf('\n');



%% Train model on all data
model = train_rank_svm(X,Y,subj_id,emotion, ml_param);

%% Predict on validation set
load(fn_val);
X = vertcat(samples(:).data);
emotion = vertcat(samples(:).emotion);
subj_id = vertcat(samples(:).subject_id);
Y = predict_rank_svm(X, subj_id, emotion, model);


fid = fopen([data_folder, 'valid_prediction.py'],'w');
fprintf(fid, 'import pickle\n\na = { ');
for i = 1:length(samples)
    if Y(i) == 1
        true_or_fake = 'true';
    else
        true_or_fake = 'fake';
    end
    fprintf(fid, '''%s'' : ''%s''',samples(i).video_name, true_or_fake);
    if i < length(samples)
        fprintf(fid, ', ');
    end
end
fprintf(fid, ' }\n\n');
fprintf(fid, 'with open(''valid_prediction.pkl'', ''wb'') as handle:\n');
fprintf(fid, '\tpickle.dump(a, handle, protocol=2)\n\n');
fclose(fid);
