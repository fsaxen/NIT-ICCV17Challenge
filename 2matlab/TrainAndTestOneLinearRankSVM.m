load([data_folder, 'train_AUOld_descriptor18.mat']);
emo_in_featvec_onehot = true;
if train_val_or_test == 1
    fn_val = [data_folder, 'val_AUOld_descriptor18.mat'];
else
    fn_val = [data_folder, 'test_AUOld_descriptor18.mat'];
end
folds = 10;


%% SVM config
ml_param = struct();
svr_param = struct;
svr_param.type = 'SVM';
svr_param.library = 'libsvm';
svr_param.kernel = 'linear';
svr_param.C = 10 ^ 0;

if 1
    % SVM ensemble
    nested_ml_param = struct();
    nested_ml_param.type = 'SVM';
    nested_ml_param.svm_param = svr_param;
    ml_param.type = 'Ensemble';
    ml_param.ensemble_num_models = 75;
    ml_param.num_samples = 0.5; % only 50% of samples
    ml_param.ensemble_param = nested_ml_param;
else
    % single SVM
    ml_param.type = 'SVM';
    ml_param.svm_param = svr_param;
end

%% Cross validation on training dataset
X = vertcat(samples(:).data);
Y = vertcat(samples(:).label);
subj_id = vertcat(samples(:).subject_id);
emotion = vertcat(samples(:).emotion);

pred = NaN(size(Y));

clear acc_
for fold = 1:folds
    idx_test = mod(subj_id,folds) == fold-1;
    
    model = train_rank_svm(X(~idx_test,:),Y(~idx_test),subj_id(~idx_test),emotion(~idx_test), ml_param, emo_in_featvec_onehot);
    pred(idx_test) = predict_rank_svm(X(idx_test,:),subj_id(idx_test,:),emotion(idx_test,:), model);
    acc_(fold) = mean(pred(idx_test) == Y(idx_test));
end

acc = mean(pred == Y);

fprintf('cross-val acc=%.4f (SD %.4f)\n', acc, std(acc_));
fprintf('cross-val acc per emotion: ');
for emo = 1:6
    idx = emotion == emo;
    fprintf('%.2f ', mean(pred(idx) == Y(idx)));
end
fprintf('\n');



%% Train model on all training data
model = train_rank_svm(X,Y,subj_id,emotion, ml_param, emo_in_featvec_onehot);


if train_val_or_test == 0
    return;
end

%% Predict on validation or test set
load(fn_val);
X = vertcat(samples(:).data);
emotion = vertcat(samples(:).emotion);
subj_id = vertcat(samples(:).subject_id);
label = vertcat(samples(:).label);
Y = predict_rank_svm(X, subj_id, emotion, model);

if ~isempty(label)
    fprintf('validation acc=%f\n', mean(Y == label));
end

% ... and save to python script file
if train_val_or_test == 1
    fn_out = 'valid_prediction';
else
    fn_out = 'test_prediction';
end
    
fid = fopen([data_folder, fn_out, '.py'],'w');
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
fprintf(fid, ['with open(''' fn_out '.pkl'', ''wb'') as handle:\n']);
fprintf(fid, '\tpickle.dump(a, handle, protocol=2)\n\n');
fclose(fid);
