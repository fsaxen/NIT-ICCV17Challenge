if train_val_or_test == 0
    fn_in_filenames = [data_folder, 'filenames_train.txt'];
    fn_in_AUs = [data_folder, 'AUOld_train_nomax.txt'];
    fn_in_identity = [data_folder, 'face_recognition_train.txt'];
    fn_out = [data_folder, 'AUOld_train.mat'];
elseif train_val_or_test == 1
    fn_in_filenames = [data_folder, 'filenames_val.txt'];
    fn_in_AUs = [data_folder, 'AUOld_val_nomax.txt'];
    fn_in_identity = [data_folder, 'face_recognition_val.txt'];
    fn_out = [data_folder, 'AUOld_val.mat'];
elseif train_val_or_test == 2
    fn_in_filenames = [data_folder, 'filenames_test.txt'];
    fn_in_AUs = [data_folder, 'AUOld_test_nomax.txt'];
    fn_in_identity = [data_folder, 'face_recognition_test.txt'];
    fn_out = [data_folder, 'AUOld_test.mat'];
end

[fid, errmsg] = fopen(fn_in_filenames,'rt');
if fid < 0
    error([fn_in_filenames ' - ' errmsg]);
end

sample_filenames = textscan(fid,'%s','Delimiter','\n');
sample_filenames = sample_filenames{1};
if training_set
    sample_filenames = cellfun(@(x) strrep(x,'Train/',''), sample_filenames, 'UniformOutput', false);
else
    sample_filenames = cellfun(@(x) strrep(x,'Val/',''), sample_filenames, 'UniformOutput', false);
end
fclose(fid);

if ~training_set
    face_recognition_val = csvread(fn_in_identity);
    face_recognition_val = face_recognition_val(:,2);
end


samples = struct('sample_id', [], ...
    'subject_id', [],...
    'label', [],...
    'video_name', cell(length(sample_filenames),1),...
    'emotion',[],...
    'data',[]);

data = load(fn_in_AUs);

for i = 1:length(sample_filenames)
    samples(i).sample_id = i;
    samples(i).data = data(data(:,1) == i-1, 3:end);
    samples(i).video_name = sample_filenames{i};
    if training_set
        [path,name] = fileparts(sample_filenames{i});
        samples(i).subject_id = str2double(path);
        switch upper(name)
            case 'N2H'
                samples(i).label = 1;
                samples(i).emotion = 1;
            case 'N2S'
                samples(i).label = 1;
                samples(i).emotion = 2;
            case 'N2D'
                samples(i).label = 1;
                samples(i).emotion = 3;
            case 'N2A'
                samples(i).label = 1;
                samples(i).emotion = 4;
            case 'N2C'
                samples(i).label = 1;
                samples(i).emotion = 5;
            case 'N2SUR'
                samples(i).label = 1;
                samples(i).emotion = 6;
            case 'S2N2H'
                samples(i).label = 0;
                samples(i).emotion = 1;
            case 'H2N2S'
                samples(i).label = 0;
                samples(i).emotion = 2;
            case 'H2N2D'
                samples(i).label = 0;
                samples(i).emotion = 3;
            case 'H2N2A'
                samples(i).label = 0;
                samples(i).emotion = 4;
            case 'H2N2C'
                samples(i).label = 0;
                samples(i).emotion = 5;
            case {'D2N2SUR','D2N2S'}
                samples(i).label = 0;
                samples(i).emotion = 6;
            otherwise
                error('unknown: %s', name);
        end
    else
        [~,name] = fileparts(sample_filenames{i});
        parts = strsplit(name,'_');
        switch upper(parts{2})
            case 'HAPPINESS'
                samples(i).emotion = 1;
            case 'SADNESS'
                samples(i).emotion = 2;
            case 'DISGUST'
                samples(i).emotion = 3;
            case 'ANGER'
                samples(i).emotion = 4;
            case 'CONTENTMENT'
                samples(i).emotion = 5;
            case 'SURPRISE'
                samples(i).emotion = 6;
            otherwise
                error('unknown: %s', name);
        end
        samples(i).subject_id = face_recognition_val(i);
    end
end

save(fn_out,'samples');
