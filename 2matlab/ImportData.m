
%% define filenames
if train_val_or_test == 0
    fn_in_filenames = [data_folder, 'train_filenames.txt'];
    fn_in_AUs = [data_folder, 'train_AUOld.txt'];
    fn_in_identity = [data_folder, 'train_face_recognition.txt'];
    fn_out = [data_folder, 'train_AUOld.mat'];
elseif train_val_or_test == 1
    fn_in_filenames = [data_folder, 'val_filenames.txt'];
    fn_in_AUs = [data_folder, 'val_AUOld.txt'];
    fn_in_identity = [data_folder, 'val_face_recognition.txt'];
    fn_out = [data_folder, 'val_AUOld.mat'];
elseif train_val_or_test == 2
    fn_in_filenames = [data_folder, 'test_filenames.txt'];
    fn_in_AUs = [data_folder, 'test_AUOld.txt'];
    fn_in_identity = [data_folder, 'test_face_recognition.txt'];
    fn_out = [data_folder, 'test_AUOld.mat'];
end


%% read sample filenames
[fid, errmsg] = fopen(fn_in_filenames,'rt');
if fid < 0
    error([fn_in_filenames ' - ' errmsg]);
end

sample_filenames = textscan(fid,'%s','Delimiter','\n');
sample_filenames = sample_filenames{1};
if train_val_or_test == 0
    for i = 1:length(sample_filenames)
        [path,name,ext] = fileparts(sample_filenames{i});
        path_dirs = strsplit(path, {'/','\\'});
        sample_filenames{i} = [path_dirs{end} '/' name ext];
    end    
else
    for i = 1:length(sample_filenames)
        [~,name,ext] = fileparts(sample_filenames{i});
        sample_filenames{i} = [name ext];
    end
end
fclose(fid);


%% face recognition results
if train_val_or_test ~= 0
    face_recognition = csvread(fn_in_identity);
    face_recognition = face_recognition(:,2);
end


%% read action unit scores
data = load(fn_in_AUs);


%% fill sample table
samples = struct('sample_id', [], ...
    'subject_id', [],...
    'label', [],...
    'video_name', cell(length(sample_filenames),1),...
    'emotion',[],...
    'data',[]);
for i = 1:length(sample_filenames)
    samples(i).sample_id = i;
    samples(i).data = data(data(:,1) == i-1, 3:end);
    samples(i).video_name = sample_filenames{i};
    if train_val_or_test == 0
        % training set
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
        % validation or test set
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
        samples(i).subject_id = face_recognition(i);
    end
end

%% save sample table
save(fn_out,'samples');
