%% setup
train_val_or_test = 2; % 0: train, 1: test on validation set, 2: test on test set
data_folder = '../exdata/';

%% import data and extract features from training, validation, and testing set (use train_val_or_test to extract features from each set if needed)
ImportData;
DescriptorExtractor18;

%% train model on training data and test on validation or testing set (choose between validation and testing set with train_val_or_test)
TrainAndTestOneLinearRankSVM;