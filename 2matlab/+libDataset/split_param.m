function param = split_param()

%   split_param
%       .method =
%           'random' :    Randomly split dataset into k parts (default) 
%           'stratified': Split dataset such that all k parts have the same
%                         number of samples for each class. Must be a
%                         discrete supervised learning problem.
%           'random_subjects': Split set according to subject, such that
%                              all samples of each subject are together
%                              and not spread out threw the sets.
%
%       .k =
%           Number of sets to split (default 2).
%
%       .ratio =
%           Scalar or vector of ratios with k elements, wheareas each element
%           corresponds to a set. Each ratio defines the number of samples
%           in the each set relative to all samples. If you provide a
%           scalar, its ratio is used for all sets (default 1/k).
%           If split_param.with_repetition = false, the sum of
%           split_param.ratio must not exceed 1.
%           If split_param.with_repetition = true, each ratio element must
%           be 0 < ratio < 1.
%           
%       .with_repetition =
%           true : Sample with repetition. A sample may occur in multiple sets, but only once in each set.
%           false: Sample without repetition (default). Each sample only occurs once in one set.

    param = struct();
end