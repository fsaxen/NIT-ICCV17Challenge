function makelibsvm
% MAKE        Compiles libsvm for matlab
%
% MAKE  This make.m is for MATLAB and OCTAVE under Windows, Mac, and Unix
    

    try
        cd +libSvm/private

        Type = ver;
        % This part is for OCTAVE
        if(strcmp(Type(1).Name, 'Octave') == 1)
            mex libsvm-3.20/matlab/libsvmread.c
            mex libsvm-3.20/matlab/libsvmwrite.c
            mex libsvm-3.20/matlab/libsvmtrain.c libsvm-3.20/svm.cpp libsvm-3.20/matlab/svm_model_matlab.c
            mex libsvm-3.20/matlab/libsvmpredict.c libsvm-3.20/svm.cpp libsvm-3.20/matlab/svm_model_matlab.c
        % This part is for MATLAB
        % Add -largeArrayDims on 64-bit machines of MATLAB
        else
            mex CFLAGS="\$CFLAGS -std=c99" -largeArrayDims libsvm-3.20/matlab/libsvmread.c
            mex CFLAGS="\$CFLAGS -std=c99" -largeArrayDims libsvm-3.20/matlab/libsvmwrite.c
            mex CFLAGS="\$CFLAGS -std=c99" -largeArrayDims libsvm-3.20/matlab/libsvmtrain.c libsvm-3.20/svm.cpp libsvm-3.20/matlab/svm_model_matlab.c
            mex CFLAGS="\$CFLAGS -std=c99" -largeArrayDims libsvm-3.20/matlab/libsvmpredict.c libsvm-3.20/svm.cpp libsvm-3.20/matlab/svm_model_matlab.c
        end
        
        cd ../..
    catch
        fprintf('makelibsvm.m failed. Please check README about detailed instructions.\n');
    end

end

