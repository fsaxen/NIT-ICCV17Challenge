function makeliblinear
% Compiles liblinear for matlab

    try
        cd +libSvm/private
    
        % This part is for OCTAVE
        if(exist('OCTAVE_VERSION', 'builtin'))
            mex liblinear-2.01/matlab/liblinearsvmread.c
            mex liblinear-2.01/matlab/liblinearsvmwrite.c
            mex -Iliblinear-2.01/ liblinear-2.01/matlab/liblineartrain.c liblinear-2.01/matlab/linear_model_matlab.c liblinear-2.01/linear.cpp liblinear-2.01/tron.cpp liblinear-2.01/blas/daxpy.c liblinear-2.01/blas/ddot.c liblinear-2.01/blas/dnrm2.c liblinear-2.01/blas/dscal.c
            mex -Iliblinear-2.01/ liblinear-2.01/matlab/liblinearpredict.c liblinear-2.01/matlab/linear_model_matlab.c liblinear-2.01/linear.cpp liblinear-2.01/tron.cpp liblinear-2.01/blas/daxpy.c liblinear-2.01/blas/ddot.c liblinear-2.01/blas/dnrm2.c liblinear-2.01/blas/dscal.c
        % This part is for MATLAB
        % Add -largeArrayDims on 64-bit machines of MATLAB
        else
            mex CFLAGS="\$CFLAGS -std=c99" -v -largeArrayDims liblinear-2.01/matlab/liblinearsvmread.c
            mex CFLAGS="\$CFLAGS -std=c99" -largeArrayDims liblinear-2.01/matlab//liblinearsvmwrite.c
            mex CFLAGS="\$CFLAGS -std=c99" -Iliblinear-2.01/ -largeArrayDims liblinear-2.01/matlab/liblineartrain.c liblinear-2.01/matlab/linear_model_matlab.c liblinear-2.01/linear.cpp liblinear-2.01/tron.cpp liblinear-2.01/blas/daxpy.c liblinear-2.01/blas/ddot.c liblinear-2.01/blas/dnrm2.c liblinear-2.01/blas/dscal.c
            mex CFLAGS="\$CFLAGS -std=c99" -Iliblinear-2.01/ -largeArrayDims liblinear-2.01/matlab/liblinearpredict.c liblinear-2.01/matlab/linear_model_matlab.c liblinear-2.01/linear.cpp liblinear-2.01/tron.cpp liblinear-2.01/blas/daxpy.c liblinear-2.01/blas/ddot.c liblinear-2.01/blas/dnrm2.c liblinear-2.01/blas/dscal.c
        end
        cd ../..
    catch
        fprintf('makeliblinear.m failed. Please check README about detailed instructions.\n');
    end
