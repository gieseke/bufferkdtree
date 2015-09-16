%module wrapper_cpu_double

%{
    #define SWIG_FILE_WITH_INIT
    #include "base.h"
    #include "global.h"
%}

%include "numpy.i"

%init %{
    import_array();
%}

%apply (double* INPLACE_ARRAY2, int DIM1, int DIM2) {(FLOAT_TYPE* Xtrain, int nXtrain, int dXtrain)}
%apply (double* INPLACE_ARRAY2, int DIM1, int DIM2) {(FLOAT_TYPE* Xtest, int nXtest, int dXtest)}

%apply (double* INPLACE_ARRAY2, int DIM1, int DIM2) {(FLOAT_TYPE* distances, int ndistances, int ddistances)}
%apply (int* INPLACE_ARRAY2, int DIM1, int DIM2) {(int* indices, int nindices, int dindices)}


%include "base.h"      
%include "global.h"
