#if USE_DOUBLE > 0
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#define FLOAT_TYPE double
#define MAX_FLOAT_TYPE     1.7976931348623158e+308
#define MIN_FLOAT_TYPE     -1.7976931348623158e+308
#else
#define FLOAT_TYPE float
#define MAX_FLOAT_TYPE     3.402823466e+38
#define MIN_FLOAT_TYPE     -3.402823466e+38
#endif

__kernel void nearest_neighbors(__global FLOAT_TYPE* Xtrain,
								__global FLOAT_TYPE* Xtest, 
								__global FLOAT_TYPE* d_min,
								__global int* idx_min,
								int nXtrain, 
								int nXtest,
                                int chunk_start) {

	// one thread is assigned to each test query
	int test_idx = get_global_id(0);

    test_idx = chunk_start + test_idx;

	// no processing if >= nXtest
	if (test_idx >= nXtest) {
		return;
	}

	int i, j;

	// allocate space for private arrays (might become a
	// problem in case K_NN gets very large)
	int idx_min_local[K_NN];
	FLOAT_TYPE d_min_local[K_NN];

	// initialize arrays for nearest neighbor indices and distances
	for (i = 0; i < K_NN; i++) {
		idx_min_local[i] = 0;
		d_min_local[i] = MAX_FLOAT_TYPE;
	}

	// generate private copy of test pattern
	FLOAT_TYPE xtest_private[DIM];
	for (j = 0; j < DIM; j++) {
		xtest_private[j] = Xtest[j * nXtest + test_idx];
	}


	// compute nearest neighbors for the test query
	FLOAT_TYPE d, tmp;
	int tmp_idx;
	for (i = 0; i < nXtrain; i++) {
		// caching: all threads access the same global memory position
		d = 0.0;
		for (j = 0; j < DIM; j++) {
			tmp = Xtrain[i*DIM + j] - xtest_private[j];
			d += tmp*tmp;
		}

        // insert distance and index in private arrays
        j = K_NN-1;
        if(d_min_local[j] > d){

			// update local arrays
            d_min_local[j] = d;
            idx_min_local[j] = i;
            
			// move to right position
            for(;j>0;j--){

	            if(d_min_local[j] < d_min_local[j-1]){

		            //swap distances
		            tmp = d_min_local[j];
		            d_min_local[j] = d_min_local[j-1];
		            d_min_local[j-1] = tmp;

		            //swap indices
		            tmp_idx = idx_min_local[j];
		            idx_min_local[j] = idx_min_local[j-1];
		            idx_min_local[j-1] = tmp_idx;

	            } else {
					// sometimes leads to worse runtime
					break;
				}
			}
        }

	}

	// copy results back to GPU
	for (i = 0; i < K_NN; i++) {
		idx_min[i*nXtest + test_idx] = idx_min_local[i];
		d_min[i*nXtest + test_idx] = d_min_local[i];
	}

}

__kernel void transpose_simple(__global FLOAT_TYPE *odata, __global FLOAT_TYPE* idata, int height, int width)
{
    unsigned int i = get_global_id(0);
    unsigned int j = get_global_id(1);
    
    if (i < height && j < width)
    {
        unsigned int in  = i*width + j;
        unsigned int out = j*height + i;
        odata[out] = idata[in]; 
    }

}

