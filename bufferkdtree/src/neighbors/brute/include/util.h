/* 
 * util.h
 */
#ifndef BRUTE_INCLUDE_UTIL_H_
#define BRUTE_INCLUDE_UTIL_H_

#include <stdlib.h>
#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <pthread.h>
#include <sched.h>
#include <errno.h>
#include <string.h>
#include <ctype.h>

#include "global.h"

/* --------------------------------------------------------------------------------
 * Sets default parameters.
 * --------------------------------------------------------------------------------
 */
void set_default_parameters(BRUTE_PARAMETERS *params);

/* --------------------------------------------------------------------------------
 * Checks parameters.
 * --------------------------------------------------------------------------------
 */
void check_parameters(BRUTE_PARAMETERS *params);


#endif /* BRUTE_INCLUDE_UTIL_H_ */
