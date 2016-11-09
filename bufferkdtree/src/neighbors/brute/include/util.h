/*
 * util.h
 *
 * Copyright (C) 2013-2016 Fabian Gieseke <fabian.gieseke@di.ku.dk>
 * License: GPL v2
 *
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

/**
 * Sets default parameters.
 *
 *@param *params Pointer to struct containing the parameters
 */
void set_default_parameters(BRUTE_PARAMETERS *params);

/**
 * Checks parameters.
 *
 *@param *params Pointer to struct containing the parameters
 */
void check_parameters(BRUTE_PARAMETERS *params);


#endif /* BRUTE_INCLUDE_UTIL_H_ */
