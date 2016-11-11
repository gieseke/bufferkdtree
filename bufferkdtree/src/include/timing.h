/*
 * timing.h
 *
 * Copyright (C) 2013-2016 Fabian Gieseke <fabian.gieseke@di.ku.dk>
 * License: GPL v2
 *
 */
#ifndef INCLUDE_TIMING_H_
#define INCLUDE_TIMING_H_

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

// don't use time if not specified
#ifndef TIMING
#define TIMING 0
#endif

// struct for input parameters
typedef struct timer_struct {

	long start_time;
	double elapsed_time;
	double elapsed_time_total;

} TIMER;

#define INIT_MY_TIMER init_my_timer
#define START_MY_TIMER start_my_timer
#define STOP_MY_TIMER stop_my_timer
#define GET_MY_TIMER get_my_timer

/**
 * Helper method for computing the current time (w.r.t to an offset).
 *
 *@return System in in microseconds
 */
long get_system_time_in_microseconds(void);

/**
 * Initializes a timer
 *
 * @param *timer Pointer to timer struct instance
 */
void init_my_timer(TIMER *timer);

/**
 * Starts a given timer
 *
 * @param *timer Pointer to timer struct instance
 */
void start_my_timer(TIMER *timer);

/**
 * Stops a given timer
 *
 * @param *timer Pointer to timer struct instance
 */
void stop_my_timer(TIMER *timer);

/**
 * Returns the time measured by a given timer
 *
 * @param *timer Pointer to timer struct instance
 * @return Passed time in seconds
 */
double get_my_timer(TIMER *timer);

// timing macros
#if TIMING > 0
#define DEFINE_TIMER(num) long start_time##num = 0; double elapsed_time##num = 0.0f; double elapsed_time_total##num = 0.0f;
#define DECLARE_TIMER(num) extern long start_time##num; extern double elapsed_time##num; extern double elapsed_time_total##num;
#define START_TIMER(num) start_time##num = get_system_time_in_microseconds();
#define STOP_TIMER(num) elapsed_time##num = (((double)get_system_time_in_microseconds())-((double)start_time##num)); elapsed_time_total##num+=elapsed_time##num;
#define GET_TIME(num) (double)(1.0*elapsed_time_total##num / 1000000.0)
#define RESET_TIMER(num) start_time##num = 0; elapsed_time##num = 0.0f; elapsed_time_total##num = 0.0f;
#else
#define DEFINE_TIMER(num)
#define DECLARE_TIMER(num)
#define START_TIMER(num)
#define STOP_TIMER(num)
#define GET_TIME(num)
#define RESET_TIMER(num)
#endif


#endif /* INCLUDE_TIMING_H_ */
