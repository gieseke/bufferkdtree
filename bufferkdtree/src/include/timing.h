/*
 * timing.h
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
#define RESUME_MY_TIMER start_my_timer
#define STOP_MY_TIMER stop_my_timer
#define GET_MY_TIMER get_my_timer

void start_my_timer(TIMER *timer);
void resume_my_timer(TIMER *timer);
void stop_my_timer(TIMER *timer);
double get_my_timer(TIMER *timer);
void init_my_timer(TIMER *timer);



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

/* --------------------------------------------------------------------------------
 * Helper method for computing the current time (w.r.t to an offset).
 * --------------------------------------------------------------------------------
 */
long get_system_time_in_microseconds(void);

#endif /* INCLUDE_TIMING_H_ */
