#include "include/timing.h"

/* --------------------------------------------------------------------------------
 * Helper method for computing the current time (w.r.t to an offset).
 * --------------------------------------------------------------------------------
 */
long get_system_time_in_microseconds(void) {
	struct timeval tempo;
	gettimeofday(&tempo, NULL);
	return tempo.tv_sec * 1000000 + tempo.tv_usec;
}

void init_my_timer(TIMER *timer){
	timer->start_time = 0;
	timer->elapsed_time = 0.0f;
	timer->elapsed_time_total = 0.0f;
}

void start_my_timer(TIMER *timer){
	timer->start_time = get_system_time_in_microseconds();
}

void stop_my_timer(TIMER *timer){
	timer->elapsed_time = (double)get_system_time_in_microseconds() - timer->start_time;
	timer->elapsed_time_total += timer->elapsed_time;
}

double get_my_timer(TIMER *timer){
	return (double)(1.0*timer->elapsed_time_total / 1000000.0);
}

