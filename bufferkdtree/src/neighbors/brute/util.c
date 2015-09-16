/* 
 * util.c
 */

#include "include/util.h"
#include "include/global.h"

/* --------------------------------------------------------------------------------
 * Sets default parameters.
 * --------------------------------------------------------------------------------
 */
void set_default_parameters(Parameters *params) {

	// default parameter values
	params->n_neighbors = 10;
	params->num_threads = 1;
	params->verbosity_level = 1;

}

/* --------------------------------------------------------------------------------
 * Checks parameters.
 * --------------------------------------------------------------------------------
 */
void check_parameters(Parameters *params) {

	// parameter checks
	if ((params->n_neighbors < 1) || (params->n_neighbors > 50)) {
		printf("Error: The parameter k must be > 0 and <= 50)\nExiting ...\n");
		exit(1);
	}

}
