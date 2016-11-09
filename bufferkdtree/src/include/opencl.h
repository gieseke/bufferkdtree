/*
 * opencl.h
 *
 * Copyright (C) 2013-2016 Fabian Gieseke <fabian.gieseke@di.ku.dk>
 *               2013      Justin Heinermann <justin.heinermann@uni-oldenburg.de>
 * License: GPL v2
 *
 */

#ifndef INCLUDE_OPENCL_H_
#define INCLUDE_OPENCL_H_

#ifdef MAC
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include <unistd.h>
#include <stdio.h>
#include <string.h>

// helper macros
#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

#define MAX_KERNEL_SOURCE_LENGTH 100000
#define MAX_KERNEL_CONSTANTS_LENGTH 1000

typedef struct {
	long device_mem_bytes;
	long device_max_alloc_bytes;
} DEVICE_INFOS;

#define ERROR_NO_PLATFORMS -1
#define ERROR_INVALID_PLATFORMS -2
#define ERROR_NO_DEVICES -3
#define ERROR_INVALID_DEVICE -4

/**
 * Initializes the OpenCL context, command_queue, platform, and device.
 *
 * @param platform_number The OpenCL platform number
 * @param *platform Pointer to platform that shall be initialized
 * @param device_number The OpenCL device id/number that shall be used
 * @param *device Pointer to corresponding device
 * @param *context The OpenCL context
 * @param *command_queue The OpenCL command queue
 * @param verbose The verbosity level (0==no output, 1==more output, ...)
 *
 */
void init_opencl(cl_uint platform_number,
		cl_platform_id *platform,
		cl_uint device_number,
		cl_device_id *device,
		cl_context *context,
		cl_command_queue *command_queue,
		int verbose);

/**
 * Initializes a command queue
 * 
 * @param *command_queue The OpenCL command queue
 * @param *device The OpenCL device
 * @param *context The OpenCL context
 * 
 */
void init_command_queue(cl_command_queue *command_queue,
		cl_device_id *device,
		cl_context *context);

/**
 * Prints the build information that stems from a kernel.
 * 
 * @param program The OpenCL program to be built
 * @param device The OpenCL device
 * 
 */
void print_build_information(cl_program program,
		cl_device_id device);

/* --------------------------------------------------------------------------------
 * Gets device infos
 * --------------------------------------------------------------------------------
 */
int get_device_infos(cl_uint platform_number, cl_uint device_number, DEVICE_INFOS *device_infos);

/* --------------------------------------------------------------------------------
 * Checks for OpenCL error.
 * --------------------------------------------------------------------------------
 */
void check_cl_error(cl_int err, const char *file, int line);

/* --------------------------------------------------------------------------------
 * Generates an OpenCL kernel from a source string.
 * --------------------------------------------------------------------------------
 */
cl_kernel make_kernel_from_file(cl_context context, cl_device_id device,
		char *kernel_constants, char *kernel_filename,
		const char *kernel_name);

/* --------------------------------------------------------------------------------
 * Reads a file (filename) and stores output in text.
 * --------------------------------------------------------------------------------
 */
void readfile(char *filename, char **text, unsigned long *size);

#endif /* INCLUDE_OPENCL_H_ */
