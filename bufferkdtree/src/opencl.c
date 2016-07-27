/*
 * opencl.c
 *
 *  Created on: 18.08.2014
 *      Author: Fabian Gieseke
 */

#include "include/opencl.h"

/* --------------------------------------------------------------------------------
 * Initializes the OpenCL context, command_queue, platform, and device.
 * --------------------------------------------------------------------------------
 */
void init_opencl(cl_uint platform_number, cl_platform_id *platform,
		cl_uint device_number, cl_device_id *device, cl_context *context,
		cl_command_queue *command_queue, int verbose) {

	int i;

	cl_int err;

	// platform related variables
	cl_uint nplatforms;
	cl_platform_id *platforms;

	// devices related variables
	cl_device_id *devices;
	cl_uint num_devices;

	// get number of available platforms (e.g., Nvidia or AMD)
	err = clGetPlatformIDs(0, NULL, &nplatforms);
	check_cl_error(err, __FILE__, __LINE__);

	// get available platforms
	platforms = (cl_platform_id *) malloc(sizeof(cl_platform_id) * nplatforms);
	err = clGetPlatformIDs(nplatforms, platforms, NULL);
	check_cl_error(err, __FILE__, __LINE__);

	if (verbose > 0) {

		char name[512];
		char vendor[512];
		char version[512];

		printf("\nDetected %i platform(s).\n", nplatforms);

		for (i=0; i < nplatforms; i++){

			printf("Details for platform %i\n", i);
			err = clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, sizeof(vendor), vendor, NULL);
			err |= clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(name), name, NULL);
			err |= clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION, sizeof(version), version, NULL);
			check_cl_error(err, __FILE__, __LINE__);
			printf(" - Vendor: %s\n", vendor);
			printf(" - Name: %s\n", name);
			printf(" - Version: %s\n", version);
		}

	}

	// select platform
	*platform = platforms[platform_number];

	// get number of devices
	err = clGetDeviceIDs(*platform, CL_DEVICE_TYPE_ALL, 1, NULL, &num_devices);
	check_cl_error(err, __FILE__, __LINE__);

	// get all devices
	devices = (cl_device_id*) malloc(sizeof(cl_device_id) * num_devices);
	err = clGetDeviceIDs(*platform, CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);
	check_cl_error(err, __FILE__, __LINE__);

	if (verbose > 0) {

		printf("\nDetected %i devices(s) on platform with id %i.\n", num_devices, platform_number);

		char name[1024];
		char version[1024];
		char driver[1024];
		cl_int num_compute_units;
		cl_ulong mem_in_bytes;
		cl_ulong max_buff_alloc_in_bytes;

		for (i=0; i<num_devices; i++){

			printf("Details for device %i\n", i);
			err = clGetDeviceInfo(devices[i], CL_DEVICE_NAME, sizeof(name), name, NULL);
			err |= clGetDeviceInfo(devices[i], CL_DEVICE_VERSION, sizeof(version), version, NULL);
			err |= clGetDeviceInfo(devices[i], CL_DRIVER_VERSION, sizeof(driver), driver, NULL);
			err |= clGetDeviceInfo(devices[i], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_int), &num_compute_units, NULL);
			err |= clGetDeviceInfo(devices[i], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &mem_in_bytes, NULL);
			err |= clGetDeviceInfo(devices[i], CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &max_buff_alloc_in_bytes, NULL);

			check_cl_error(err, __FILE__, __LINE__);

			printf(" - Name: %s\n", name);
			printf(" - Version: %s\n", version);
			printf(" - Driver: %s\n", driver);
			printf(" - Number of compute units: %i\n", num_compute_units);
			printf(" - Size of memory (GB): %f\n", ((double)mem_in_bytes) / 1073741824);
			printf(" - Maximum memory allocation (GB): %f\n", ((double)max_buff_alloc_in_bytes) / 1073741824);

		}

		printf("\n");
	}


	// generate context
	*device = devices[device_number];
	*context = clCreateContext(NULL, 1, device, NULL, NULL, &err);
	check_cl_error(err, __FILE__, __LINE__);

	// command queue
	if (command_queue != NULL){
		*command_queue = clCreateCommandQueue(*context, *device, 0, &err);
		check_cl_error(err, __FILE__, __LINE__);
	}

	// free memory
	free(platforms);
	free(devices);

}

/* --------------------------------------------------------------------------------
 * Initializes a command queue
 * --------------------------------------------------------------------------------
 */
void init_command_queue(cl_command_queue *command_queue, cl_device_id *device, cl_context *context){

	cl_int err;

	*command_queue = clCreateCommandQueue(*context, *device, 0, &err);
	check_cl_error(err, __FILE__, __LINE__);

}

/* --------------------------------------------------------------------------------
 * Prints the build information that stems from a kernel.
 * --------------------------------------------------------------------------------
 */
void print_build_information(cl_program program, cl_device_id device) {

	cl_int err;
	size_t build_log_size;

	char *build_log;

	// get size of build info
	err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL,
			&build_log_size);
	check_cl_error(err, __FILE__, __LINE__);

	// get build info
	build_log = (char *) malloc(build_log_size);
	err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
			build_log_size, build_log, NULL);
	check_cl_error(err, __FILE__, __LINE__);

	printf("Build Log\n%s\n", build_log);
	// free memory
	free(build_log);

}

/* --------------------------------------------------------------------------------
 * Gets device infos
 * --------------------------------------------------------------------------------
 */
int get_device_infos(cl_uint platform_number, cl_uint device_number, \
		DEVICE_INFOS *device_infos){

	cl_int err;

	// platform related variables
	cl_uint nplatforms;
	cl_platform_id *platforms;

	// devices related variables
	cl_device_id *devices;
	cl_uint num_devices;

	// get number of available platforms (e.g., Nvidia or AMD)
	err = clGetPlatformIDs(0, NULL, &nplatforms);
	check_cl_error(err, __FILE__, __LINE__);

	if (nplatforms == 0){
		return ERROR_NO_PLATFORMS;
	}

	// get available platforms
	platforms = (cl_platform_id *) malloc(sizeof(cl_platform_id) * nplatforms);
	err = clGetPlatformIDs(nplatforms, platforms, NULL);
	check_cl_error(err, __FILE__, __LINE__);

	if (platform_number < 0 || platform_number >= nplatforms){
		return ERROR_INVALID_PLATFORMS;
	}

	// get number of devices
	err = clGetDeviceIDs(platforms[platform_number], CL_DEVICE_TYPE_ALL, 1, NULL, &num_devices);
	check_cl_error(err, __FILE__, __LINE__);

	if (num_devices == 0){
		return ERROR_NO_DEVICES;
	}

	// get all devices
	devices = (cl_device_id*) malloc(sizeof(cl_device_id) * num_devices);
	err = clGetDeviceIDs(platforms[platform_number], CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);
	check_cl_error(err, __FILE__, __LINE__);

	if (device_number < 0 || device_number >= num_devices){
		return ERROR_INVALID_DEVICE;
	}

	char name[512];
	char version[512];
	char driver[512];

	cl_int num_compute_units;
	cl_ulong mem_in_bytes;
	cl_ulong max_buff_alloc_in_bytes;

	err = clGetDeviceInfo(devices[device_number], CL_DEVICE_NAME, sizeof(name), name, NULL);
	err |= clGetDeviceInfo(devices[device_number], CL_DEVICE_VERSION, sizeof(version), version, NULL);
	err |= clGetDeviceInfo(devices[device_number], CL_DRIVER_VERSION, sizeof(driver), driver, NULL);
	err |= clGetDeviceInfo(devices[device_number], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_int), &num_compute_units, NULL);
	err |= clGetDeviceInfo(devices[device_number], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &mem_in_bytes, NULL);
	err |= clGetDeviceInfo(devices[device_number], CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &max_buff_alloc_in_bytes, NULL);
	check_cl_error(err, __FILE__, __LINE__);

	if (device_infos != NULL){
		device_infos->device_mem_bytes = (long) mem_in_bytes;
		device_infos->device_max_alloc_bytes = (long) max_buff_alloc_in_bytes;
	}

	return 0;
}


/* --------------------------------------------------------------------------------
 * Checks for OpenCL error.
 * --------------------------------------------------------------------------------
 */
void check_cl_error(cl_int err, const char *file, int line) {

	if (err != CL_SUCCESS) {
		printf("An OpenCL error with code %i in file %s and line %i occurred ...\n", err, file, line);
		exit(1);
	}

}

/* --------------------------------------------------------------------------------
 * Generates an OpenCL kernel from a source string.
 * --------------------------------------------------------------------------------
 */
cl_kernel make_kernel_from_file(cl_context context, cl_device_id device,
		char *kernel_constants, char *kernel_filename, const char *kernel_name) {

	cl_int err;

	// kernel and program
	cl_kernel kernel;
	cl_program program;

	// read kernel sources
	char *kernel_source;
	unsigned long size;
	readfile(kernel_filename, &kernel_source, &size);
	if (size > MAX_KERNEL_SOURCE_LENGTH - MAX_KERNEL_CONSTANTS_LENGTH) {
		printf("Kernel source file too long ...\n");
		exit(1);
	}

	char outbuff[MAX_KERNEL_SOURCE_LENGTH] = "";
	strcat(outbuff, kernel_constants);
	strncat(outbuff, kernel_source, size);
	strcat(outbuff, "\0");
	const char *outbuff_new = (const char*) outbuff;
	size_t outbuff_new_length = (size_t) strlen(outbuff_new);

	// generate program
	program = clCreateProgramWithSource(context, 1, &outbuff_new,
			&outbuff_new_length, &err);
	check_cl_error(err, __FILE__, __LINE__);

	// generate for all devices
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

	// print build log if needed
	if (err != CL_SUCCESS) {
		printf("Error while compiling file %s\n", kernel_filename);
		print_build_information(program, device);
	}
	check_cl_error(err, __FILE__, __LINE__);

	// generate kernel
	kernel = clCreateKernel(program, kernel_name, &err);
	check_cl_error(err, __FILE__, __LINE__);

	// release program
	clReleaseProgram(program);

	// release memory
	free(kernel_source);

	return kernel;
}

/* --------------------------------------------------------------------------------
 * Reads a file (filename) and stores output in text.
 * --------------------------------------------------------------------------------
 */
void readfile(char *filename, char **text, unsigned long *size) {
	FILE *fp;
	char ch;

	fp = fopen(filename, "r");

	if (fp == NULL) {
		printf("Cannot open file %s\n", filename);

	    char cwdout[2048];
	    if (getcwd(cwdout, sizeof(cwdout)) != NULL){
	    	fprintf(stdout, "Current working directory is: %s\n", cwdout);
	    }

		exit(0);
	}
	if (fseek(fp, 0, SEEK_END) == 0) {
		*size = ftell(fp);
		fseek(fp, 0, SEEK_SET);
	}
	*text = (char*) malloc(sizeof(char) * (*size));
	int i = 0;
	while ((ch = fgetc(fp)) != EOF) {
		(*text)[i] = ch;
		i++;
	}
	fclose(fp);
}
