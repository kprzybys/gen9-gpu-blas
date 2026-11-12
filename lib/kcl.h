// kcl: the OpenCL subset used by gen9blas.c, implemented directly on i915 (without NEO). Description: ../docs/no-neo.md.
// Types and constants come from the Khronos headers; functions are renamed to kcl_* (the library does not export cl* symbols, so
// it does not clash with a real OpenCL in the same process).
#pragma once
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <CL/cl_ext.h>

#define clGetPlatformIDs kcl_GetPlatformIDs
#define clGetDeviceIDs kcl_GetDeviceIDs
#define clCreateContext kcl_CreateContext
#define clCreateCommandQueueWithProperties kcl_CreateCommandQueueWithProperties
#define clCreateBuffer kcl_CreateBuffer
#define clCreateSubBuffer kcl_CreateSubBuffer
#define clReleaseMemObject kcl_ReleaseMemObject
#define clGetMemObjectInfo kcl_GetMemObjectInfo
#define clCreateKernel kcl_CreateKernel
#define clSetKernelArg kcl_SetKernelArg
#define clEnqueueNDRangeKernel kcl_EnqueueNDRangeKernel
#define clEnqueueMarkerWithWaitList kcl_EnqueueMarkerWithWaitList
#define clFlush kcl_Flush
#define clFinish kcl_Finish
#define clWaitForEvents kcl_WaitForEvents
#define clGetEventInfo kcl_GetEventInfo
#define clGetEventProfilingInfo kcl_GetEventProfilingInfo
#define clReleaseEvent kcl_ReleaseEvent
#define clGetExtensionFunctionAddressForPlatform kcl_GetExtensionFunctionAddressForPlatform

cl_int kcl_GetPlatformIDs(cl_uint, cl_platform_id *, cl_uint *);
cl_int kcl_GetDeviceIDs(cl_platform_id, cl_device_type, cl_uint, cl_device_id *, cl_uint *);
cl_context kcl_CreateContext(const cl_context_properties *, cl_uint, const cl_device_id *, void *, void *, cl_int *);
cl_command_queue kcl_CreateCommandQueueWithProperties(cl_context, cl_device_id, const cl_queue_properties *, cl_int *);
cl_mem kcl_CreateBuffer(cl_context, cl_mem_flags, size_t, void *, cl_int *);
cl_mem kcl_CreateSubBuffer(cl_mem, cl_mem_flags, cl_buffer_create_type, const void *, cl_int *);
cl_int kcl_ReleaseMemObject(cl_mem);
cl_int kcl_GetMemObjectInfo(cl_mem, cl_mem_info, size_t, void *, size_t *);
cl_kernel kcl_CreateKernel(cl_program, const char *, cl_int *);
cl_int kcl_SetKernelArg(cl_kernel, cl_uint, size_t, const void *);
cl_int kcl_EnqueueNDRangeKernel(cl_command_queue, cl_kernel, cl_uint, const size_t *, const size_t *, const size_t *,
                                cl_uint, const cl_event *, cl_event *);
cl_int kcl_EnqueueMarkerWithWaitList(cl_command_queue, cl_uint, const cl_event *, cl_event *);
cl_int kcl_Flush(cl_command_queue);
cl_int kcl_Finish(cl_command_queue);
cl_int kcl_WaitForEvents(cl_uint, const cl_event *);
cl_int kcl_GetEventInfo(cl_event, cl_event_info, size_t, void *, size_t *);
cl_int kcl_GetEventProfilingInfo(cl_event, cl_profiling_info, size_t, void *, size_t *);
cl_int kcl_ReleaseEvent(cl_event);
void *kcl_GetExtensionFunctionAddressForPlatform(cl_platform_id, const char *);

// Instead of clCreateProgramWithSource/Binary + clBuildProgram: a built-in program (kernels.c), selected by file
// name and build options (options = NULL: binary from the assembler). Returns CL_SUCCESS or CL_INVALID_PROGRAM.
cl_int kcl_program(const char *file, const char *options, cl_program *out);
// Ordinary process memory [ptr, ptr+n) for zero-copy kernels (i915 userptr, objects kept between calls; wr: the
// GPU writes to the region). Returns a cl_mem for clSetKernelArg (valid until the argument is set in the current operation) or
// NULL (then copy through the CPU). KCL_USERPTR_MB: limit of pinned memory (default 1024).
cl_mem kcl_userptr(const void *ptr, size_t n, int wr);
// Start of an operation (BLAS function): regions of one operation that share pages go through one userptr object.
void kcl_new_operation(void);
// Submit the current batch buffer and wait for all work to finish (sleeps in the Linux kernel, GEM_WAIT).
cl_int kcl_wait(void);
