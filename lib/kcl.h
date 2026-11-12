// kcl: podzbiór OpenCL używany przez gen9blas.c, zaimplementowany bezpośrednio na i915 (bez NEO). Opis: ../BEZNEO.md.
// Typy i stałe z nagłówków Khronosa; funkcje przemianowane na kcl_* (biblioteka nie eksportuje symboli cl*, więc
// nie koliduje z prawdziwym OpenCL w tym samym procesie).
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

// Zamiast clCreateProgramWithSource/Binary + clBuildProgram: program wbudowany (kernels.c), wybierany po nazwie
// pliku i opcjach kompilacji (opcje = NULL: binarka z asemblera). Zwraca CL_SUCCESS albo CL_INVALID_PROGRAM.
cl_int kcl_program(const char *plik, const char *opcje, cl_program *wynik);
// Zwykła pamięć procesu [ptr, ptr+n) dla jąder bez kopii (userptr i915, obiekty trzymane między wywołaniami; zapis:
// GPU pisze do obszaru). Zwraca cl_mem dla clSetKernelArg (ważny do ustawienia argumentu w bieżącej operacji) albo
// NULL (wtedy kopia przez CPU). KCL_USERPTR_MB: limit przypiętej pamięci (domyślnie 1024).
cl_mem kcl_userptr(const void *ptr, size_t n, int zapis);
// Początek operacji (funkcji BLAS): obszary jednej operacji ze wspólnymi stronami idą przez jeden obiekt userptr.
void kcl_new_operation(void);
// Wysłanie bieżącego bufora komend i czekanie na koniec całej pracy (śpi w jądrze Linuksa, GEM_WAIT).
cl_int kcl_wait(void);
