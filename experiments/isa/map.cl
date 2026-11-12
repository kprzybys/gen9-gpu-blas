// Mapa instrukcji: każdy kernel testuje jedną kategorię operacji.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void f32_mad(__global float *o, float a, float b) { float x = o[get_global_id(0)]; x = mad(x,a,b); o[get_global_id(0)] = x; }
__kernel void f32_fma(__global float *o, float a, float b) { float x = o[get_global_id(0)]; x = fma(x,a,b); o[get_global_id(0)] = x; }
__kernel void f16_mad(__global half *o, half a, half b) { half x = o[get_global_id(0)]; x = mad(x,a,b); o[get_global_id(0)] = x; }
__kernel void f64_mad(__global double *o, double a, double b) { double x = o[get_global_id(0)]; x = mad(x,a,b); o[get_global_id(0)] = x; }
__kernel void i32_ops(__global int *o, int a, int b) { int x = o[get_global_id(0)]; x = x*a + b; x ^= x >> 3; o[get_global_id(0)] = x; }
__kernel void i32_mad24(__global int *o, int a, int b) { int x = o[get_global_id(0)]; o[get_global_id(0)] = mad24(x,a,b); }
__kernel void i32_mulhi(__global int *o, int a) { int x = o[get_global_id(0)]; o[get_global_id(0)] = mul_hi(x,a); }
__kernel void i16_ops(__global short *o, short a, short b) { short x = o[get_global_id(0)]; x = x*a + b; o[get_global_id(0)] = x; }
__kernel void i8_dot(__global char4 *v, __global int *o, char4 w) { char4 x = v[get_global_id(0)]; o[get_global_id(0)] = x.x*w.x + x.y*w.y + x.z*w.z + x.w*w.w; }
__kernel void i64_ops(__global long *o, long a) { long x = o[get_global_id(0)]; o[get_global_id(0)] = x*a + (x>>7); }
__kernel void div_f32(__global float *o, float a) { o[get_global_id(0)] = o[get_global_id(0)] / a; }
__kernel void div_native(__global float *o, float a) { o[get_global_id(0)] = native_divide(o[get_global_id(0)], a); }
__kernel void sqrt_f32(__global float *o) { o[get_global_id(0)] = sqrt(o[get_global_id(0)]); }
__kernel void rsqrt_native(__global float *o) { o[get_global_id(0)] = native_rsqrt(o[get_global_id(0)]); }
__kernel void sin_f32(__global float *o) { o[get_global_id(0)] = sin(o[get_global_id(0)]); }
__kernel void sin_native(__global float *o) { o[get_global_id(0)] = native_sin(o[get_global_id(0)]); }
__kernel void exp_native(__global float *o) { o[get_global_id(0)] = native_exp2(o[get_global_id(0)]); }
__kernel void log_native(__global float *o) { o[get_global_id(0)] = native_log2(o[get_global_id(0)]); }
__kernel void sqrt_f64(__global double *o) { o[get_global_id(0)] = sqrt(o[get_global_id(0)]); }
__kernel void slm_barrier(__global float *o) { __local float t[256]; int l = get_local_id(0); t[l] = o[get_global_id(0)]; barrier(CLK_LOCAL_MEM_FENCE); o[get_global_id(0)] = t[255-l]; }
__kernel void atomic_global(__global int *c) { atomic_add(c, 1); }
__kernel void atomic_local(__global int *c) { __local int s; if (get_local_id(0)==0) s=0; barrier(CLK_LOCAL_MEM_FENCE); atomic_add(&s,1); barrier(CLK_LOCAL_MEM_FENCE); if (get_local_id(0)==0) c[get_group_id(0)]=s; }
__kernel void atomic_fmax(__global float *c, float v) { atomic_max((volatile __global int*)c, as_int(v)); }
__kernel void sg_shuffle(__global float *o) { float x = o[get_global_id(0)]; o[get_global_id(0)] = sub_group_shuffle(x, (get_sub_group_local_id()+1) % get_max_sub_group_size()); }
__kernel void sg_reduce(__global float *o) { float x = o[get_global_id(0)]; o[get_global_id(0)] = sub_group_reduce_add(x); }
__kernel void cvt_f16_f32(__global half *i, __global float *o) { o[get_global_id(0)] = vload_half(get_global_id(0), i); }
__kernel void branch_div(__global float *o) { float x = o[get_global_id(0)]; if (x > 0.5f) x = x*x; else x = x+1.0f; o[get_global_id(0)] = x; }
__kernel void img_sample(__read_only image2d_t im, sampler_t s, __global float4 *o) { int i = get_global_id(0); o[i] = read_imagef(im, s, (float2)(i*0.01f, 0.5f)); }
