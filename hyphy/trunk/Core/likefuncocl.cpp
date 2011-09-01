// *********************************************************************
// OpenCL likelihood function Notes:  
//
// Runs computations with OpenCL on the GPU device and then checks results 
// against basic host CPU/C++ computation.
// 
//
// *********************************************************************

#ifdef MDSOCL

#include <stdio.h>
#include <assert.h>
#include <sys/sysctl.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include "calcnode.h"

//#define FLOAT
//#define OCLVERBOSE

#if defined(__APPLE__)
#include <OpenCL/OpenCL.h>
typedef float fpoint;
typedef cl_float clfp;
//#define FLOATPREC "typedef float fpoint; \n"
//#define PRAGMADEF "#pragma OPENCL EXTENSION cl_khr_fp64: enable \n"
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#elif defined(NVIDIA)
#define __OCLPOSIX__
#include <oclUtils.h>
typedef double fpoint;
typedef cl_double clfp;
#define FLOATPREC "typedef double fpoint; \n"
#define PRAGMADEF "#pragma OPENCL EXTENSION cl_khr_fp64: enable \n"
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#elif defined(AMD)
#define __OCLPOSIX__
#include <CL/opencl.h>
typedef double fpoint;
typedef cl_double clfp;
#define FLOATPREC "typedef double fpoint; \n"
#define PRAGMADEF "#pragma OPENCL EXTENSION cl_amd_fp64: enable \n"
//#pragma OPENCL EXTENSION cl_amd_fp64: enable
#elif defined(FLOAT)
#include <CL/opencl.h>
typedef float fpoint;
typedef cl_float clfp;
#define FLOATPREC "typedef float fpoint; \n"
#define PRAGMADEF " \n"
#endif

//#define __VERBOSE__
//#define OCLGPU
#ifdef OCLGPU
#define OCLTARGET " #define BLOCK_SIZE 16 \n" 
#else
#define OCLTARGET " #define BLOCK_SIZE 1 \n"
#endif

#define MIN(a,b) ((a)>(b)?(b):(a))

// time stuff:
#define BILLION 1E9
struct timespec mainStart, mainEnd, bufferStart, bufferEnd, queueStart, queueEnd, setupStart, setupEnd;
double mainSecs;
double buffSecs;
double queueSecs;
double setupSecs;

bool clean;

cl_context cxGPUContext;        // OpenCL context
cl_command_queue cqCommandQueue;// OpenCL command que
cl_platform_id cpPlatform;      // OpenCL platform
cl_device_id cdDevice;          // OpenCL device
cl_program cpMLProgram;
cl_program cpLeafProgram;
cl_program cpInternalProgram;
cl_program cpAmbigProgram;
cl_program cpResultProgram;
cl_kernel ckLeafKernel;
cl_kernel ckInternalKernel;
cl_kernel ckAmbigKernel;
cl_kernel ckResultKernel;
size_t szGlobalWorkSize[2];        // 1D var for Total # of work items
size_t szLocalWorkSize[2];         // 1D var for # of work items in the work group 
size_t localMemorySize;         // size of local memory buffer for kernel scratch
size_t szParmDataBytes;         // Byte size of context information
size_t szKernelLength;          // Byte size of kernel code
cl_int ciErr1, ciErr2;          // Error code var

cl_mem cmNode_cache;
cl_mem cmModel_cache;
cl_mem cmNodRes_cache;
cl_mem cmNodFlag_cache;
cl_mem cmroot_cache;
cl_mem cmroot_scalings;
cl_mem cmScalings_cache;
cl_mem cmFreq_cache;
cl_mem cmProb_cache;
cl_mem cmResult_cache;
long siteCount, alphabetDimension; 
long* lNodeFlags;
_SimpleList     updateNodes, 
                flatParents,
                flatNodes,
                flatCLeaves,
                flatLeaves,
                flatTree,
                theFrequencies;
_Parameter      *iNodeCache,
                *theProbs;
_SimpleList taggedInternals;
_GrowingVector* lNodeResolutions;
float scalar;

void *node_cache, *nodRes_cache, *nodFlag_cache, *scalings_cache, *prob_cache, *freq_cache, *root_cache, *result_cache, *root_scalings, *model;

void _OCLEvaluator::init(   long esiteCount,
                                    long ealphabetDimension,
                                    _Parameter* eiNodeCache)
{
    clean = false;
    contextSet = false;
    siteCount = esiteCount;
    alphabetDimension = ealphabetDimension;
    iNodeCache = eiNodeCache;
    mainSecs = 0.0;
    buffSecs = 0.0;
    queueSecs = 0.0;
    setupSecs = 0.0;
    scalar = 10.0;
}

// So the two interfacing functions will be the constructor, called in SetupLFCaches, and launchmdsocl, called in ComputeBlock.
// Therefore all of these functions need to be finished, the context needs to be setup separately from the execution, the data needs 
// to be passed piecewise, and a pointer needs to be passed around in likefunc2.cpp. After that things should be going a bit faster, 
// though honestly this solution is geared towards analyses with a larger number of sites. 

// *********************************************************************
int _OCLEvaluator::setupContext(void)
{
#ifdef __OCLPOSIX__
    clock_gettime(CLOCK_MONOTONIC, &setupStart);
#endif 
    //printf("Made it to the oclmain() function!\n");

    //long nodeResCount = sizeof(lNodeResolutions->theData)/sizeof(lNodeResolutions->theData[0]);
    long nodeFlagCount = flatLeaves.lLength*siteCount;
    long nodeResCount = lNodeResolutions->GetUsed();
    int roundCharacters = roundUpToNextPowerOfTwo(alphabetDimension);
//    long nodeCount = flatLeaves.lLength + flatNodes.lLength + 1;
//    long iNodeCount = flatNodes.lLength + 1;

    bool ambiguousNodes = true;
    if (nodeResCount == 0) 
    {
        nodeResCount++;
        ambiguousNodes = false;
    }

    //printf("Got the sizes of nodeRes and nodeFlag: %i, %i\n", nodeResCount, nodeFlagCount);

    // Make transitionMatrixArray, do other host stuff:
    node_cache      = (void*)malloc(sizeof(cl_float)*roundCharacters*siteCount*(flatNodes.lLength)); 
    nodRes_cache    = (void*)malloc(sizeof(cl_float)*roundUpToNextPowerOfTwo(nodeResCount));
    nodFlag_cache   = (void*)malloc(sizeof(cl_long)*roundUpToNextPowerOfTwo(nodeFlagCount));
    scalings_cache  = (void*)malloc(sizeof(cl_int)*roundCharacters*siteCount*(flatNodes.lLength));
    prob_cache      = (void*)malloc(sizeof(cl_float)*roundCharacters);
    freq_cache      = (void*)malloc(sizeof(cl_int)*siteCount);
    result_cache    = (void*)malloc(sizeof(cl_float)*siteCount);
    freq_cache      = (void*)malloc(sizeof(cl_int)*siteCount);
    root_cache      = (void*)malloc(sizeof(cl_float)*siteCount*roundCharacters);
    root_scalings   = (void*)malloc(sizeof(cl_int)*siteCount*roundCharacters);
    result_cache    = (void*)malloc(sizeof(cl_float)*siteCount);
    model           = (void*)malloc(sizeof(cl_float)*roundCharacters*roundCharacters*(flatParents.lLength-1));

    //printf("Allocated all of the arrays!\n");
    //printf("setup the model, fixed tagged internals!\n");
    printf("flatleaves: %i\n", flatLeaves.lLength);
    printf("flatParents: %i\n", flatParents.lLength);
    //printf("flatCleaves: %i\n", flatCLeaves.lLength);
    printf("flatNodes: %i\n", flatNodes.lLength);
    printf("updateNodes: %i\n", updateNodes.lLength);
    printf("flatTree: %i\n", flatTree.lLength);
    //printf("nodeFlagCount: %i\n", nodeFlagCount);
    //printf("nodeResCount: %i\n", nodeResCount);

    //for (int i = 0; i < nodeCount*siteCount*alphabetDimension; i++)
    printf("siteCount: %i, alphabetDimension: %i \n", siteCount, alphabetDimension);
    //printf("NodeCache: \n");
/*
    int alphaI = 0;
    for (long i = 0; i < (flatNodes.lLength)*roundCharacters*siteCount; i++)
    {
        ((int*)scalings_cache)[i] = 0.;
        if (i%(roundCharacters) < alphabetDimension)
        {
            ((float*)node_cache)[i] = (float)iNodeCache[alphaI];
            alphaI++;
            //printf("%g ", (float)iNodeCache[alphaI]);
        }
        else ((float*)node_cache)[i] = 0.0;
//      double t = iNodeCache[i];        
//      if (i%(siteCount*alphabetDimension) == 0)
//            printf("Got another one %g\n",t);
        //printf ("%i\n",i);
    }
        //printf("\n");
    //printf("Node resolutions: \n");
*/
    if (ambiguousNodes)
        for (int i = 0; i < nodeResCount; i++)
            ((float*)nodRes_cache)[i] = (float)(lNodeResolutions->theData[i]);
    for (int i = 0; i < nodeFlagCount; i++)
        ((long*)nodFlag_cache)[i] = lNodeFlags[i];
    for (int i = 0; i < siteCount; i++)
        ((int*)freq_cache)[i] = theFrequencies[i];
    for (int i = 0; i < alphabetDimension; i++)
        ((float*)prob_cache)[i] = theProbs[i];

    //printf("Created all of the arrays!\n");

    // alright, by now taggedInternals have been taken care of, and model has
    // been filled with all of the transition matrices. 

#ifdef __OCLPOSIX__
    clock_gettime(CLOCK_MONOTONIC, &setupEnd);
    setupSecs += (setupEnd.tv_sec - setupStart.tv_sec)+(setupEnd.tv_nsec - setupStart.tv_nsec)/BILLION;
#endif

    
    
    //**************************************************
    
    //Get an OpenCL platform
    ciErr1 = clGetPlatformIDs(1, &cpPlatform, NULL);
    
//    printf("clGetPlatformID...\n"); 
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error in clGetPlatformID, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(EXIT_FAILURE);
    }
    
    
    //Get the devices
#ifdef OCLGPU
    ciErr1 = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &cdDevice, NULL);
#else
    ciErr1 = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_CPU, 1, &cdDevice, NULL);
#endif
 //   printf("clGetDeviceIDs...\n"); 
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error in clGetDeviceIDs, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(EXIT_FAILURE);
    }
    
    size_t maxWorkGroupSize;
    ciErr1 = clGetDeviceInfo(cdDevice, CL_DEVICE_MAX_WORK_GROUP_SIZE, 
                             sizeof(size_t), &maxWorkGroupSize, NULL);
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Getting max work group size failed!\n");
    }
    printf("Max work group size: %lu\n", (unsigned long)maxWorkGroupSize);

    size_t maxLocalSize;
    ciErr1 = clGetDeviceInfo(cdDevice, CL_DEVICE_LOCAL_MEM_SIZE, 
                             sizeof(size_t), &maxLocalSize, NULL);
    size_t maxConstSize;
    ciErr1 = clGetDeviceInfo(cdDevice, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, 
                             sizeof(size_t), &maxConstSize, NULL);
    printf("LocalSize: %i, Const size: %i\n", maxLocalSize, maxConstSize);

    printf("sites: %d\n", siteCount);
    
    // set and log Global and Local work size dimensions
    
#ifdef OCLGPU
    szLocalWorkSize[0] = 16; // All of these will have to be generalized. 
    szLocalWorkSize[1] = 16;
#else
    szLocalWorkSize[0] = 1; // All of these will have to be generalized. 
    szLocalWorkSize[1] = 1;
#endif
    szGlobalWorkSize[0] = 64;
    szGlobalWorkSize[1] = ((siteCount + 16)/16)*16;
    //szGlobalWorkSize[1] = roundUpToNextPowerOfTwo(siteCount);
    printf("Global Work Size \t\t= %d, %d\nLocal Work Size \t\t= %d, %d\n# of Work Groups \t\t= %d\n\n", 
           szGlobalWorkSize[0], szGlobalWorkSize[1], szLocalWorkSize[0], szLocalWorkSize[1], 
           ((szGlobalWorkSize[0]*szGlobalWorkSize[1])/(szLocalWorkSize[0]*szLocalWorkSize[1]))); 

    
    size_t returned_size = 0;
    cl_char vendor_name[1024] = {0};
    cl_char device_name[1024] = {0};
    ciErr1 = clGetDeviceInfo(cdDevice, CL_DEVICE_VENDOR, sizeof(vendor_name), 
                             vendor_name, &returned_size);
    ciErr1 |= clGetDeviceInfo(cdDevice, CL_DEVICE_NAME, sizeof(device_name), 
                              device_name, &returned_size);
    assert(ciErr1 == CL_SUCCESS);
//    printf("Connecting to %s %s...\n", vendor_name, device_name);
    
    //Create the context
    cxGPUContext = clCreateContext(0, 1, &cdDevice, NULL, NULL, &ciErr1);
//    printf("clCreateContext...\n"); 
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error in clCreateContext, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(EXIT_FAILURE);
    }
    
    // Create a command-queue
    cqCommandQueue = clCreateCommandQueue(cxGPUContext, cdDevice, 0, &ciErr1);
//    printf("clCreateCommandQueue...\n"); 
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error in clCreateCommandQueue, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(EXIT_FAILURE);
    }


    printf("Setup all of the OpenCL stuff!\n");

    // Allocate the OpenCL buffer memory objects for the input and output on the
    // device GMEM
    cmNode_cache = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                    sizeof(cl_float)*roundCharacters*siteCount*(flatNodes.lLength), node_cache,
                    &ciErr1);
    cmModel_cache = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY,
                    sizeof(cl_float)*roundCharacters*roundCharacters*(flatParents.lLength-1), 
                    NULL, &ciErr2);
    ciErr1 |= ciErr2;
    cmScalings_cache = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                    sizeof(cl_int)*roundCharacters*siteCount*flatNodes.lLength, scalings_cache, &ciErr2);
    ciErr1 |= ciErr2;
    cmNodRes_cache = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                    sizeof(cl_float)*roundUpToNextPowerOfTwo(nodeResCount), nodRes_cache, &ciErr2);
    ciErr1 |= ciErr2;
    cmNodFlag_cache = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                    sizeof(cl_long)*roundUpToNextPowerOfTwo(nodeFlagCount), nodFlag_cache, &ciErr2);
    ciErr1 |= ciErr2;
    cmroot_cache = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE,
                    sizeof(cl_float)*siteCount*roundCharacters, NULL, &ciErr2);
    ciErr1 |= ciErr2;
    cmroot_scalings = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE,
                    sizeof(cl_int)*siteCount*roundCharacters, NULL, &ciErr2);
    ciErr1 |= ciErr2;
    cmProb_cache = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                    sizeof(cl_float)*roundCharacters, prob_cache, &ciErr2);
    ciErr1 |= ciErr2;
    cmFreq_cache = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY,
                    sizeof(cl_float)*siteCount, NULL, &ciErr2);
    ciErr1 |= ciErr2;
    cmResult_cache = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY,
                    sizeof(cl_float)*siteCount, NULL, &ciErr2);
    ciErr1 |= ciErr2;
//    printf("clCreateBuffer...\n");
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error in clCreateBuffer, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        switch(ciErr1)
        {
            case   CL_INVALID_CONTEXT: printf("CL_INVALID_CONTEXT\n"); break;
            case   CL_INVALID_VALUE: printf("CL_INVALID_VALUE\n"); break;
            case   CL_INVALID_BUFFER_SIZE: printf("CL_INVALID_BUFFER_SIZE\n"); break;
            case   CL_MEM_OBJECT_ALLOCATION_FAILURE: printf("CL_MEM_OBJECT_ALLOCATION_FAILURE\n"); break; 
            case   CL_OUT_OF_HOST_MEMORY: printf("CL_OUT_OF_HOST_MEMORY\n"); break;
            default: printf("Strange error\n"); 
        }
        Cleanup(EXIT_FAILURE);
    }


#ifdef __OCLPOSIX__
    clock_gettime(CLOCK_MONOTONIC, &setupStart);
#endif
/*
    for (int i = 0; i < siteCount*roundCharacters; i++)
    {
        (root_cache)[i] = 0.0;
        (root_scalings)[i] = 1;
    }
*/
#ifdef __OCLPOSIX__
    clock_gettime(CLOCK_MONOTONIC, &setupEnd);
    setupSecs += (setupEnd.tv_sec - setupStart.tv_sec)+(setupEnd.tv_nsec - setupStart.tv_nsec)/BILLION;
#endif

    printf("Made all of the buffers on the device!\n");
    
//    printf("clCreateBuffer...\n");
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error in clCreateBuffer, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(EXIT_FAILURE);
    }
    
//  "" FLOATPREC                                                                                                                        \
    // Create the program
    const char *program_source = "\n" \
    "" OCLTARGET                                                                                                                        \
    " #define MIN(a,b) ((a)>(b)?(b):(a))                                                                                            \n" \
    "__kernel void LeafKernel(  __global float* node_cache,                 // argument 0                                           \n" \
    "                           __global const float* model,                // argument 1                                           \n" \
    "                           __global const float* nodRes_cache,         // argument 2                                           \n" \
    "                           __constant long* nodFlag_cache,             // argument 3                                           \n" \
    "                           long sites,                                 // argument 4                                           \n" \
    "                           long characters,                            // argument 5                                           \n" \
    "                           long childNodeIndex,                        // argument 6                                           \n" \
    "                           long parentNodeIndex,                       // argument 7                                           \n" \
    "                           long roundCharacters,                       // argument 8                                           \n" \
    "                           int intTagState,                            // argument 9                                           \n" \
    "                           int nodeID,                                 // argument 10                                          \n" \
    "                           __global int* scalings,                     // argument 11                                          \n" \
    "                           float scalar,                               // argument 12                                          \n" \
    "                           float uFlowThresh                           // argument 13                                          \n" \
    "                           )                                                                                                   \n" \
    "{                                                                                                                              \n" \
    "   int gx = get_global_id(0); // pchar                                                                                         \n" \
    "   if (gx > characters) return;                                                                                                \n" \
    "   int gy = get_global_id(1); // site                                                                                          \n" \
    "   if (gy > sites) return;                                                                                                     \n" \
    "   long parentCharacterIndex = parentNodeIndex*sites*roundCharacters + gy*roundCharacters + gx;                                \n" \
    "   float privateParentScratch = 1.0f;                                                                                          \n" \
    "   int scale = 0;                                                                                                              \n" \
    "   if (intTagState == 1)                                                                                                       \n" \
    "   {                                                                                                                           \n" \
    "       privateParentScratch = node_cache[parentCharacterIndex];                                                                \n" \
    "       scale = scalings[parentCharacterIndex];                                                                                 \n" \
    "   }                                                                                                                           \n" \
    "   long siteState = nodFlag_cache[childNodeIndex*sites + gy];                                                                  \n" \
    "   privateParentScratch *= model[nodeID*roundCharacters*roundCharacters + siteState*roundCharacters + gx];                     \n" \
    "   if (gy < sites && gx < characters)                                                                                          \n" \
    "   {                                                                                                                           \n" \
    "       node_cache[parentCharacterIndex] = privateParentScratch;                                                                \n" \
    "       scalings[parentCharacterIndex] = scale;                                                                                 \n" \
    "   }                                                                                                                           \n" \
    "}                                                                                                                              \n" \
    "__kernel void AmbigKernel(     __global float* node_cache,                 // argument 0                                       \n" \
    "                               __global const float* model,                // argument 1                                       \n" \
    "                               __global const float* nodRes_cache,         // argument 2                                       \n" \
    "                               __constant long* nodFlag_cache,             // argument 3                                       \n" \
    "                               long sites,                                 // argument 4                                       \n" \
    "                               long characters,                            // argument 5                                       \n" \
    "                               long childNodeIndex,                        // argument 6                                       \n" \
    "                               long parentNodeIndex,                       // argument 7                                       \n" \
    "                               long roundCharacters,                       // argument 8                                       \n" \
    "                               int intTagState,                            // argument 9                                       \n" \
    "                               int nodeID,                                 // argument 10                                      \n" \
    "                               __global int* scalings,                     // argument 11                                      \n" \
    "                               float scalar,                               // argument 12                                      \n" \
    "                               float uFlowThresh                           // argument 13                                      \n" \
    "                               )                                                                                               \n" \
    "{                                                                                                                              \n" \
    "   // block index                                                                                                              \n" \
    "   int bx = get_group_id(0);                                                                                                   \n" \
    "   int by = get_group_id(1);                                                                                                   \n" \
    "   // thread index                                                                                                             \n" \
    "   int tx = get_local_id(0);                                                                                                   \n" \
    "   int ty = get_local_id(1);                                                                                                   \n" \
    "   // global index                                                                                                             \n" \
    "   int gx = get_global_id(0);                                                                                                  \n" \
    "   int gy = get_global_id(1);                                                                                                  \n" \
    "   long parentCharacterIndex = parentNodeIndex*sites*roundCharacters + gy*roundCharacters + gx;                                \n" \
    "   float privateParentScratch = 1.0f;                                                                                          \n" \
    "   int scale = 0;                                                                                                              \n" \
    "   if (intTagState == 1 && gy < sites && gx < characters)                                                                      \n" \
    "   {                                                                                                                           \n" \
    "       privateParentScratch = node_cache[parentCharacterIndex];                                                                \n" \
    "       scale = scalings[parentCharacterIndex];                                                                                 \n" \
    "   }                                                                                                                           \n" \
    "   float sum = 0.f;                                                                                                            \n" \
    "   float childSum = 0.f;                                                                                                       \n" \
    "   int scaleScratch = 0;                                                                                                       \n" \
    "   __local float childScratch[BLOCK_SIZE][BLOCK_SIZE];                                                                         \n" \
    "   __local float modelScratch[BLOCK_SIZE][BLOCK_SIZE];                                                                         \n" \
    "   int siteState = nodFlag_cache[childNodeIndex*sites + gy];                                                                   \n" \
    "   int ambig = 0;                                                                                                              \n" \
    "   if (siteState < 0)                                                                                                          \n" \
    "   {                                                                                                                           \n" \
    "       ambig = 1;                                                                                                              \n" \
    "       siteState = -siteState-1;                                                                                               \n" \
    "   }                                                                                                                           \n" \
    "   int cChar = 0;                                                                                                              \n" \
    "   for (int charBlock = 0; charBlock < 64/BLOCK_SIZE; charBlock++)                                                             \n" \
    "   {                                                                                                                           \n" \
    "       if (ambig)                                                                                                              \n" \
    "           childScratch[ty][tx] =                                                                                              \n" \
    "               nodRes_cache[siteState*characters + (charBlock*BLOCK_SIZE) + tx];                                               \n" \
    "       else                                                                                                                    \n" \
    "       {                                                                                                                       \n" \
    "           if (charBlock*BLOCK_SIZE + tx == siteState)                                                                         \n" \
    "               childScratch[ty][tx] = 1;                                                                                       \n" \
    "           else                                                                                                                \n" \
    "               childScratch[ty][tx] = 0;                                                                                       \n" \
    "       }                                                                                                                       \n" \
    "       modelScratch[ty][tx] = model[nodeID*roundCharacters*roundCharacters + roundCharacters*((charBlock*BLOCK_SIZE)+ty) + gx];\n" \
    "       barrier(CLK_LOCAL_MEM_FENCE);                                                                                           \n" \
    "       for (int myChar = 0; myChar < MIN(BLOCK_SIZE, (characters-cChar)); myChar++)                                            \n" \
    "       {                                                                                                                       \n" \
    "           sum += childScratch[ty][myChar] * modelScratch[myChar][tx];                                                         \n" \
    "           childSum += childScratch[ty][myChar];                                                                               \n" \
    "       }                                                                                                                       \n" \
    "       barrier(CLK_LOCAL_MEM_FENCE);                                                                                           \n" \
    "       cChar += BLOCK_SIZE;                                                                                                    \n" \
    "   }                                                                                                                           \n" \
    "   while (childSum < 1 && childSum != 0)                                                                                       \n" \
    "   {                                                                                                                           \n" \
    "       childSum *= scalar;                                                                                                     \n" \
    "       sum *= scalar;                                                                                                          \n" \
    "       scaleScratch++;                                                                                                         \n" \
    "   }                                                                                                                           \n" \
    "   scale += scaleScratch;                                                                                                      \n" \
    "   privateParentScratch *= sum;                                                                                                \n" \
    "   if (gy < sites && gx < characters)                                                                                          \n" \
    "   {                                                                                                                           \n" \
    "       scalings    [parentCharacterIndex]  = scale;                                                                            \n" \
    "       node_cache  [parentCharacterIndex]  = privateParentScratch;                                                             \n" \
    "   }                                                                                                                           \n" \
    "}                                                                                                                              \n" \
    "__kernel void InternalKernel(  __global float* node_cache,                 // argument 0                                       \n" \
    "                               __global const float* model,                // argument 1                                       \n" \
    "                               __global const float* nodRes_cache,         // argument 2                                       \n" \
    "                               long sites,                                 // argument 3                                       \n" \
    "                               long characters,                            // argument 4                                       \n" \
    "                               long childNodeIndex,                        // argument 5                                       \n" \
    "                               long parentNodeIndex,                       // argument 6                                       \n" \
    "                               long roundCharacters,                       // argument 7                                       \n" \
    "                               int intTagState,                            // argument 8                                       \n" \
    "                               int nodeID,                                 // argument 9                                       \n" \
    "                               __global float* root_cache,                 // argument 10                                      \n" \
    "                               __global int* scalings,                     // argument 11                                      \n" \
    "                               float scalar,                               // argument 12                                      \n" \
    "                               float uFlowThresh,                          // argument 13                                      \n" \
    "                               __global int* root_scalings                 // argument 10                                      \n" \
    "                               )                                                                                               \n" \
    "{                                                                                                                              \n" \
    "   // thread index                                                                                                             \n" \
    "   int tx = get_local_id(0);   //local pchar                                                                                   \n" \
    "   int ty = get_local_id(1);   //local site                                                                                    \n" \
    "   // global index                                                                                                             \n" \
    "   int gx = get_global_id(0);                                                                                                  \n" \
    "   int gy = get_global_id(1);                                                                                                  \n" \
    "   long parentCharacterIndex = parentNodeIndex*sites*roundCharacters + gy*roundCharacters + gx;                                \n" \
    "   float privateParentScratch = 1.0f;                                                                                          \n" \
    "   int scale = 0;                                                                                                              \n" \
    "   if (intTagState == 1 && gy < sites && gx < characters)                                                                      \n" \
    "   {                                                                                                                           \n" \
    "       privateParentScratch = node_cache[parentCharacterIndex];                                                                \n" \
    "       scale = scalings[parentCharacterIndex];                                                                                 \n" \
    "   }                                                                                                                           \n" \
    "   float sum = 0.f;                                                                                                                \n" \
    "   float childSum = 0.f;                                                                                                       \n" \
    "   int scaleScratch = scalings[childNodeIndex*sites*roundCharacters + gy*roundCharacters + gx];                                \n" \
    "   __local float  childScratch[BLOCK_SIZE][BLOCK_SIZE];                                                                        \n" \
    "   __local float  modelScratch[BLOCK_SIZE][BLOCK_SIZE];                                                                        \n" \
    "   int cChar = 0;                                                                                                              \n" \
    "   for (int charBlock = 0; charBlock < 64/BLOCK_SIZE; charBlock++)                                                             \n" \
    "   {                                                                                                                           \n" \
    "       childScratch[ty][tx] =                                                                                                  \n" \
    "           node_cache[childNodeIndex*sites*roundCharacters + roundCharacters*gy + (charBlock*BLOCK_SIZE) + tx];                \n" \
    "       modelScratch[ty][tx] = model[nodeID*roundCharacters*roundCharacters + roundCharacters*((charBlock*BLOCK_SIZE)+ty) + gx];\n" \
    "       barrier(CLK_LOCAL_MEM_FENCE);                                                                                           \n" \
    "       for (int myChar = 0; myChar < MIN(BLOCK_SIZE, (characters-cChar)); myChar++)                                            \n" \
    "       {                                                                                                                       \n" \
    "           sum += childScratch[ty][myChar] * modelScratch[myChar][tx];                                                         \n" \
    "           childSum += childScratch[ty][myChar];                                                                               \n" \
    "       }                                                                                                                       \n" \
    "       barrier(CLK_LOCAL_MEM_FENCE);                                                                                           \n" \
    "       cChar += BLOCK_SIZE;                                                                                                    \n" \
    "   }                                                                                                                           \n" \
    "   while (childSum < 1 && childSum != 0)                                                                                       \n" \
    "   {                                                                                                                           \n" \
    "       childSum *= scalar;                                                                                                     \n" \
    "       sum *= scalar;                                                                                                          \n" \
    "       scaleScratch++;                                                                                                         \n" \
    "   }                                                                                                                           \n" \
    "   scale += scaleScratch;                                                                                                      \n" \
    "   privateParentScratch *= sum;                                                                                                \n" \
    "   if (gy < sites && gx < characters)                                                                                          \n" \
    "   {                                                                                                                           \n" \
    "       scalings     [parentCharacterIndex]  = scale;                                                                           \n" \
    "       root_scalings[gy*roundCharacters+gx] = scale;                                                                           \n" \
    "       node_cache   [parentCharacterIndex]  = privateParentScratch;                                                            \n" \
    "       root_cache   [gy*roundCharacters+gx] = privateParentScratch;                                                            \n" \
    "   }                                                                                                                           \n" \
    "}                                                                                                                              \n" \
    "__kernel void ResultKernel (   __global int* freq_cache,                   // argument 0                                       \n" \
    "                               __global float* prob_cache,                 // argument 1                                       \n" \
    "                               __global float* result_cache,               // argument 2                                       \n" \
    "                               __global float* root_cache,                 // argument 3                                       \n" \
    "                               __global int* root_scalings,                // argument 4                                       \n" \
    "                               long sites,                                 // argument 5                                       \n" \
    "                               long roundCharacters,                       // argument 6                                       \n" \
    "                               float scalar,                               // argument 7                                       \n" \
    "                               long characters                             // argument 8                                       \n" \
    "                           )                                                                                                   \n" \
    "{                                                                                                                              \n" \
    "   int pchar = get_global_id(0);                                                                                               \n" \
    "   int site = get_global_id(1);                                                                                                \n" \
    "   if (pchar != 0) return;                                                                                                     \n" \
    "   float acc = 0.0f;                                                                                                           \n" \
    "   int scale = root_scalings[site*roundCharacters];                                                                            \n" \
    "   for (int rChar = 0; rChar < characters; rChar++)                                                                            \n" \
    "   {                                                                                                                           \n" \
    "       acc += root_cache[site*roundCharacters + rChar] * prob_cache[rChar];                                                    \n" \
    "   }                                                                                                                           \n" \
    "   if (site < sites)                                                                                                           \n" \
    "       result_cache[site] = (native_log(acc)-scale*native_log(scalar)) * freq_cache[site];                                     \n" \
    "}                                                                                                                              \n" \
    "\n"; 
    const char *leaf_source = "\n" \
    "__kernel void LeafKernel(  __global float* node_cache,                 // argument 0                                           \n" \
    "                           __global const float* model,                // argument 1                                           \n" \
    "                           __global const float* nodRes_cache,         // argument 2                                           \n" \
    "                           __constant long* nodFlag_cache,             // argument 3                                           \n" \
    "                           long sites,                                 // argument 4                                           \n" \
    "                           long characters,                            // argument 5                                           \n" \
    "                           long childNodeIndex,                        // argument 6                                           \n" \
    "                           long parentNodeIndex,                       // argument 7                                           \n" \
    "                           long roundCharacters,                       // argument 8                                           \n" \
    "                           int intTagState,                            // argument 9                                           \n" \
    "                           int nodeID,                                 // argument 10                                          \n" \
    "                           __global int* scalings,                     // argument 11                                          \n" \
    "                           float scalar,                               // argument 12                                          \n" \
    "                           float uFlowThresh                           // argument 13                                          \n" \
    "                           )                                                                                                   \n" \
    "{                                                                                                                              \n" \
    "   int gx = get_global_id(0); // pchar                                                                                         \n" \
    "   if (gx > characters) return;                                                                                                \n" \
    "   int gy = get_global_id(1); // site                                                                                          \n" \
    "   if (gy > sites) return;                                                                                                     \n" \
    "   long parentCharacterIndex = parentNodeIndex*sites*roundCharacters + gy*roundCharacters + gx;                                \n" \
    "   float privateParentScratch = 1.0;                                                                                           \n" \
    "   int scale = 0;                                                                                                              \n" \
    "   if (intTagState == 1)                                                                                                       \n" \
    "   {                                                                                                                           \n" \
    "       privateParentScratch = node_cache[parentCharacterIndex];                                                                \n" \
    "       scale = scalings[parentCharacterIndex];                                                                                 \n" \
    "   }                                                                                                                           \n" \
    "   long siteState = nodFlag_cache[childNodeIndex*sites + gy];                                                                  \n" \
    "   privateParentScratch *= model[nodeID*roundCharacters*roundCharacters + siteState*roundCharacters + gx];                     \n" \
    "   if (gy < sites && gx < characters)                                                                                          \n" \
    "   {                                                                                                                           \n" \
    "       node_cache[parentCharacterIndex] = privateParentScratch;                                                                \n" \
    "       scalings[parentCharacterIndex] = scale;                                                                                 \n" \
    "   }                                                                                                                           \n" \
    "}                                                                                                                              \n" \
    "\n";
    const char *ambig_source = "\n" \
    "" OCLTARGET                                                                                                                        \
    " #define MIN(a,b) ((a)>(b)?(b):(a))                                                                                            \n" \
    "__kernel void AmbigKernel(     __global float* node_cache,                 // argument 0                                       \n" \
    "                               __global const float* model,                // argument 1                                       \n" \
    "                               __global const float* nodRes_cache,         // argument 2                                       \n" \
    "                               __constant long* nodFlag_cache,             // argument 3                                       \n" \
    "                               long sites,                                 // argument 4                                       \n" \
    "                               long characters,                            // argument 5                                       \n" \
    "                               long childNodeIndex,                        // argument 6                                       \n" \
    "                               long parentNodeIndex,                       // argument 7                                       \n" \
    "                               long roundCharacters,                       // argument 8                                       \n" \
    "                               int intTagState,                            // argument 9                                       \n" \
    "                               int nodeID,                                 // argument 10                                      \n" \
    "                               __global int* scalings,                     // argument 11                                      \n" \
    "                               float scalar,                               // argument 12                                      \n" \
    "                               float uFlowThresh                           // argument 13                                      \n" \
    "                               )                                                                                               \n" \
    "{                                                                                                                              \n" \
    "   // block index                                                                                                              \n" \
    "   int bx = get_group_id(0);                                                                                                   \n" \
    "   int by = get_group_id(1);                                                                                                   \n" \
    "   // thread index                                                                                                             \n" \
    "   int tx = get_local_id(0);                                                                                                   \n" \
    "   int ty = get_local_id(1);                                                                                                   \n" \
    "   // global index                                                                                                             \n" \
    "   int gx = get_global_id(0);                                                                                                  \n" \
    "   int gy = get_global_id(1);                                                                                                  \n" \
    "   long parentCharacterIndex = parentNodeIndex*sites*roundCharacters + gy*roundCharacters + gx;                                \n" \
    "   float privateParentScratch = 1.0;                                                                                           \n" \
    "   int scale = 0;                                                                                                              \n" \
    "   if (intTagState == 1 && gy < sites && gx < characters)                                                                      \n" \
    "   {                                                                                                                           \n" \
    "       privateParentScratch = node_cache[parentCharacterIndex];                                                                \n" \
    "       scale = scalings[parentCharacterIndex];                                                                                 \n" \
    "   }                                                                                                                           \n" \
    "   float sum = 0.;                                                                                                             \n" \
    "   float childSum = 0.;                                                                                                        \n" \
    "   int scaleScratch = 0;                                                                                                       \n" \
    "   __local float childScratch[BLOCK_SIZE][BLOCK_SIZE];                                                                         \n" \
    "   __local float modelScratch[BLOCK_SIZE][BLOCK_SIZE];                                                                         \n" \
    "   int siteState = nodFlag_cache[childNodeIndex*sites + gy];                                                                   \n" \
    "   int ambig = 0;                                                                                                              \n" \
    "   if (siteState < 0)                                                                                                          \n" \
    "   {                                                                                                                           \n" \
    "       ambig = 1;                                                                                                              \n" \
    "       siteState = -siteState-1;                                                                                               \n" \
    "   }                                                                                                                           \n" \
    "   int cChar = 0;                                                                                                              \n" \
    "   for (int charBlock = 0; charBlock < 64/BLOCK_SIZE; charBlock++)                                                             \n" \
    "   {                                                                                                                           \n" \
    "       if (ambig)                                                                                                              \n" \
    "           childScratch[ty][tx] =                                                                                              \n" \
    "               nodRes_cache[siteState*characters + (charBlock*BLOCK_SIZE) + tx];                                               \n" \
    "       else                                                                                                                    \n" \
    "       {                                                                                                                       \n" \
    "           if (charBlock*BLOCK_SIZE + tx == siteState)                                                                         \n" \
    "               childScratch[ty][tx] = 1;                                                                                       \n" \
    "           else                                                                                                                \n" \
    "               childScratch[ty][tx] = 0;                                                                                       \n" \
    "       }                                                                                                                       \n" \
    "       modelScratch[ty][tx] = model[nodeID*roundCharacters*roundCharacters + roundCharacters*((charBlock*BLOCK_SIZE)+ty) + gx];\n" \
    "       barrier(CLK_LOCAL_MEM_FENCE);                                                                                           \n" \
    "       for (int myChar = 0; myChar < MIN(BLOCK_SIZE, (characters-cChar)); myChar++)                                            \n" \
    "       {                                                                                                                       \n" \
    "           sum += childScratch[ty][myChar] * modelScratch[myChar][tx];                                                         \n" \
    "           childSum += childScratch[ty][myChar];                                                                               \n" \
    "       }                                                                                                                       \n" \
    "       barrier(CLK_LOCAL_MEM_FENCE);                                                                                           \n" \
    "       cChar += BLOCK_SIZE;                                                                                                    \n" \
    "   }                                                                                                                           \n" \
    "   while (childSum < 1 && childSum != 0)                                                                                       \n" \
    "   {                                                                                                                           \n" \
    "       childSum *= scalar;                                                                                                     \n" \
    "       sum *= scalar;                                                                                                          \n" \
    "       scaleScratch++;                                                                                                         \n" \
    "   }                                                                                                                           \n" \
    "   scale += scaleScratch;                                                                                                      \n" \
    "   privateParentScratch *= sum;                                                                                                \n" \
    "   if (gy < sites && gx < characters)                                                                                          \n" \
    "   {                                                                                                                           \n" \
    "       scalings    [parentCharacterIndex]  = scale;                                                                            \n" \
    "       node_cache  [parentCharacterIndex]  = privateParentScratch;                                                             \n" \
    "   }                                                                                                                           \n" \
    "}                                                                                                                              \n" \
    "\n";
    const char *internal_source = "\n" \
    "" OCLTARGET                                                                                                                        \
    " #define MIN(a,b) ((a)>(b)?(b):(a))                                                                                            \n" \
    "__kernel void InternalKernel(  __global float* node_cache,                 // argument 0                                       \n" \
    "                               __global const float* model,                // argument 1                                       \n" \
    "                               __global const float* nodRes_cache,         // argument 2                                       \n" \
    "                               long sites,                                 // argument 3                                       \n" \
    "                               long characters,                            // argument 4                                       \n" \
    "                               long childNodeIndex,                        // argument 5                                       \n" \
    "                               long parentNodeIndex,                       // argument 6                                       \n" \
    "                               long roundCharacters,                       // argument 7                                       \n" \
    "                               int intTagState,                            // argument 8                                       \n" \
    "                               int nodeID,                                 // argument 9                                       \n" \
    "                               __global float* root_cache,                 // argument 10                                      \n" \
    "                               __global int* scalings,                     // argument 11                                      \n" \
    "                               float scalar,                               // argument 12                                      \n" \
    "                               float uFlowThresh,                          // argument 13                                      \n" \
    "                               __global int* root_scalings                 // argument 10                                      \n" \
    "                               )                                                                                               \n" \
    "{                                                                                                                              \n" \
    "   // thread index                                                                                                             \n" \
    "   int tx = get_local_id(0);   //local pchar                                                                                   \n" \
    "   int ty = get_local_id(1);   //local site                                                                                    \n" \
    "   // global index                                                                                                             \n" \
    "   int gx = get_global_id(0);                                                                                                  \n" \
    "   int gy = get_global_id(1);                                                                                                  \n" \
    "   long parentCharacterIndex = parentNodeIndex*sites*roundCharacters + gy*roundCharacters + gx;                                \n" \
    "   float privateParentScratch = 1.0;                                                                                           \n" \
    "   int scale = 0;                                                                                                              \n" \
    "   if (intTagState == 1 && gy < sites && gx < characters)                                                                      \n" \
    "   {                                                                                                                           \n" \
    "       privateParentScratch = node_cache[parentCharacterIndex];                                                                \n" \
    "       scale = scalings[parentCharacterIndex];                                                                                 \n" \
    "   }                                                                                                                           \n" \
    "   float sum = 0.;                                                                                                             \n" \
    "   float childSum = 0.;                                                                                                        \n" \
    "   int scaleScratch = scalings[childNodeIndex*sites*roundCharacters + gy*roundCharacters + gx];                                \n" \
    "   __local float  childScratch[BLOCK_SIZE][BLOCK_SIZE];                                                                        \n" \
    "   __local float  modelScratch[BLOCK_SIZE][BLOCK_SIZE];                                                                        \n" \
    "   int cChar = 0;                                                                                                              \n" \
    "   for (int charBlock = 0; charBlock < 64/BLOCK_SIZE; charBlock++)                                                             \n" \
    "   {                                                                                                                           \n" \
    "       childScratch[ty][tx] =                                                                                                  \n" \
    "           node_cache[childNodeIndex*sites*roundCharacters + roundCharacters*gy + (charBlock*BLOCK_SIZE) + tx];                \n" \
    "       modelScratch[ty][tx] = model[nodeID*roundCharacters*roundCharacters + roundCharacters*((charBlock*BLOCK_SIZE)+ty) + gx];\n" \
    "       barrier(CLK_LOCAL_MEM_FENCE);                                                                                           \n" \
    "       for (int myChar = 0; myChar < MIN(BLOCK_SIZE, (characters-cChar)); myChar++)                                            \n" \
    "       {                                                                                                                       \n" \
    "           sum += childScratch[ty][myChar] * modelScratch[myChar][tx];                                                         \n" \
    "           childSum += childScratch[ty][myChar];                                                                               \n" \
    "       }                                                                                                                       \n" \
    "       barrier(CLK_LOCAL_MEM_FENCE);                                                                                           \n" \
    "       cChar += BLOCK_SIZE;                                                                                                    \n" \
    "   }                                                                                                                           \n" \
    "   while (childSum < 1 && childSum != 0)                                                                                       \n" \
    "   {                                                                                                                           \n" \
    "       childSum *= scalar;                                                                                                     \n" \
    "       sum *= scalar;                                                                                                          \n" \
    "       scaleScratch++;                                                                                                         \n" \
    "   }                                                                                                                           \n" \
    "   scale += scaleScratch;                                                                                                      \n" \
    "   privateParentScratch *= sum;                                                                                                \n" \
    "   if (gy < sites && gx < characters)                                                                                          \n" \
    "   {                                                                                                                           \n" \
    "       scalings     [parentCharacterIndex]  = scale;                                                                           \n" \
    "       root_scalings[gy*roundCharacters+gx] = scale;                                                                           \n" \
    "       node_cache   [parentCharacterIndex]  = privateParentScratch;                                                            \n" \
    "       root_cache   [gy*roundCharacters+gx] = privateParentScratch;                                                            \n" \
    "   }                                                                                                                           \n" \
    "}                                                                                                                              \n" \
    "\n";
    const char *result_source = "\n" \
    "__kernel void ResultKernel (   __global int* freq_cache,                   // argument 0                                       \n" \
    "                               __global float* prob_cache,                 // argument 1                                       \n" \
    "                               __global float* result_cache,               // argument 2                                       \n" \
    "                               __global float* root_cache,                 // argument 3                                       \n" \
    "                               __global int* root_scalings,                // argument 4                                       \n" \
    "                               long sites,                                 // argument 5                                       \n" \
    "                               long roundCharacters,                       // argument 6                                       \n" \
    "                               float scalar,                               // argument 7                                       \n" \
    "                               long characters                             // argument 8                                       \n" \
    "                           )                                                                                                   \n" \
    "{                                                                                                                              \n" \
    "   int pchar = get_global_id(0);                                                                                               \n" \
    "   int site = get_global_id(1);                                                                                                \n" \
    "   if (pchar != 0) return;                                                                                                     \n" \
    "   float acc = 0.0;                                                                                                            \n" \
    "   int scale = root_scalings[site*roundCharacters];                                                                            \n" \
    "   for (int rChar = 0; rChar < characters; rChar++)                                                                            \n" \
    "   {                                                                                                                           \n" \
    "       acc += root_cache[site*roundCharacters + rChar] * prob_cache[rChar];                                                    \n" \
    "   }                                                                                                                           \n" \
    "   if (site < sites)                                                                                                           \n" \
    "       result_cache[site] = (native_log(acc)-scale*native_log(scalar)) * freq_cache[site];                                                 \n" \
    "}                                                                                                                              \n" \
    "\n"; 
    
    cpMLProgram = clCreateProgramWithSource(cxGPUContext, 1, (const char**)&program_source,
                                          NULL, &ciErr1);
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error in clCreateProgramWithSource, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(EXIT_FAILURE);
    }
/*
    cpLeafProgram = clCreateProgramWithSource(cxGPUContext, 1, (const char**)&leaf_source,
                                            NULL, &ciErr1);
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error in clCreateProgramWithSource, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(EXIT_FAILURE);
    }

    cpInternalProgram = clCreateProgramWithSource(cxGPUContext, 1, (const char**)&internal_source,
                                            NULL, &ciErr1);
    
    //printf("clCreateProgramWithSource...\n"); 
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error in clCreateProgramWithSource, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(EXIT_FAILURE);
    }

    cpAmbigProgram = clCreateProgramWithSource(cxGPUContext, 1, (const char**)&ambig_source,
                                            NULL, &ciErr1);
    //cpAmbigProgram = clCreateProgramWithSource(cxGPUContext, 1, (const char**)&ambigOLD_source,
    //                                      NULL, &ciErr1);
    
    //printf("clCreateProgramWithSource...\n"); 
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error in clCreateProgramWithSource, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(EXIT_FAILURE);
    }

    cpResultProgram = clCreateProgramWithSource(cxGPUContext, 1, (const char**)&result_source,
                                            NULL, &ciErr1);
    
    //printf("clCreateProgramWithSource...\n"); 
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error in clCreateProgramWithSource, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(EXIT_FAILURE);
    }
*/
    
    ciErr1 = clBuildProgram(cpMLProgram, 1, &cdDevice, "-cl-mad-enable -cl-fast-relaxed-math", NULL, NULL);
    if (ciErr1 != CL_SUCCESS)
    {
        printf("%i\n", ciErr1); //prints "1"
        switch(ciErr1)
        {
            case   CL_INVALID_PROGRAM: printf("CL_INVALID_PROGRAM\n"); break;
            case   CL_INVALID_VALUE: printf("CL_INVALID_VALUE\n"); break;
            case   CL_INVALID_DEVICE: printf("CL_INVALID_DEVICE\n"); break;
            case   CL_INVALID_BINARY: printf("CL_INVALID_BINARY\n"); break; 
            case   CL_INVALID_BUILD_OPTIONS: printf("CL_INVALID_BUILD_OPTIONS\n"); break;
            case   CL_COMPILER_NOT_AVAILABLE: printf("CL_COMPILER_NOT_AVAILABLE\n"); break;
            case   CL_BUILD_PROGRAM_FAILURE: printf("CL_BUILD_PROGRAM_FAILURE\n"); break;
            case   CL_INVALID_OPERATION: printf("CL_INVALID_OPERATION\n"); break;
            case   CL_OUT_OF_HOST_MEMORY: printf("CL_OUT_OF_HOST_MEMORY\n"); break;
            default: printf("Strange error\n"); //This is printed
        }
        printf("Error in clBuildProgram, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(EXIT_FAILURE);
    }

/*
    //ciErr1 = clBuildProgram(cpLeafProgram, 1, &cdDevice, NULL, NULL, NULL);
    ciErr1 = clBuildProgram(cpLeafProgram, 1, &cdDevice, "-cl-mad-enable -cl-fast-relaxed-math", NULL, NULL);
    if (ciErr1 != CL_SUCCESS)
    {
        printf("%i\n", ciErr1); //prints "1"
        switch(ciErr1)
        {
            case   CL_INVALID_PROGRAM: printf("CL_INVALID_PROGRAM\n"); break;
            case   CL_INVALID_VALUE: printf("CL_INVALID_VALUE\n"); break;
            case   CL_INVALID_DEVICE: printf("CL_INVALID_DEVICE\n"); break;
            case   CL_INVALID_BINARY: printf("CL_INVALID_BINARY\n"); break; 
            case   CL_INVALID_BUILD_OPTIONS: printf("CL_INVALID_BUILD_OPTIONS\n"); break;
            case   CL_COMPILER_NOT_AVAILABLE: printf("CL_COMPILER_NOT_AVAILABLE\n"); break;
            case   CL_BUILD_PROGRAM_FAILURE: printf("CL_BUILD_PROGRAM_FAILURE\n"); break;
            case   CL_INVALID_OPERATION: printf("CL_INVALID_OPERATION\n"); break;
            case   CL_OUT_OF_HOST_MEMORY: printf("CL_OUT_OF_HOST_MEMORY\n"); break;
            default: printf("Strange error\n"); //This is printed
        }
        printf("Error in clBuildProgram, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(EXIT_FAILURE);
    }

    ciErr1 = clBuildProgram(cpAmbigProgram, 1, &cdDevice, "-cl-mad-enable -cl-fast-relaxed-math", NULL, NULL);
    if (ciErr1 != CL_SUCCESS)
    {
        printf("%i\n", ciErr1); //prints "1"
        switch(ciErr1)
        {
            case   CL_INVALID_PROGRAM: printf("CL_INVALID_PROGRAM\n"); break;
            case   CL_INVALID_VALUE: printf("CL_INVALID_VALUE\n"); break;
            case   CL_INVALID_DEVICE: printf("CL_INVALID_DEVICE\n"); break;
            case   CL_INVALID_BINARY: printf("CL_INVALID_BINARY\n"); break; 
            case   CL_INVALID_BUILD_OPTIONS: printf("CL_INVALID_BUILD_OPTIONS\n"); break;
            case   CL_COMPILER_NOT_AVAILABLE: printf("CL_COMPILER_NOT_AVAILABLE\n"); break;
            case   CL_BUILD_PROGRAM_FAILURE: printf("CL_BUILD_PROGRAM_FAILURE\n"); break;
            case   CL_INVALID_OPERATION: printf("CL_INVALID_OPERATION\n"); break;
            case   CL_OUT_OF_HOST_MEMORY: printf("CL_OUT_OF_HOST_MEMORY\n"); break;
            default: printf("Strange error\n"); //This is printed
        }
        printf("Error in clBuildProgram, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(EXIT_FAILURE);
    }

    //ciErr1 = clBuildProgram(cpInternalProgram, 1, &cdDevice, NULL, NULL, NULL);
    ciErr1 = clBuildProgram(cpInternalProgram, 1, &cdDevice, "-cl-mad-enable -cl-fast-relaxed-math", NULL, NULL);
    if (ciErr1 != CL_SUCCESS)
    {
        printf("%i\n", ciErr1); //prints "1"
        switch(ciErr1)
        {
            case   CL_INVALID_PROGRAM: printf("CL_INVALID_PROGRAM\n"); break;
            case   CL_INVALID_VALUE: printf("CL_INVALID_VALUE\n"); break;
            case   CL_INVALID_DEVICE: printf("CL_INVALID_DEVICE\n"); break;
            case   CL_INVALID_BINARY: printf("CL_INVALID_BINARY\n"); break; 
            case   CL_INVALID_BUILD_OPTIONS: printf("CL_INVALID_BUILD_OPTIONS\n"); break;
            case   CL_COMPILER_NOT_AVAILABLE: printf("CL_COMPILER_NOT_AVAILABLE\n"); break;
            case   CL_BUILD_PROGRAM_FAILURE: printf("CL_BUILD_PROGRAM_FAILURE\n"); break;
            case   CL_INVALID_OPERATION: printf("CL_INVALID_OPERATION\n"); break;
            case   CL_OUT_OF_HOST_MEMORY: printf("CL_OUT_OF_HOST_MEMORY\n"); break;
            default: printf("Strange error\n"); //This is printed
        }
        printf("Error in clBuildProgram, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(EXIT_FAILURE);
    }

    //ciErr1 = clBuildProgram(cpResultProgram, 1, &cdDevice, NULL, NULL, NULL);
    ciErr1 = clBuildProgram(cpResultProgram, 1, &cdDevice, "-cl-mad-enable -cl-fast-relaxed-math", NULL, NULL);
    if (ciErr1 != CL_SUCCESS)
    {
        printf("%i\n", ciErr1); //prints "1"
        switch(ciErr1)
        {
            case   CL_INVALID_PROGRAM: printf("CL_INVALID_PROGRAM\n"); break;
            case   CL_INVALID_VALUE: printf("CL_INVALID_VALUE\n"); break;
            case   CL_INVALID_DEVICE: printf("CL_INVALID_DEVICE\n"); break;
            case   CL_INVALID_BINARY: printf("CL_INVALID_BINARY\n"); break; 
            case   CL_INVALID_BUILD_OPTIONS: printf("CL_INVALID_BUILD_OPTIONS\n"); break;
            case   CL_COMPILER_NOT_AVAILABLE: printf("CL_COMPILER_NOT_AVAILABLE\n"); break;
            case   CL_BUILD_PROGRAM_FAILURE: printf("CL_BUILD_PROGRAM_FAILURE\n"); break;
            case   CL_INVALID_OPERATION: printf("CL_INVALID_OPERATION\n"); break;
            case   CL_OUT_OF_HOST_MEMORY: printf("CL_OUT_OF_HOST_MEMORY\n"); break;
            default: printf("Strange error\n"); //This is printed
        }
        printf("Error in clBuildProgram, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(EXIT_FAILURE);
    }
    //printf("clBuildProgram...\n"); 
*/
    
    // Shows the log
    char* build_log;
    size_t log_size;
    // First call to know the proper size
    clGetProgramBuildInfo(cpMLProgram, cdDevice, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
    build_log = new char[log_size+1];   
/*  clGetProgramBuildInfo(cpLeafProgram, cdDevice, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
    build_log = new char[log_size+1];   
    clGetProgramBuildInfo(cpInternalProgram, cdDevice, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
    build_log = new char[log_size+1];   
    clGetProgramBuildInfo(cpAmbigProgram, cdDevice, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
    build_log = new char[log_size+1];   
    clGetProgramBuildInfo(cpResultProgram, cdDevice, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
    build_log = new char[log_size+1];   
*/
    // Second call to get the log
    clGetProgramBuildInfo(cpMLProgram, cdDevice, CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL);
    build_log[log_size] = '\0';
/*
    clGetProgramBuildInfo(cpLeafProgram, cdDevice, CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL);
    build_log[log_size] = '\0';
    clGetProgramBuildInfo(cpInternalProgram, cdDevice, CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL);
    build_log[log_size] = '\0';
    clGetProgramBuildInfo(cpAmbigProgram, cdDevice, CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL);
    build_log[log_size] = '\0';
    clGetProgramBuildInfo(cpResultProgram, cdDevice, CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL);
    build_log[log_size] = '\0';
*/
    printf("%s", build_log);
    delete[] build_log;
    
    if (ciErr1 != CL_SUCCESS)
    {
        printf("%i\n", ciErr1); //prints "1"
        switch(ciErr1)
        {
            case   CL_INVALID_PROGRAM: printf("CL_INVALID_PROGRAM\n"); break;
            case   CL_INVALID_VALUE: printf("CL_INVALID_VALUE\n"); break;
            case   CL_INVALID_DEVICE: printf("CL_INVALID_DEVICE\n"); break;
            case   CL_INVALID_BINARY: printf("CL_INVALID_BINARY\n"); break; 
            case   CL_INVALID_BUILD_OPTIONS: printf("CL_INVALID_BUILD_OPTIONS\n"); break;
            case   CL_COMPILER_NOT_AVAILABLE: printf("CL_COMPILER_NOT_AVAILABLE\n"); break;
            case   CL_BUILD_PROGRAM_FAILURE: printf("CL_BUILD_PROGRAM_FAILURE\n"); break;
            case   CL_INVALID_OPERATION: printf("CL_INVALID_OPERATION\n"); break;
            case   CL_OUT_OF_HOST_MEMORY: printf("CL_OUT_OF_HOST_MEMORY\n"); break;
            default: printf("Strange error\n"); //This is printed
        }
        printf("Error in clBuildProgram, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(EXIT_FAILURE);
    }
    
    // Create the kernel
    //ckKernel = clCreateKernel(cpProgram, "FirstLoop", &ciErr1);
    ckLeafKernel = clCreateKernel(cpMLProgram, "LeafKernel", &ciErr1);
    printf("clCreateKernel (LeafKernel)...\n"); 
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error in clCreateKernel, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(EXIT_FAILURE);
    }
    ckAmbigKernel = clCreateKernel(cpMLProgram, "AmbigKernel", &ciErr1);
    printf("clCreateKernel (AmbigKernel)...\n"); 
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error in clCreateKernel, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(EXIT_FAILURE);
    }
    ckInternalKernel = clCreateKernel(cpMLProgram, "InternalKernel", &ciErr1);
    printf("clCreateKernel (InternalKernel)...\n"); 
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error in clCreateKernel, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(EXIT_FAILURE);
    }
    ckResultKernel = clCreateKernel(cpMLProgram, "ResultKernel", &ciErr1);
    printf("clCreateKernel (ResultKernel)...\n"); 
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error in clCreateKernel, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(EXIT_FAILURE);
    }
/*
    ckLeafKernel = clCreateKernel(cpLeafProgram, "LeafKernel", &ciErr1);
    printf("clCreateKernel (LeafKernel)...\n"); 
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error in clCreateKernel, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(EXIT_FAILURE);
    }
    ckAmbigKernel = clCreateKernel(cpAmbigProgram, "AmbigKernel", &ciErr1);
    printf("clCreateKernel (AmbigKernel)...\n"); 
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error in clCreateKernel, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(EXIT_FAILURE);
    }
    ckInternalKernel = clCreateKernel(cpInternalProgram, "InternalKernel", &ciErr1);
    printf("clCreateKernel (InternalKernel)...\n"); 
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error in clCreateKernel, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(EXIT_FAILURE);
    }
    ckResultKernel = clCreateKernel(cpResultProgram, "ResultKernel", &ciErr1);
    printf("clCreateKernel (ResultKernel)...\n"); 
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error in clCreateKernel, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(EXIT_FAILURE);
    }
*/

    size_t maxKernelSize;
    ciErr1 = clGetKernelWorkGroupInfo(ckLeafKernel, cdDevice, CL_KERNEL_WORK_GROUP_SIZE, 
                             sizeof(size_t), &maxKernelSize, NULL);
    printf("Max Leaf Kernel Work Group Size: %i \n", maxKernelSize);
    ciErr1 = clGetKernelWorkGroupInfo(ckAmbigKernel, cdDevice, CL_KERNEL_WORK_GROUP_SIZE, 
                             sizeof(size_t), &maxKernelSize, NULL);
    printf("Max Ambig Kernel Work Group Size: %i \n", maxKernelSize);
    ciErr1 = clGetKernelWorkGroupInfo(ckInternalKernel, cdDevice, CL_KERNEL_WORK_GROUP_SIZE, 
                             sizeof(size_t), &maxKernelSize, NULL);
    printf("Max Internal Kernel Work Group Size: %i \n", maxKernelSize);
    ciErr1 = clGetKernelWorkGroupInfo(ckResultKernel, cdDevice, CL_KERNEL_WORK_GROUP_SIZE, 
                             sizeof(size_t), &maxKernelSize, NULL);
    printf("Max Result Kernel Work Group Size: %i \n", maxKernelSize);
    
    long tempLeafState = 1;
    long tempSiteCount = siteCount;
    long tempCharCount = alphabetDimension;
    long tempChildNodeIndex = 0;
    long tempParentNodeIndex = 0;
    long tempRoundCharCount = roundUpToNextPowerOfTwo(alphabetDimension);
    int  tempTagIntState = 0;
    int   tempNodeID = 0;
    float tempScalar = scalar;
    float tempuFlowThresh = 0.000000001f;

    ciErr1  = clSetKernelArg(ckLeafKernel, 0, sizeof(cl_mem), (void*)&cmNode_cache);
    ciErr1 |= clSetKernelArg(ckLeafKernel, 1, sizeof(cl_mem), (void*)&cmModel_cache);
    ciErr1 |= clSetKernelArg(ckLeafKernel, 2, sizeof(cl_mem), (void*)&cmNodRes_cache);
    ciErr1 |= clSetKernelArg(ckLeafKernel, 3, sizeof(cl_mem), (void*)&cmNodFlag_cache);
    ciErr1 |= clSetKernelArg(ckLeafKernel, 4, sizeof(cl_long), (void*)&tempSiteCount);
    ciErr1 |= clSetKernelArg(ckLeafKernel, 5, sizeof(cl_long), (void*)&tempCharCount);
    ciErr1 |= clSetKernelArg(ckLeafKernel, 6, sizeof(cl_long), (void*)&tempChildNodeIndex); // reset this in the loop
    ciErr1 |= clSetKernelArg(ckLeafKernel, 7, sizeof(cl_long), (void*)&tempParentNodeIndex); // reset this in the loop
    ciErr1 |= clSetKernelArg(ckLeafKernel, 8, sizeof(cl_long), (void*)&tempRoundCharCount); 
    ciErr1 |= clSetKernelArg(ckLeafKernel, 9, sizeof(cl_int), (void*)&tempTagIntState); // reset this in the loop
    ciErr1 |= clSetKernelArg(ckLeafKernel, 10, sizeof(cl_int), (void*)&tempNodeID); // reset this in the loop
    ciErr1 |= clSetKernelArg(ckLeafKernel, 11, sizeof(cl_mem), (void*)&cmScalings_cache);
    ciErr1 |= clSetKernelArg(ckLeafKernel, 12, sizeof(cl_float), (void*)&tempScalar);
    ciErr1 |= clSetKernelArg(ckLeafKernel, 13, sizeof(cl_float), (void*)&tempuFlowThresh);

    ciErr1 |= clSetKernelArg(ckAmbigKernel, 0, sizeof(cl_mem), (void*)&cmNode_cache);
    ciErr1 |= clSetKernelArg(ckAmbigKernel, 1, sizeof(cl_mem), (void*)&cmModel_cache);
    ciErr1 |= clSetKernelArg(ckAmbigKernel, 2, sizeof(cl_mem), (void*)&cmNodRes_cache);
    ciErr1 |= clSetKernelArg(ckAmbigKernel, 3, sizeof(cl_mem), (void*)&cmNodFlag_cache);
    ciErr1 |= clSetKernelArg(ckAmbigKernel, 4, sizeof(cl_long), (void*)&tempSiteCount);
    ciErr1 |= clSetKernelArg(ckAmbigKernel, 5, sizeof(cl_long), (void*)&tempCharCount);
    ciErr1 |= clSetKernelArg(ckAmbigKernel, 6, sizeof(cl_long), (void*)&tempChildNodeIndex); // reset this in the loop
    ciErr1 |= clSetKernelArg(ckAmbigKernel, 7, sizeof(cl_long), (void*)&tempParentNodeIndex); // reset this in the loop
    ciErr1 |= clSetKernelArg(ckAmbigKernel, 8, sizeof(cl_long), (void*)&tempRoundCharCount); 
    ciErr1 |= clSetKernelArg(ckAmbigKernel, 9, sizeof(cl_int), (void*)&tempTagIntState);
    ciErr1 |= clSetKernelArg(ckAmbigKernel, 10, sizeof(cl_int), (void*)&tempNodeID); 
    ciErr1 |= clSetKernelArg(ckAmbigKernel, 11, sizeof(cl_mem), (void*)&cmScalings_cache);
    ciErr1 |= clSetKernelArg(ckAmbigKernel, 12, sizeof(cl_float), (void*)&tempScalar);
    ciErr1 |= clSetKernelArg(ckAmbigKernel, 13, sizeof(cl_float), (void*)&tempuFlowThresh);

    ciErr1 |= clSetKernelArg(ckInternalKernel, 0, sizeof(cl_mem), (void*)&cmNode_cache);
    ciErr1 |= clSetKernelArg(ckInternalKernel, 1, sizeof(cl_mem), (void*)&cmModel_cache);
    ciErr1 |= clSetKernelArg(ckInternalKernel, 2, sizeof(cl_mem), (void*)&cmNodRes_cache);
    ciErr1 |= clSetKernelArg(ckInternalKernel, 3, sizeof(cl_long), (void*)&tempSiteCount);
    ciErr1 |= clSetKernelArg(ckInternalKernel, 4, sizeof(cl_long), (void*)&tempCharCount);
    ciErr1 |= clSetKernelArg(ckInternalKernel, 5, sizeof(cl_long), (void*)&tempChildNodeIndex); // reset this in the loop
    ciErr1 |= clSetKernelArg(ckInternalKernel, 6, sizeof(cl_long), (void*)&tempParentNodeIndex); // reset this in the loop
    ciErr1 |= clSetKernelArg(ckInternalKernel, 7, sizeof(cl_long), (void*)&tempRoundCharCount); 
    ciErr1 |= clSetKernelArg(ckInternalKernel, 8, sizeof(cl_int), (void*)&tempTagIntState); // reset this in the loop
    ciErr1 |= clSetKernelArg(ckInternalKernel, 9, sizeof(cl_int), (void*)&tempNodeID); // reset this in the loop
    ciErr1 |= clSetKernelArg(ckInternalKernel, 10, sizeof(cl_mem), (void*)&cmroot_cache);
    ciErr1 |= clSetKernelArg(ckInternalKernel, 11, sizeof(cl_mem), (void*)&cmScalings_cache);
    ciErr1 |= clSetKernelArg(ckInternalKernel, 12, sizeof(cl_float), (void*)&tempScalar);
    ciErr1 |= clSetKernelArg(ckInternalKernel, 13, sizeof(cl_float), (void*)&tempuFlowThresh);
    ciErr1 |= clSetKernelArg(ckInternalKernel, 14, sizeof(cl_mem), (void*)&cmroot_scalings);

    ciErr1 |= clSetKernelArg(ckResultKernel, 0, sizeof(cl_mem), (void*)&cmFreq_cache);
    ciErr1 |= clSetKernelArg(ckResultKernel, 1, sizeof(cl_mem), (void*)&cmProb_cache);
    ciErr1 |= clSetKernelArg(ckResultKernel, 2, sizeof(cl_mem), (void*)&cmResult_cache);
    ciErr1 |= clSetKernelArg(ckResultKernel, 3, sizeof(cl_mem), (void*)&cmroot_cache);
    ciErr1 |= clSetKernelArg(ckResultKernel, 4, sizeof(cl_mem), (void*)&cmroot_scalings);
    ciErr1 |= clSetKernelArg(ckResultKernel, 5, sizeof(cl_long), (void*)&tempSiteCount);
    ciErr1 |= clSetKernelArg(ckResultKernel, 6, sizeof(cl_long), (void*)&tempRoundCharCount); 
    ciErr1 |= clSetKernelArg(ckResultKernel, 7, sizeof(cl_float), (void*)&tempScalar);
    ciErr1 |= clSetKernelArg(ckResultKernel, 8, sizeof(cl_long), (void*)&tempCharCount);


    //printf("clSetKernelArg 0 - 12...\n\n"); 
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error in clSetKernelArg, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(EXIT_FAILURE);
    }
    
    // --------------------------------------------------------
    // Start Core sequence... copy input data to GPU, compute, copy results back
    // Asynchronous write of data to GPU device
/*
    ciErr1 = clEnqueueWriteBuffer(cqCommandQueue, cmNode_cache, CL_FALSE, 0,
                sizeof(cl_float)*roundCharacters*siteCount*flatNodes.lLength, node_cache, 
                0, NULL, NULL);

    if (ciErr1 != CL_SUCCESS)
    {
        printf("%i\n", ciErr1); //prints "1"
        switch(ciErr1)
        {
            case   CL_INVALID_COMMAND_QUEUE: printf("CL_INVALID_COMMAND_QUEUE\n"); break;
            case   CL_INVALID_CONTEXT: printf("CL_INVALID_CONTEXT\n"); break;
            case   CL_INVALID_MEM_OBJECT: printf("CL_INVALID_MEM_OBJECT\n"); break;
            case   CL_INVALID_VALUE: printf("CL_INVALID_VALUE\n"); break;   
            case   CL_INVALID_EVENT_WAIT_LIST: printf("CL_INVALID_EVENT_WAIT_LIST\n"); break;
                //          case   CL_MISALIGNED_SUB_BUFFER_OFFSET: printf("CL_MISALIGNED_SUB_BUFFER_OFFSET\n"); break;
                //          case   CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST: printf("CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST\n"); break;
            case   CL_MEM_OBJECT_ALLOCATION_FAILURE: printf("CL_MEM_OBJECT_ALLOCATION_FAILURE\n"); break;
            case   CL_OUT_OF_RESOURCES: printf("CL_OUT_OF_RESOURCES\n"); break;
            case   CL_OUT_OF_HOST_MEMORY: printf("CL_OUT_OF_HOST_MEMORY\n"); break;
            default: printf("Strange error\n"); //This is printed
        }
        printf("Error in clEnqueueWriteBuffer, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(EXIT_FAILURE);
    }

    ciErr1 |= clEnqueueWriteBuffer(cqCommandQueue, cmNodRes_cache, CL_FALSE, 0,
                sizeof(clfp)*roundUpToNextPowerOfTwo(nodeResCount), nodRes_cache, 0, NULL, NULL);

    ciErr1 |= clEnqueueWriteBuffer(cqCommandQueue, cmNodFlag_cache, CL_FALSE, 0,
                sizeof(cl_long)*roundUpToNextPowerOfTwo(nodeFlagCount), nodFlag_cache, 0, NULL, NULL);
    
    for (int i = 0; i < siteCount; i++)
        ((float*)result_cache)[i] = 0.0;
    ciErr1 |= clEnqueueWriteBuffer(cqCommandQueue, cmroot_cache, CL_FALSE, 0,
                sizeof(cl_float)*siteCount*roundCharacters, root_cache, 0, NULL, NULL);
    ciErr1 |= clEnqueueWriteBuffer(cqCommandQueue, cmroot_scalings, CL_FALSE, 0,
                sizeof(cl_int)*siteCount*roundCharacters, root_scalings, 0, NULL, NULL);
    ciErr1 |= clEnqueueWriteBuffer(cqCommandQueue, cmResult_cache, CL_FALSE, 0,
                sizeof(cl_float)*siteCount, result_cache, 0, NULL, NULL);
 */   
    ciErr1 |= clEnqueueWriteBuffer(cqCommandQueue, cmFreq_cache, CL_FALSE, 0,
                sizeof(cl_int)*siteCount, freq_cache, 0, NULL, NULL);
    printf("clEnqueueWriteBuffer (root_cache, etc.)..."); 
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error in clEnqueueWriteBuffer, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(EXIT_FAILURE);
    }
    printf(" Done!\n"); 
}   

double _OCLEvaluator::oclmain(void)
{
    //printf("newLF!\n");
    //printf("LF");
    // so far this wholebuffer rebuild takes almost no time at all. Perhaps not true re:queue
    // Fix the model cache
#ifdef __OCLPOSIX__
    clock_gettime(CLOCK_MONOTONIC, &bufferStart);
#endif
    int roundCharacters = roundUpToNextPowerOfTwo(alphabetDimension);
/*
    printf("Update Nodes:");
    for (int i = 0; i < updateNodes.lLength; i++)
    {
        printf(" %i ", updateNodes.lData[i]);
    }
    printf("\n");

    printf("Tagged Internals:");
    for (int i = 0; i < taggedInternals.lLength; i++)
    {
        printf(" %i", taggedInternals.lData[i]);
    }
    printf("\n");
*/
    long nodeCode, parentCode;
    bool isLeaf;
    _Parameter* tMatrix;
    int a1, a2;
    //printf("updateNodes.lLength: %i", updateNodes.lLength);
    //#pragma omp parallel for default(none) shared(updateNodes, flatParents, flatLeaves, flatCLeaves, flatTree, alphabetDimension, model, roundCharacters) private(nodeCode, parentCode, isLeaf, tMatrix, a1, a2)
    for (int nodeID = 0; nodeID < updateNodes.lLength; nodeID++)
    {
        nodeCode = updateNodes.lData[nodeID];
        parentCode = flatParents.lData[nodeCode];

        isLeaf = nodeCode < flatLeaves.lLength;

        if (!isLeaf) nodeCode -= flatLeaves.lLength;

        tMatrix = (isLeaf? ((_CalcNode*) flatCLeaves (nodeCode)):
                   ((_CalcNode*) flatTree    (nodeCode)))->GetCompExp(0)->theData;
        
        for (a1 = 0; a1 < alphabetDimension; a1++)
        {
            for (a2 = 0; a2 < alphabetDimension; a2++)
            {
                ((float*)model)[nodeID*roundCharacters*roundCharacters+a2*roundCharacters+a1] =
                   (float)(tMatrix[a1*alphabetDimension+a2]);
            }
        }
    }
    
    // enqueueing the read and write buffers takes 1/2 the time, the kernel takes the other 1/2.
    // with no queueing, however, we still only see ~700lf/s, which isn't much better than the threaded CPU code.
    ciErr1 |= clEnqueueWriteBuffer(cqCommandQueue, cmModel_cache, CL_TRUE, 0,
                sizeof(cl_float)*roundCharacters*roundCharacters*(flatParents.lLength-1),
                model, 0, NULL, NULL);
    //clFinish(cqCommandQueue);
    if (ciErr1 != CL_SUCCESS)
    {
        printf("%i\n", ciErr1); //prints "1"
        switch(ciErr1)
        {
            case   CL_INVALID_COMMAND_QUEUE: printf("CL_INVALID_COMMAND_QUEUE\n"); break;
            case   CL_INVALID_CONTEXT: printf("CL_INVALID_CONTEXT\n"); break;
            case   CL_INVALID_MEM_OBJECT: printf("CL_INVALID_MEM_OBJECT\n"); break;
            case   CL_INVALID_VALUE: printf("CL_INVALID_VALUE\n"); break;   
            case   CL_INVALID_EVENT_WAIT_LIST: printf("CL_INVALID_EVENT_WAIT_LIST\n"); break;
                //          case   CL_MISALIGNED_SUB_BUFFER_OFFSET: printf("CL_MISALIGNED_SUB_BUFFER_OFFSET\n"); break;
                //          case   CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST: printf("CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST\n"); break;
            case   CL_MEM_OBJECT_ALLOCATION_FAILURE: printf("CL_MEM_OBJECT_ALLOCATION_FAILURE\n"); break;
            case   CL_OUT_OF_RESOURCES: printf("CL_OUT_OF_RESOURCES\n"); break;
            case   CL_OUT_OF_HOST_MEMORY: printf("CL_OUT_OF_HOST_MEMORY\n"); break;
            default: printf("Strange error\n"); //This is printed
        }
        printf("Error in clEnqueueWriteBuffer, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(EXIT_FAILURE);
    }
    /*
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error in clEnqueueWriteBuffer, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(EXIT_FAILURE);
    }
    */
#ifdef __OCLPOSIX__
    clock_gettime(CLOCK_MONOTONIC, &bufferEnd);
    buffSecs += (bufferEnd.tv_sec - bufferStart.tv_sec)+(bufferEnd.tv_nsec - bufferStart.tv_nsec)/BILLION;

    clock_gettime(CLOCK_MONOTONIC, &queueStart);
#endif
    //printf("Finished writing the model stuff\n");
    // Launch kernel
    for (int nodeIndex = 0; nodeIndex < updateNodes.lLength; nodeIndex++)
    {
        //printf("NewNode\n");
        long    nodeCode = updateNodes.lData[nodeIndex],
                parentCode = flatParents.lData[nodeCode];

        //printf("NewNode: %i, NodeCode: %i\n", nodeIndex, nodeCode);
        bool isLeaf = nodeCode < flatLeaves.lLength;

        if (isLeaf)
        {
            long nodeCodeTemp = nodeCode;
            int tempIntTagState = taggedInternals.lData[parentCode];
            int ambig = 0;
            for (int aI = 0; aI < siteCount; aI++)
                if (lNodeFlags[nodeCode*siteCount + aI] < 0) 
                    {
                        ambig = 1;
                    }
            if (!ambig)
            {   
                ciErr1 |= clSetKernelArg(ckLeafKernel, 6, sizeof(cl_long), (void*)&nodeCodeTemp);
                ciErr1 |= clSetKernelArg(ckLeafKernel, 7, sizeof(cl_long), (void*)&parentCode);
                ciErr1 |= clSetKernelArg(ckLeafKernel, 9, sizeof(cl_int), (void*)&tempIntTagState);
                ciErr1 |= clSetKernelArg(ckLeafKernel, 10, sizeof(cl_int), (void*)&nodeIndex);
                taggedInternals.lData[parentCode] = 1;
    
                //printf("Leaf!\n");
#ifdef __VERBOSE__
                printf("Leaf/Ambig Started (ParentCode: %i)...", parentCode);
#endif
                ciErr1 = clEnqueueNDRangeKernel(cqCommandQueue, ckLeafKernel, 2, NULL, 
                                                szGlobalWorkSize, szLocalWorkSize, 0, NULL, NULL);
            }
            else
            {
                ciErr1 |= clSetKernelArg(ckAmbigKernel, 6, sizeof(cl_long), (void*)&nodeCodeTemp);
                ciErr1 |= clSetKernelArg(ckAmbigKernel, 7, sizeof(cl_long), (void*)&parentCode);
                ciErr1 |= clSetKernelArg(ckAmbigKernel, 9, sizeof(cl_int), (void*)&tempIntTagState);
                ciErr1 |= clSetKernelArg(ckAmbigKernel, 10, sizeof(cl_int), (void*)&nodeIndex);
                taggedInternals.lData[parentCode] = 1;
    
                //printf("ambig!\n");
#ifdef __VERBOSE__
                printf("Leaf/Ambig Started ...");
#endif
                ciErr1 = clEnqueueNDRangeKernel(cqCommandQueue, ckAmbigKernel, 2, NULL, 
                                                szGlobalWorkSize, szLocalWorkSize, 0, NULL, NULL);
            }
            ciErr1 |= clFlush(cqCommandQueue);
            clFinish(cqCommandQueue);
#ifdef __VERBOSE__
            printf("Finished\n");
#endif
        }
        else
        {   
            long tempLeafState = 0;
            nodeCode -= flatLeaves.lLength;
            long nodeCodeTemp = nodeCode;
            int tempIntTagState = taggedInternals.lData[parentCode];
            ciErr1 |= clSetKernelArg(ckInternalKernel, 5, sizeof(cl_long), (void*)&nodeCodeTemp);
            ciErr1 |= clSetKernelArg(ckInternalKernel, 6, sizeof(cl_long), (void*)&parentCode);
            ciErr1 |= clSetKernelArg(ckInternalKernel, 8, sizeof(cl_int), (void*)&tempIntTagState);
            ciErr1 |= clSetKernelArg(ckInternalKernel, 9, sizeof(cl_int), (void*)&nodeIndex);
            taggedInternals.lData[parentCode] = 1;
#ifdef __VERBOSE__
            printf("Internal Started (ParentCode: %i)...", parentCode);
#endif
            ciErr1 = clEnqueueNDRangeKernel(cqCommandQueue, ckInternalKernel, 2, NULL, 
                                            szGlobalWorkSize, szLocalWorkSize, 0, NULL, NULL);

            //printf("internal!\n");
            ciErr1 |= clFlush(cqCommandQueue);
            clFinish(cqCommandQueue);
#ifdef __VERBOSE__
            printf("Finished\n");
#endif
        }
        if (ciErr1 != CL_SUCCESS)
        {
            printf("%i\n", ciErr1); //prints "1"
            switch(ciErr1)
            {
                case   CL_INVALID_PROGRAM_EXECUTABLE: printf("CL_INVALID_PROGRAM_EXECUTABLE\n"); break;
                case   CL_INVALID_COMMAND_QUEUE: printf("CL_INVALID_COMMAND_QUEUE\n"); break;
                case   CL_INVALID_KERNEL: printf("CL_INVALID_KERNEL\n"); break;
                case   CL_INVALID_CONTEXT: printf("CL_INVALID_CONTEXT\n"); break;   
                case   CL_INVALID_KERNEL_ARGS: printf("CL_INVALID_KERNEL_ARGS\n"); break;
                case   CL_INVALID_WORK_DIMENSION: printf("CL_INVALID_WORK_DIMENSION\n"); break;
                case   CL_INVALID_GLOBAL_WORK_SIZE: printf("CL_INVALID_GLOBAL_WORK_SIZE\n"); break;
                case   CL_INVALID_GLOBAL_OFFSET: printf("CL_INVALID_GLOBAL_OFFSET\n"); break;
                case   CL_INVALID_WORK_GROUP_SIZE: printf("CL_INVALID_WORK_GROUP_SIZE\n"); break;
                case   CL_INVALID_WORK_ITEM_SIZE: printf("CL_INVALID_WORK_ITEM_SIZE\n"); break;
                case   CL_INVALID_IMAGE_SIZE: printf("CL_INVALID_IMAGE_SIZE\n"); break;
                case   CL_OUT_OF_RESOURCES: printf("CL_OUT_OF_RESOURCES\n"); break;
                case   CL_MEM_OBJECT_ALLOCATION_FAILURE: printf("CL_MEM_OBJECT_ALLOCATION_FAILURE\n"); break;
                case   CL_INVALID_EVENT_WAIT_LIST: printf("CL_INVALID_EVENT_WAIT_LIST\n"); break;
                case   CL_OUT_OF_HOST_MEMORY: printf("CL_OUT_OF_HOST_MEMORY\n"); break;
                default: printf("Strange error\n"); //This is printed
            }
            printf("Error in clEnqueueNDRangeKernel, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
            Cleanup(EXIT_FAILURE);
        }
    }
    ciErr1 |= clEnqueueNDRangeKernel(cqCommandQueue, ckResultKernel, 2, NULL,
        szGlobalWorkSize, szLocalWorkSize, 0, NULL, NULL); 
   
    if (ciErr1 != CL_SUCCESS)
    {
        printf("%i\n", ciErr1); //prints "1"
        switch(ciErr1)
        {
            case   CL_INVALID_PROGRAM_EXECUTABLE: printf("CL_INVALID_PROGRAM_EXECUTABLE\n"); break;
            case   CL_INVALID_COMMAND_QUEUE: printf("CL_INVALID_COMMAND_QUEUE\n"); break;
            case   CL_INVALID_KERNEL: printf("CL_INVALID_KERNEL\n"); break;
            case   CL_INVALID_CONTEXT: printf("CL_INVALID_CONTEXT\n"); break;   
            case   CL_INVALID_KERNEL_ARGS: printf("CL_INVALID_KERNEL_ARGS\n"); break;
            case   CL_INVALID_WORK_DIMENSION: printf("CL_INVALID_WORK_DIMENSION\n"); break;
            case   CL_INVALID_GLOBAL_WORK_SIZE: printf("CL_INVALID_GLOBAL_WORK_SIZE\n"); break;
            case   CL_INVALID_GLOBAL_OFFSET: printf("CL_INVALID_GLOBAL_OFFSET\n"); break;
            case   CL_INVALID_WORK_GROUP_SIZE: printf("CL_INVALID_WORK_GROUP_SIZE\n"); break;
            case   CL_INVALID_WORK_ITEM_SIZE: printf("CL_INVALID_WORK_ITEM_SIZE\n"); break;
                //          case   CL_MISALIGNED_SUB_BUFFER_OFFSET: printf("CL_OUT_OF_HOST_MEMORY\n"); break;
            case   CL_INVALID_IMAGE_SIZE: printf("CL_INVALID_IMAGE_SIZE\n"); break;
            case   CL_OUT_OF_RESOURCES: printf("CL_OUT_OF_RESOURCES\n"); break;
            case   CL_MEM_OBJECT_ALLOCATION_FAILURE: printf("CL_MEM_OBJECT_ALLOCATION_FAILURE\n"); break;
            case   CL_INVALID_EVENT_WAIT_LIST: printf("CL_INVALID_EVENT_WAIT_LIST\n"); break;
            case   CL_OUT_OF_HOST_MEMORY: printf("CL_OUT_OF_HOST_MEMORY\n"); break;
            default: printf("Strange error\n"); //This is printed
        }
        printf("Error in clEnqueueNDRangeKernel, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(EXIT_FAILURE);
    }
    // Synchronous/blocking read of results, and check accumulated errors
    ciErr1 = clEnqueueReadBuffer(cqCommandQueue, cmResult_cache, CL_FALSE, 0,
            sizeof(cl_float)*siteCount, result_cache, 0,
            NULL, NULL);
/*
    ciErr1 |= clEnqueueReadBuffer(cqCommandQueue, cmNode_cache, CL_FALSE, 0,
            sizeof(cl_float)*flatNodes.lLength*siteCount*roundCharacters, node_cache, 0,
            NULL,NULL);
    
    ciErr1 |= clEnqueueReadBuffer(cqCommandQueue, cmroot_cache, CL_FALSE, 0,
            sizeof(cl_float)*roundCharacters*siteCount, root_cache, 0,
            NULL, NULL);
    ciErr1 |= clEnqueueReadBuffer(cqCommandQueue, cmroot_scalings, CL_FALSE, 0,
            sizeof(cl_int)*roundCharacters*siteCount, root_scalings, 0,
            NULL, NULL);

    printf("ResultCache: ");
    for (int i = 0; i < siteCount; i++)
    {
        printf(" %g", ((float*)result_cache)[i]);
    }
    printf("\n");
    printf("iNodeCache: ");
    for (int i = 0; i < (flatNodes.lLength)*roundCharacters*siteCount; i++)
    {
        if (i%(roundCharacters*siteCount) == 0) printf("NEWNODE \n");
        printf(" %g", ((float*)node_cache)[i]);
    }
    printf("\n");

    for (int i = 0; i < (flatNodes.lLength); i++)
    {
        printf("NEWNODE");
        for (int k = 0; k < siteCount; k++)
        {
            for (int j = 0; j < roundCharacters; j++)
            {
                printf(" %g", ((float*)node_cache)[i*roundCharacters*siteCount + k*roundCharacters + j]);
            }
            printf("\n");
        }
    }
    printf("\n");
*/

    if (ciErr1 != CL_SUCCESS)
    {
        printf("%i\n", ciErr1); //prints "1"
        switch(ciErr1)
        {
            case   CL_INVALID_COMMAND_QUEUE: printf("CL_INVALID_COMMAND_QUEUE\n"); break;
            case   CL_INVALID_CONTEXT: printf("CL_INVALID_CONTEXT\n"); break;
            case   CL_INVALID_MEM_OBJECT: printf("CL_INVALID_MEM_OBJECT\n"); break;
            case   CL_INVALID_VALUE: printf("CL_INVALID_VALUE\n"); break;   
            case   CL_INVALID_EVENT_WAIT_LIST: printf("CL_INVALID_EVENT_WAIT_LIST\n"); break;
                //          case   CL_MISALIGNED_SUB_BUFFER_OFFSET: printf("CL_MISALIGNED_SUB_BUFFER_OFFSET\n"); break;
                //          case   CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST: printf("CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST\n"); break;
            case   CL_MEM_OBJECT_ALLOCATION_FAILURE: printf("CL_MEM_OBJECT_ALLOCATION_FAILURE\n"); break;
            case   CL_OUT_OF_RESOURCES: printf("CL_OUT_OF_RESOURCES\n"); break;
            case   CL_OUT_OF_HOST_MEMORY: printf("CL_OUT_OF_HOST_MEMORY\n"); break;
            default: printf("Strange error\n"); //This is printed
        }
        printf("Error in clEnqueueReadBuffer, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(EXIT_FAILURE);
    }
    //--------------------------------------------------------
    
    
    clFinish(cqCommandQueue);
/*
    int* rootScalings = root_scalings;
    double rootVals[alphabetDimension*siteCount];

    //printf("rootCache: \n");
    //printf("rootScalings: \n");
    //#pragma omp parallel for
    for (int site = 0; site < siteCount; site++)
    {
        for (int pChar = 0; pChar < alphabetDimension; pChar++)
        {
            double scalingMul = pow(scalar, -rootScalings[site*roundCharacters+pChar]);
            double mantissa = ((float*)root_cache)[site*roundCharacters+pChar];
            rootVals[site*alphabetDimension + pChar] = mantissa*scalingMul;
            //printf("%i ", rootScalings[site*roundCharacters +pChar]);
            //rootVals[site*alphabetDimension + pChar] = mantissa;
// TODO: fix this, it might work just fine but bad numbers were going through it
        //  if (mantissa < 0) mantissa *= -1;
            //printf("%g ", ((float*)root_cache)[site*roundCharacters+pChar]);
        }
    }
    //printf("\n");
    //printf("resultList: \n");
    //printf("rootConditionals: \n");
    double resultList[siteCount];
    //#pragma omp parallel for
    for (int siteID = 0; siteID < siteCount; siteID++)
    {
        double accumulator = 0.;
        for (int p = 0; p < alphabetDimension; p++)
        {
            accumulator += rootVals[siteID*alphabetDimension + p] * theProbs[p];
            //printf("%g ", rootVals[siteID*alphabetDimension+p]);
        }
    //  printf("\n");
        resultList[siteID] = log(accumulator) * theFrequencies[siteID];
        //if (resultList[siteID] < -20000)
            //printf("Accumulator: %g", accumulator);
        //printf("%g ", resultList[siteID]);
    }
    //printf("\n");

    double result = 0.0;
    int i;
    //#pragma omp parallel for reduction (+:result) schedule(static)
    for (i = 0; i < siteCount; i++)
    {
        result += resultList[i];
    }
    */
    
#ifdef __OCLPOSIX__
    clock_gettime(CLOCK_MONOTONIC, &queueEnd);
    queueSecs += (queueEnd.tv_sec - queueStart.tv_sec)+(queueEnd.tv_nsec - queueStart.tv_nsec)/BILLION;
    clock_gettime(CLOCK_MONOTONIC, &mainStart);
#endif
    double oResult = 0.0;
    //#pragma omp parallel for reduction (+:oResult) schedule(static)
    for (int i = 0; i < siteCount; i++)
    {
        oResult += ((float*)result_cache)[i];
    }
#ifdef __OCLPOSIX__
    clock_gettime(CLOCK_MONOTONIC, &mainEnd);
    mainSecs += (mainEnd.tv_sec - mainStart.tv_sec)+(mainEnd.tv_nsec - mainStart.tv_nsec)/BILLION;
#endif
    //printf("! ");
    return oResult;
    //return result;
}


double _OCLEvaluator::launchmdsocl( _SimpleList& eupdateNodes,
                                    _SimpleList& eflatParents,
                                    _SimpleList& eflatNodes,
                                    _SimpleList& eflatCLeaves,
                                    _SimpleList& eflatLeaves,
                                    _SimpleList& eflatTree,
                                    _Parameter* etheProbs,
                                    _SimpleList& etheFrequencies,
                                    long* elNodeFlags,
                                    _SimpleList& etaggedInternals,
                                    _GrowingVector* elNodeResolutions)
{
#ifdef __OCLPOSIX__
    clock_gettime(CLOCK_MONOTONIC, &mainStart);
#endif

    
    updateNodes = eupdateNodes;
    taggedInternals = etaggedInternals;
    theFrequencies = etheFrequencies;


    if (!contextSet)
    {
        theProbs = etheProbs;
        flatNodes = eflatNodes;
        flatCLeaves = eflatCLeaves;
        flatLeaves = eflatLeaves;
        flatTree = eflatTree;
        flatParents = eflatParents;
        lNodeFlags = elNodeFlags;
        lNodeResolutions = elNodeResolutions;
        setupContext();
        contextSet = true;
    }

#ifdef __OCLPOSIX__
    clock_gettime(CLOCK_MONOTONIC, &mainEnd);
    mainSecs += (mainEnd.tv_sec - mainStart.tv_sec)+(mainEnd.tv_nsec - mainStart.tv_nsec)/BILLION;
#endif

    return oclmain();
}


void _OCLEvaluator::Cleanup (int iExitCode)
{
    if (!clean)
    {   
        printf("Time in main: %.4lf seconds\n", mainSecs);
        printf("Time in updating transition buffer: %.4lf seconds\n", buffSecs);
        printf("Time in queue: %.4lf seconds\n", queueSecs);
        printf("Time in Setup: %.4lf seconds\n", setupSecs);
        // Cleanup allocated objects
        printf("Starting Cleanup...\n\n");
        if(ckLeafKernel)clReleaseKernel(ckLeafKernel);  
        if(ckInternalKernel)clReleaseKernel(ckInternalKernel);  
        if(ckAmbigKernel)clReleaseKernel(ckAmbigKernel);  
        if(ckResultKernel)clReleaseKernel(ckResultKernel);  
        if(cpLeafProgram)clReleaseProgram(cpLeafProgram);
        if(cpInternalProgram)clReleaseProgram(cpInternalProgram);
        if(cpAmbigProgram)clReleaseProgram(cpAmbigProgram);
        if(cpResultProgram)clReleaseProgram(cpResultProgram);
        if(cqCommandQueue)clReleaseCommandQueue(cqCommandQueue);
        printf("Halfway...\n\n");
        if(cxGPUContext)clReleaseContext(cxGPUContext);
        if(cmNode_cache)clReleaseMemObject(cmNode_cache);
        if(cmModel_cache)clReleaseMemObject(cmModel_cache);
        if(cmNodRes_cache)clReleaseMemObject(cmNodRes_cache);
        if(cmNodFlag_cache)clReleaseMemObject(cmNodFlag_cache);
        if(cmroot_cache)clReleaseMemObject(cmroot_cache);
        if(cmroot_scalings)clReleaseMemObject(cmroot_scalings);
        if(cmScalings_cache)clReleaseMemObject(cmScalings_cache);
        if(cmFreq_cache)clReleaseMemObject(cmFreq_cache);
        if(cmProb_cache)clReleaseMemObject(cmProb_cache);
        if(cmResult_cache)clReleaseMemObject(cmResult_cache);
        printf("Done with ocl stuff...\n\n");
        // Free host memory
        free(node_cache); 
        free(model);
        free(nodRes_cache);
        free(nodFlag_cache);
        free(scalings_cache);
        free(prob_cache);
        free(freq_cache);
        free(root_cache);
        free(result_cache);
        free(root_scalings);
        printf("Done!\n\n");
        clean = true;
        exit(0);
        
        if (iExitCode = EXIT_FAILURE)
            exit (iExitCode);
    }
}

unsigned int _OCLEvaluator::roundUpToNextPowerOfTwo(unsigned int x)
{
    x--;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x++;
    
    return x;
}

double _OCLEvaluator::roundDoubleUpToNextPowerOfTwo(double x)
{
    return pow(2, ceil(log2(x)));
}
#endif

