// *********************************************************************
// OpenCL likelihood function Notes:  
//
// Runs computations with OpenCL on the GPU device and then checks results 
// against basic host CPU/C++ computation.
// 
// TODO: array instantiation needs to be nixed in favor of pulling
// array dimensions from parameters. 
// TODO: return the results the results are the parentcache.
//
// *********************************************************************

#include <stdio.h>
#include <assert.h>
#include <sys/sysctl.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include "calcnode.h"

//struct timespec begin;
//struct timespec end;

#if defined(__APPLE__)
#include <OpenCL/OpenCL.h>
typedef float fpoint;
typedef cl_float clfp;
#define FLOATPREC "typedef float fpoint; \n"
#else
#include <oclUtils.h>
typedef double fpoint;
typedef cl_double clfp;
#define FLOATPREC "typedef double fpoint; \n"
#endif

// Constants
//**********************************************************************
//#define SITES           1000    //originally 1000
//#define CHARACTERS      61      //originally 61 (codons)
//#define NODES           150        //originally 100


// Scaling elements
//**********************************************************************
fpoint uflowThresh     = 0.00000000000000000000000000000000000000000000000000000001;
fpoint scalar          = 100.0;
 void* scalings;
cl_mem cmScalings;

// Name of the file with the source code for the computation kernel
// *********************************************************************
//const char* cSourceFile = "oclFirstLoop.cl";

// Host buffers for demo
// *********************************************************************
 void* Golden;                   // Host buffer for host golden processing cross check
 void    *node_cache, *parent_cache, *model, *parent_results;

// OpenCL Vars
cl_context cxGPUContext;        // OpenCL context
cl_command_queue cqCommandQueue;// OpenCL command que
cl_platform_id cpPlatform;      // OpenCL platform
cl_device_id cdDevice;          // OpenCL device
cl_program cpProgram;           // OpenCL program
cl_kernel ckKernel;             // OpenCL kernel
cl_mem cmNode_cache;            // OpenCL device source for node_cache
cl_mem cmParent_cache;          // OpenCL device source for parent_cache
cl_mem cmModel;                 // OpenCL device source for model
size_t szGlobalWorkSize;        // 1D var for Total # of work items
size_t szLocalWorkSize;         // 1D var for # of work items in the work group 
size_t localMemorySize;         // size of local memory buffer for kernel scratch
size_t szParmDataBytes;         // Byte size of context information
size_t szKernelLength;          // Byte size of kernel code
cl_int ciErr1, ciErr2;          // Error code var
char* cPathAndName = NULL;      // var for full paths to data, src, etc.
int sites, nodes, chars;

// Forward Declarations
// *********************************************************************
void FirstLoopHost(const fpoint* node_cache, const fpoint* model, fpoint* parent_cache);
void Cleanup (int iExitCode);
unsigned int roundUpToNextPowerOfTwo(unsigned int x);
double roundDoubleUpToNextPowerOfTwo(double x);
int launchmdsocl(double * parent_results,
                 long siteCount,
                 double nodeCount,
                 long alphabetDimension,
                 _SimpleList& updateNodes,
                 _SimpleList& flatParents,
                 _SimpleList& flatNodes,
                 _SimpleList& flatCLeaves,
                 _SimpleList& flatTree,
                 _Parameter* iNodeCache,
                 long* lNodeFlags,
                 _SimpleList taggedInternals,
                 _GrowingVector* lNodeResolutions);
//extern char* load_program_source(const char *filename, const char *argv, size_t *szKernelLength);


// Main function 
// *********************************************************************
int oclmain()
{
	
    // time stuff:
    time_t dtimer;
    time_t htimer;
	
    // set and log Global and Local work size dimensions
    
    szLocalWorkSize = roundDoubleUpToNextPowerOfTwo(chars);
	//  szGlobalWorkSize = NODES * SITES * CHARACTERS;
    szGlobalWorkSize = roundDoubleUpToNextPowerOfTwo(sites) * roundDoubleUpToNextPowerOfTwo(chars);
    localMemorySize = roundDoubleUpToNextPowerOfTwo(chars);
    printf("Global Work Size \t\t= %d\nLocal Work Size \t\t= %d\n# of Work Groups \t\t= %d\n\n", 
           szGlobalWorkSize, szLocalWorkSize, 
           (szGlobalWorkSize % szLocalWorkSize + szGlobalWorkSize/szLocalWorkSize)); 
	
    // Allocate and initialize host arrays 
    //*************************************************
    printf( "Allocate and Init Host Mem...\n"); 
/*
    node_cache      = (void*)malloc (sizeof(clfp)*CHARACTERS*SITES);
    parent_cache    = (void*)malloc (sizeof(clfp)*CHARACTERS*SITES);
    scalings        = (void*)malloc (sizeof(int)*CHARACTERS*SITES);
    model           = (void*)malloc (sizeof(clfp)*CHARACTERS*CHARACTERS);
//    Golden          = (void*)malloc (sizeof(clfp)*CHARACTERS*SITES);
	
    long tempindex = 0;
    // initialize the vectors
    for (tempindex = 0; tempindex < (CHARACTERS*SITES); tempindex++)
    {
        ((fpoint*)node_cache)[tempindex] = 1./CHARACTERS; // this is just dummy filler
//        ((fpoint*)Golden)[tempindex] = 1.;
        ((fpoint*)parent_cache)[tempindex] = 1.;
        ((int*)scalings)[tempindex] = 0;
    }
	
    // initialize the model
    for (tempindex = 0; tempindex < (CHARACTERS*CHARACTERS); tempindex++)
    {
        ((fpoint*)model)[tempindex] = 1./CHARACTERS; // this is just dummy filler
    }
*/	
    //**************************************************
    dtimer = time(NULL); 
	
    //Get an OpenCL platform
    ciErr1 = clGetPlatformIDs(1, &cpPlatform, NULL);
	
    printf("clGetPlatformID...\n"); 
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error in clGetPlatformID, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(EXIT_FAILURE);
    }
	
    //Get the devices
    ciErr1 = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &cdDevice, NULL);
    printf("clGetDeviceIDs...\n"); 
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
    
    cl_uint extcheck;
    ciErr1 = clGetDeviceInfo(cdDevice, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, 
							 sizeof(cl_uint), &extcheck, NULL);
    if (extcheck ==0 ) 
    {
        printf("Device does not support double precision.\n");
    }
    
    size_t returned_size = 0;
    cl_char vendor_name[1024] = {0};
    cl_char device_name[1024] = {0};
    ciErr1 = clGetDeviceInfo(cdDevice, CL_DEVICE_VENDOR, sizeof(vendor_name), 
							 vendor_name, &returned_size);
    ciErr1 |= clGetDeviceInfo(cdDevice, CL_DEVICE_NAME, sizeof(device_name), 
							  device_name, &returned_size);
    assert(ciErr1 == CL_SUCCESS);
    printf("Connecting to %s %s...\n", vendor_name, device_name);
	
    //Create the context
    cxGPUContext = clCreateContext(0, 1, &cdDevice, NULL, NULL, &ciErr1);
    printf("clCreateContext...\n"); 
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error in clCreateContext, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(EXIT_FAILURE);
    }
	
    // Create a command-queue
    cqCommandQueue = clCreateCommandQueue(cxGPUContext, cdDevice, 0, &ciErr1);
    printf("clCreateCommandQueue...\n"); 
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error in clCreateCommandQueue, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(EXIT_FAILURE);
    }
	
    // Allocate the OpenCL buffer memory objects for source and result on the device GMEM
    cmNode_cache = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY, 
								  sizeof(clfp) * chars * sites, NULL, &ciErr1);
    cmModel = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY, 
							 sizeof(clfp) * chars * chars, NULL, &ciErr2);
    ciErr1 |= ciErr2;
    cmParent_cache = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, 
									sizeof(clfp) * chars * sites, NULL, &ciErr2);
    ciErr1 |= ciErr2;
    cmScalings = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE,
								sizeof(cl_int) * chars * sites, NULL, &ciErr2);
    ciErr1 |= ciErr2;
    printf("clCreateBuffer...\n"); 
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error in clCreateBuffer, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(EXIT_FAILURE);
    }
	
	// Create the program
    
	// Read the OpenCL kernel in from source file
    
	const char *program_source = "\n" \
	"#pragma OPENCL EXTENSION cl_khr_fp64: enable																				\n" \
	"" FLOATPREC                                                                                                                    \
	"__kernel void FirstLoop(__global const fpoint* node_cache, __global const fpoint* model, __global fpoint* parent_cache, 	\n" \
    "    __local fpoint* nodeScratch, __local fpoint * modelScratch, int nodes, int sites, int characters,						\n" \
    "   __global int* scalings, fpoint uflowthresh, fpoint scalar)																\n" \
	"{																															\n" \
	"   int parentCharGlobal = get_global_id(0); // a unique global ID for each parentcharacter									\n" \
    "   int parentCharLocal = get_local_id(0); // a local ID unique within the site.											\n" \
	"	if ((parentCharGlobal/characters) >= sites) return;																		\n" \
	"	if (parentCharLocal >= characters) return;																				\n" \
	"   nodeScratch[parentCharLocal] = node_cache[parentCharGlobal];															\n" \
    "   modelScratch[parentCharLocal] = model[parentCharLocal * characters + parentCharLocal];									\n" \
	"	barrier(CLK_LOCAL_MEM_FENCE);																							\n" \
	"   fpoint sum = 0.;																										\n" \
    "   long myChar;																											\n" \
    "   for (myChar = 0; myChar < characters; myChar++)																			\n" \
    "   {																														\n" \
    "       sum += nodeScratch[myChar] * modelScratch[myChar];																	\n" \
    "   }																														\n" \
    "   barrier(CLK_LOCAL_MEM_FENCE);																							\n" \
    "   while (parent_cache[parentCharGlobal] < uflowthresh)																	\n" \
    "   {																														\n" \
    "       parent_cache[parentCharGlobal] *= scalar;																			\n" \
    "    	scalings[parentCharGlobal] += 1;																					\n" \
    "	}																														\n" \
	"   parent_cache[parentCharGlobal] *= sum;																					\n" \
	"}																															\n" \
	"\n";
    
    
    //	printf("LoadProgSource (%s)...\n", cSourceFile); 
    //	char *program_source = load_program_source(cSourceFile, argv[0],  &szKernelLength);
	cpProgram = clCreateProgramWithSource(cxGPUContext, 1, (const char**)&program_source,
                                          NULL, &ciErr1);
	
    printf("clCreateProgramWithSource...\n"); 
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error in clCreateProgramWithSource, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(EXIT_FAILURE);
    }
	
    ciErr1 = clBuildProgram(cpProgram, 1, &cdDevice, NULL, NULL, NULL);
    printf("clBuildProgram...\n"); 
	
    // Shows the log
    char* build_log;
    size_t log_size;
    // First call to know the proper size
    clGetProgramBuildInfo(cpProgram, cdDevice, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
    build_log = new char[log_size+1];   
    // Second call to get the log
    clGetProgramBuildInfo(cpProgram, cdDevice, CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL);
    build_log[log_size] = '\0';
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
    ckKernel = clCreateKernel(cpProgram, "FirstLoop", &ciErr1);
    printf("clCreateKernel (FirstLoop)...\n"); 
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error in clCreateKernel, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(EXIT_FAILURE);
    }
	
	
    int tempNodeCount = nodes;
    int tempSiteCount = sites;
    int tempCharCount = chars;
	
    // Set the Argument values
    ciErr1 = clSetKernelArg(ckKernel, 0, sizeof(cl_mem), (void*)&cmNode_cache);
    ciErr1 |= clSetKernelArg(ckKernel, 1, sizeof(cl_mem), (void*)&cmModel);
    ciErr1 |= clSetKernelArg(ckKernel, 2, sizeof(cl_mem), (void*)&cmParent_cache);
    ciErr1 |= clSetKernelArg(ckKernel, 3, localMemorySize * sizeof(fpoint), NULL);
    ciErr1 |= clSetKernelArg(ckKernel, 4, localMemorySize * sizeof(fpoint), NULL);
    ciErr1 |= clSetKernelArg(ckKernel, 5, sizeof(cl_int), (void*)&tempNodeCount);
    ciErr1 |= clSetKernelArg(ckKernel, 6, sizeof(cl_int), (void*)&tempSiteCount);
    ciErr1 |= clSetKernelArg(ckKernel, 7, sizeof(cl_int), (void*)&tempCharCount);
    ciErr1 |= clSetKernelArg(ckKernel, 8, sizeof(cl_mem), (void*)&cmScalings);
    ciErr1 |= clSetKernelArg(ckKernel, 9, sizeof(clfp), (void*)&uflowThresh);
    ciErr1 |= clSetKernelArg(ckKernel, 10, sizeof(clfp), (void*)&scalar);
    printf("clSetKernelArg 0 - 10...\n\n"); 
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error in clSetKernelArg, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(EXIT_FAILURE);
    }
	
    // --------------------------------------------------------
    // Start Core sequence... copy input data to GPU, compute, copy results back
	
    // Asynchronous write of data to GPU device
    ciErr1 = clEnqueueWriteBuffer(cqCommandQueue, cmNode_cache, CL_FALSE, 0, 
								  sizeof(clfp) * chars * sites, node_cache, 0, NULL, NULL);
    ciErr1 |= clEnqueueWriteBuffer(cqCommandQueue, cmModel, CL_FALSE, 0, 
								   sizeof(clfp) * chars * chars, model, 0, NULL, NULL);
    ciErr1 |= clEnqueueWriteBuffer(cqCommandQueue, cmParent_cache, CL_FALSE, 0, 
								   sizeof(clfp) * chars * sites, parent_cache, 0, NULL, NULL);
    ciErr1 |= clEnqueueWriteBuffer(cqCommandQueue, cmScalings, CL_FALSE, 0,
								   sizeof(cl_int) * chars * sites, scalings, 0, NULL, NULL);
    printf("clEnqueueWriteBuffer (node_cache, parent_cache and model)...\n"); 
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error in clEnqueueWriteBuffer, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(EXIT_FAILURE);
    }
	
    // Launch kernel
    
    int nodeIndex;
	
	//    clock_gettime(CLOCK_REALTIME, &begin);
    
    printf("clEnqueueNDRangeKernel (FirstLoop)...\n"); 
    for (nodeIndex = 0; nodeIndex < nodes; nodeIndex++)
    {
		
        
        
        ciErr1 = clEnqueueNDRangeKernel(cqCommandQueue, ckKernel, 1, NULL, 
										&szGlobalWorkSize, &szLocalWorkSize, 0, NULL, NULL);
        
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
		
		//      ciErr1 = clEnqueueBarrier(cqCommandQueue);
		
    }
    
	//    clock_gettime(CLOCK_REALTIME, &end);
	
    // Synchronous/blocking read of results, and check accumulated errors
    ciErr1 = clEnqueueReadBuffer(cqCommandQueue, cmParent_cache, CL_TRUE, 0, sizeof(clfp) * chars * sites, parent_results, 0, NULL, NULL);
    ciErr1 = clEnqueueReadBuffer(cqCommandQueue, cmScalings, CL_TRUE, 0, sizeof(cl_int) * chars * sites, scalings, 0, NULL, NULL);
    printf("clEnqueueReadBuffer...\n\n"); 
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
    printf("%f seconds on device\n", difftime(time(NULL), dtimer));
	//    double timeDifference = ((double)end.tv_sec + ((double)end.tv_nsec/1000.0))-((double)begin.tv_sec + ((double)begin.tv_nsec/1000.0));
	//    printf("%f seconds on device\n", timeDifference);
    
    htimer = time(NULL);
	
	
    // Compute and compare results for golden-host and report errors and pass/fail
//    printf("Comparing against Host/C++ computation...\n\n"); 
    
//    FirstLoopHost ((const fpoint*)node_cache, (const fpoint*)model, (fpoint*)Golden);
	
//    printf("%f seconds on host\n", difftime(time(NULL), htimer));
	
	/*
	 int goldenLoop = 0;
	 for (goldenLoop = 0; goldenLoop < SITES; goldenLoop++)
	 {
	 printf("Golden: %e\n", ((fpoint*)Golden)[goldenLoop*CHARACTERS]);
	 }
	 */
	
	
	// Unscaling
	//***************************************************************************
    int scIndex;
    for (scIndex = 0; scIndex < chars * sites; scIndex++)
    {
        while (((int*)scalings)[scIndex] > 0)
        {
            ((fpoint*)parent_results)[scIndex] /= scalar;
            ((int*)scalings)[scIndex]--;
        }
    }
	
/*	
    bool match = true;
	//        int unmatching = 0;
	//        long firstUnmatch = -1;
	//        long lastUnmatch = -1;
    int verI;
    for (verI  = 0; verI < CHARACTERS*SITES; verI++)
    {
		if (verI%(SITES)==0)
			printf("Device: %e, Host: %e, Scalings: %i\n", ((fpoint*)parent_cache)[verI], ((fpoint*)Golden)[verI], ((int*)scalings)[verI]); 
		//                if (((fpoint*)parent_cache)[i] != ((fpoint*)Golden)[i]) match = false;
        if (((fpoint*)parent_cache)[verI] != ((fpoint*)Golden)[verI])
        {
            match = false;
			//                        unmatching++;
			//                        if (firstUnmatch == -1) firstUnmatch = i;
			//                        if (lastUnmatch < i) lastUnmatch = i;
        }
    }
	//        printf("Unmatching: %i, First: %d, Last: %d\n", unmatching, firstUnmatch, lastUnmatch);
    printf("%s\n\n", (match) ? "PASSED" : "FAILED");
*/	
	
    // Cleanup and leave
    Cleanup (EXIT_SUCCESS);
    
    return 0;
}


int launchmdsocl(double * parent_results,
                 long siteCount,
                 double nodeCount,
                 long alphabetDimension,
                 _SimpleList& updateNodes,
                 _SimpleList& flatParents,
                 _SimpleList& flatNodes,
                 _SimpleList& flatCLeaves,
                 _SimpleList& flatTree,
                 _Parameter* iNodeCache,
                 long* lNodeFlags,
                 _SimpleList taggedInternals,
                 _GrowingVector* lNodeResolutions)
{
    // so I have all of this in OpenCL land now. All of the operations that remain should be setting up memory or in the Node loop above. 
    
    // what about taggedInternals? This can be done on the CPU or the gpu, realistically. doesn't matter. 
    
    // memory setup:
    // cache from which everything is read:
        // proof of concept: node_cache
        // hyphy: (per node) childVector -> iNodeCache @ nodeCode*siteCount*alphabetDimension
        // OpenCL: node_cache: -> iNodeCache, childVector index is determined in the node loop I think? (and then passed as a param?)
        // *NOTE: for ambiguous characters we will have to use the LUT on the device.
        // Probably move lNodeResolutions to the GPU
    
    // cache to which everything is written:
        // proof of concept: parent_cache
        // hyphy: (per node) parentConditionals -> iNodeCache @ parentCode*siteCount*alphabetDimension
        // OpenCL: node_cache: -> iNodeCache, parentConditional index is determined in the node loop I think? (and then passed as a param?)
    
    // transition matrix:
        // proof of concept: model_cache
        // hyphy: flatCLeaves or flatTree->GetCompExp(0)
        // OpenCL: Don't know! TODO!
        // build now, move onto GPU all at once, move a chunk into memory in each kernel. 
    
    // To move onto GPU:
        // iNodeCache
        // lNodeResolutions
        // transitionMatrixArray
    
    // To give each kernel:
        // an identity (node, site, parentChar)
        // appropriate pointers to the iNodeCache
        // whatever else is associated with the implementation
        
    
    // save all of the rest of the above somewhere. 
    
    // run oclmain()
    
    // put the result in parent_results
    
    
    
#ifdef MDSOCL 
    
    
    
    node_cache = (void *)extnode_cache;
    parent_cache = (void *)extparent_cache;
    scalings = (void *)extscalings;
    model = (void *)extmodel;
    parent_results = (void *)extparent_results;
    sites = extSites;
    nodes = extNodes;
    chars = extChars;
    
    double *model = (double *)malloc (sizeof(double)*characters*characters);
    double *node_cache = (double *)malloc (sizeof(double)*characters*sites);
    double *parent_cache = (double *)malloc (sizeof(double)*characters*sites);
    int *scalings = (int *)malloc (sizeof(int)*characters*sites);
    
    
    
#endif
    return oclmain();
}


void Cleanup (int iExitCode)
{
    // Cleanup allocated objects
    printf("Starting Cleanup...\n\n");
    if(cPathAndName)free(cPathAndName);
    if(ckKernel)clReleaseKernel(ckKernel);  
    if(cpProgram)clReleaseProgram(cpProgram);
    if(cqCommandQueue)clReleaseCommandQueue(cqCommandQueue);
    if(cxGPUContext)clReleaseContext(cxGPUContext);
    if(cmNode_cache)clReleaseMemObject(cmNode_cache);
    if(cmModel)clReleaseMemObject(cmModel);
    if(cmParent_cache)clReleaseMemObject(cmParent_cache);
	
    // Free host memory
    free(node_cache); 
    free(model);
    free (parent_cache);
    free(Golden);
    
    // exit (iExitCode);
}

// "Golden" Host processing vector addition function for comparison purposes
// *********************************************************************
void FirstLoopHost(const fpoint* hnode_cache, const fpoint* hmodel, fpoint* hparent_cache)
{
    long index, myChar, parentChar, site;
	
    fpoint sum;
	
	//        long i=0;
	//        for (i=0;i< CHARACTERS*SITES;i++)
	//        {
	//                if (i%10000==0)
	//                {
	//                        printf("Node_cache: %e, Parent_cache: %e\n", ((fpoint*)hnode_cache)[i], ((fpoint*)hparent_cache)[i]);
	//                }
	//        }
	
    for (index = 0; index < nodes; index++) // this is meant to simulate looping over tree nodes
    {        
        for (site = 0; site < sites; site++)
        {
            int siteIndex = site*chars;
            for (parentChar = 0; parentChar < chars; parentChar++)
            {
                sum = 0.;
                //here
                for (myChar = 0; myChar < chars; myChar++)
                {
                    sum += hnode_cache[site*chars+myChar] * hmodel[parentChar*chars+myChar];
                }
                hparent_cache[siteIndex+parentChar] *= sum;
            }
        }
		//                printf("Index: %i, ParentCache: %e\n", index, hparent_cache[0]);
    }      
	//        for (index = 0; index < SITES; index++)
	//        {
	//                printf("hparent_cache: %e\n", hparent_cache[index*CHARACTERS]);
	//        }
}

unsigned int roundUpToNextPowerOfTwo(unsigned int x)
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

double roundDoubleUpToNextPowerOfTwo(double x)
{
	return pow(2, ceil(log2(x)));
}
