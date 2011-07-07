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

#if defined(__APPLE__)
#include <OpenCL/OpenCL.h>
typedef float fpoint;
typedef cl_float clfp;
#define FLOATPREC "typedef float fpoint; \n"
#define PRAGMADEF "#pragma OPENCL EXTENSION cl_khr_fp64: enable \n"
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#elif defined(NVIDIA)
#include <oclUtils.h>
typedef double fpoint;
typedef cl_double clfp;
#define FLOATPREC "typedef double fpoint; \n"
#define PRAGMADEF "#pragma OPENCL EXTENSION cl_khr_fp64: enable \n"
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#elif defined(AMD)
#include <CL/opencl.h>
typedef double fpoint;
typedef cl_double clfp;
#define FLOATPREC "typedef double fpoint; \n"
#define PRAGMADEF "#pragma OPENCL EXTENSION cl_amd_fp64: enable \n"
#pragma OPENCL EXTENSION cl_amd_fp64: enable
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
//cl_program cpProgram;           // OpenCL program
//cl_kernel ckKernel;             // OpenCL kernel
cl_program cpLeafProgram;
cl_program cpInternalProgram;
cl_program cpAmbigProgram;
cl_kernel ckLeafKernel;
cl_kernel ckInternalKernel;
cl_kernel ckAmbigKernel;
size_t szGlobalWorkSize[2];        // 1D var for Total # of work items
size_t szLocalWorkSize[2];         // 1D var for # of work items in the work group 
size_t localMemorySize;         // size of local memory buffer for kernel scratch
size_t szParmDataBytes;         // Byte size of context information
size_t szKernelLength;          // Byte size of kernel code
cl_int ciErr1, ciErr2;          // Error code var
char* cPathAndName; 		    // var for full paths to data, src, etc.

cl_mem cmNode_cache;
cl_mem cmModel_cache;
cl_mem cmNodRes_cache;
cl_mem cmNodFlag_cache;
cl_mem cmroot_cache;
long siteCount, alphabetDimension; 
long* lNodeFlags;
_SimpleList    updateNodes, 
				flatParents,
				flatNodes,
				flatCLeaves,
				flatLeaves,
				flatTree,
				theFrequencies;
_Parameter 		*iNodeCache,
				*theProbs;
_SimpleList taggedInternals;
_GrowingVector* lNodeResolutions;

void *node_cache, *nodRes_cache, *nodFlag_cache;
double *root_cache, *model;



void _OCLEvaluator::init(	long esiteCount,
									long ealphabetDimension,
									_Parameter* eiNodeCache)
{
	clean = false;
	cPathAndName = NULL;
	contextSet = false;
    siteCount = esiteCount;
    alphabetDimension = ealphabetDimension;
    iNodeCache = eiNodeCache;
	mainSecs = 0.0;
	buffSecs = 0.0;
	queueSecs = 0.0;
	setupSecs = 0.0;

}

// So the two interfacing functions will be the constructor, called in SetupLFCaches, and launchmdsocl, called in ComputeBlock.
// Therefore all of these functions need to be finished, the context needs to be setup separately from the execution, the data needs 
// to be passed piecewise, and a pointer needs to be passed around in likefunc2.cpp. After that things should be going a bit faster, 
// though honestly this solution is geared towards analyses with a larger number of sites. 

// *********************************************************************
int _OCLEvaluator::setupContext(void)
{
	clock_gettime(CLOCK_MONOTONIC, &setupStart);
    //printf("Made it to the oclmain() function!\n");

    //long nodeResCount = sizeof(lNodeResolutions->theData)/sizeof(lNodeResolutions->theData[0]);
    long nodeFlagCount = updateNodes.lLength*siteCount;
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
    node_cache = (void*)malloc
        (sizeof(clfp)*roundCharacters*siteCount*(flatNodes.lLength)); // +1 for root
    nodRes_cache = (void*)malloc
        (sizeof(clfp)*roundUpToNextPowerOfTwo(nodeResCount));
	nodFlag_cache = (void*)malloc(sizeof(cl_long)*roundUpToNextPowerOfTwo(nodeFlagCount));

    //printf("Allocated all of the arrays!\n");
    //printf("setup the model, fixed tagged internals!\n");
//    printf("flatleaves: %i\n", flatLeaves.lLength);
//    printf("flatParents: %i\n", flatParents.lLength);
//    printf("flatCleaves: %i\n", flatCLeaves.lLength);
//    printf("flatNodes: %i\n", flatNodes.lLength);
//    printf("updateNodes: %i\n", updateNodes.lLength);
//    printf("flatTree: %i\n", flatTree.lLength);

    //for (int i = 0; i < nodeCount*siteCount*alphabetDimension; i++)
//	printf("siteCount: %i, alphabetDimension: %i \n", siteCount, alphabetDimension);
	int alphaI = 0;
    for (int i = 0; i < (flatNodes.lLength)*roundCharacters*siteCount; i++)
    {
		if (i%(roundCharacters) < alphabetDimension)
		{
        	((fpoint*)node_cache)[i] = iNodeCache[alphaI];
			alphaI++;
		}
//		double t = iNodeCache[i];        
//		if (i%(siteCount*alphabetDimension) == 0)
//            printf("Got another one %g\n",t);
		//printf ("%i\n",i);
    }
    //printf("Built node_cache\n");
    if (ambiguousNodes)
        for (int i = 0; i < nodeResCount; i++)
            ((fpoint*)nodRes_cache)[i] = (double)(lNodeResolutions->theData[i]);
    //printf("Built nodRes_cache\n");
	for (int i = 0; i < nodeFlagCount; i++)
		((long*)nodFlag_cache)[i] = lNodeFlags[i];

    //printf("Created all of the arrays!\n");

    // alright, by now taggedInternals have been taken care of, and model has
    // been filled with all of the transition matrices. 

	clock_gettime(CLOCK_MONOTONIC, &setupEnd);
	setupSecs += (setupEnd.tv_sec - setupStart.tv_sec)+(setupEnd.tv_nsec - setupStart.tv_nsec)/BILLION;

    
    
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
    ciErr1 = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 4, &cdDevice, NULL);
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
    
    // set and log Global and Local work size dimensions
    
    szLocalWorkSize[0] = 16; // All of these will have to be genericized. 
    szLocalWorkSize[1] = 16;
    szGlobalWorkSize[0] = 64;
    szGlobalWorkSize[1] = ((siteCount + 8)/16)*16;
    printf("Global Work Size \t\t= %d, %d\nLocal Work Size \t\t= %d, %d\n# of Work Groups \t\t= %d\n\n", 
           sszGlobalWorkSize[0], zGlobalWorkSize[1], sszLocalWorkSize[0], zLocalWorkSize[1], 
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
                    sizeof(clfp)*roundCharacters*siteCount*flatNodes.lLength, node_cache,
                    &ciErr1);
    cmModel_cache = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                    sizeof(clfp)*roundCharacters*roundCharacters*updateNodes.lLength, 
                    NULL, &ciErr2);
    ciErr1 |= ciErr2;
    cmNodRes_cache = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                    sizeof(clfp)*roundUpToNextPowerOfTwo(nodeResCount), nodRes_cache, &ciErr2);
    ciErr1 |= ciErr2;
	cmNodFlag_cache = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
					sizeof(cl_long)*roundUpToNextPowerOfTwo(nodeFlagCount), nodFlag_cache, &ciErr2);
	ciErr1 |= ciErr2;
	cmroot_cache = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
					sizeof(clfp)*siteCount*roundCharacters, NULL, &ciErr2);
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

	//root_cache = (void*)malloc(sizeof(clfp)*roundCharacters*siteCount);
	root_cache = (double*)clEnqueueMapBuffer(cqCommandQueue, cmroot_cache, CL_TRUE,
												CL_MAP_READ, 0, sizeof(clfp)*siteCount*roundCharacters,
												0, NULL, NULL, NULL);
	clock_gettime(CLOCK_MONOTONIC, &setupStart);
	for (int i = 0; i < siteCount*roundCharacters; i++)
		(root_cache)[i] = 0.0;
	clock_gettime(CLOCK_MONOTONIC, &setupEnd);
	setupSecs += (setupEnd.tv_sec - setupStart.tv_sec)+(setupEnd.tv_nsec - setupStart.tv_nsec)/BILLION;
	model = (double*)clEnqueueMapBuffer(cqCommandQueue, cmModel_cache, CL_TRUE, CL_MAP_WRITE, 0, 
										sizeof(clfp)*roundCharacters*roundCharacters*updateNodes.lLength,
										0, NULL, NULL, NULL);
    //model = (void*)malloc
    //    (sizeof(clfp)*roundCharacters*roundCharacters*updateNodes.lLength);

    printf("Made all of the buffers on the device!\n");
    
//    printf("clCreateBuffer...\n");
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error in clCreateBuffer, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(EXIT_FAILURE);
    }

    
    // Create the program
	const char *leaf_source = "\n" \
	"" PRAGMADEF                                                                                                                        \
	"" FLOATPREC                                                                                                                        \
	"__kernel void LeafKernel(	__global fpoint* node_cache, 				// argument 0											\n" \
	"							__global const fpoint* model, 				// argument 1											\n" \
	"							__global const fpoint* nodRes_cache,   		// argument 2										 	\n" \
    "    						__constant long* nodFlag_cache, 			// argument 3											\n" \
    "    						long sites, 								// argument 4											\n" \
	"							long characters, 							// argument 5											\n" \
	"							long childNodeIndex, 						// argument 6											\n" \
	"							long parentNodeIndex, 						// argument 7											\n" \
	"							long roundCharacters, 						// argument 8											\n" \
	"							int intTagState, 							// argument 9											\n" \
	"							long nodeID			)						// argument 10											\n" \
	"{																														    	\n" \
	"	// block index																										    	\n" \
	"   int bx = get_block_id(0); 																									\n" \
	"   int by = get_block_id(1); 																									\n" \
	"   // thread index 																											\n" \
    "   int tx = get_local_id(0);																								 	\n" \
    "   int ty = get_local_id(1); 																							    	\n" \
    "   int childStart = ; 																							    	\n" \
	"	long site = parentCharGlobal/roundCharacters;																				\n" \
	"	long parentCharacter = parentCharGlobal & (roundCharacters-1);																\n" \
    "   int parentCharacterIndex = parentNodeIndex*sites*roundCharacters + site*roundCharacters + parentCharacter; 		            \n" \
    "   fpoint privateParentScratch = 1.0; 		        																		    \n" \
	"	if (intTagState == 1) 																										\n" \
	"		privateParentScratch = node_cache[parentCharacterIndex];																\n" \
	"	long siteState = nodFlag_cache[childNodeIndex*sites + site];																\n" \
	"	privateParentScratch *= model[nodeID*roundCharacters*roundCharacters + siteState*roundCharacters + parentCharacter];		\n" \
	"	node_cache[parentCharacterIndex] = privateParentScratch;																	\n" \
	"}																													    		\n" \
	"\n";
	const char *ambig_source = "\n" \
	"" PRAGMADEF                                                                                                                        \
	"" FLOATPREC                                                                                                                        \
	"__kernel void InternalKernel(	__global fpoint* node_cache, 				// argument 0										\n" \
	"								__global const fpoint* model, 				// argument 1										\n" \
	"								__global const fpoint* nodRes_cache,   		// argument 2									 	\n" \
    "    							__constant long* nodFlag_cache, 			// argument 3										\n" \
    "    							long sites, 								// argument 4										\n" \
	"								long characters, 							// argument 5										\n" \
	"								long childNodeIndex, 						// argument 6										\n" \
	"								long parentNodeIndex, 						// argument 7										\n" \
	"								long roundCharacters, 						// argument 8										\n" \
	"								int intTagState, 							// argument 9										\n" \
	"								long nodeID					)				// argument 10										\n" \
	"{																														    	\n" \
	"   int parentCharGlobal = get_global_id(0); // a unique global ID for each parentcharacter in the whole node's analysis 	   	\n" \
    "   int parentCharLocal = get_local_id(0); // a local ID unique within this set of parentcharacters in the site.		    	\n" \
	"	__local double childScratch[64];																							\n" \
	"	long site = parentCharGlobal/roundCharacters;																				\n" \
	"	long parentCharacter = parentCharGlobal & (roundCharacters-1);																\n" \
    "   int parentCharacterIndex = parentNodeIndex*sites*roundCharacters + site*roundCharacters + parentCharacter; 		            \n" \
	"	long siteState = nodFlag_cache[childNodeIndex*sites + site];																\n" \
    "   fpoint privateParentScratch = 1.0; 		        																		    \n" \
	"	if (intTagState == 1) 																										\n" \
	"		privateParentScratch = node_cache[parentCharacterIndex];																\n" \
	"	childScratch[parentCharLocal] = nodRes_cache[(-siteState-1)*roundCharacters + parentCharLocal];								\n" \
	"	barrier(CLK_LOCAL_MEM_FENCE);																						    	\n" \
	"	fpoint sum = 0.;																											\n" \
	"	long myChar;																												\n" \
	"	for (myChar = 0; myChar < characters; myChar++)																				\n" \
	"	{																															\n" \
    "  	 	sum += childScratch[myChar] * model[nodeID*roundCharacters*roundCharacters + myChar*roundCharacters + parentCharacter]; \n" \
	"	}																															\n" \
	"	privateParentScratch *= sum;																								\n" \
	"	node_cache[parentCharacterIndex] = privateParentScratch;																	\n" \
	"}																													    		\n" \
	"\n";
	const char *internal_source = "\n" \
	"" PRAGMADEF                                                                                                                        \
	"" FLOATPREC                                                                                                                        \
	"__kernel void InternalKernel(	__global fpoint* node_cache, 				// argument 0										\n" \
	"								__global const fpoint* model, 				// argument 1										\n" \
	"								__global const fpoint* nodRes_cache,   		// argument 2									 	\n" \
    "    							long sites, 								// argument 3										\n" \
	"								long characters, 							// argument 4										\n" \
	"								long childNodeIndex, 						// argument 5										\n" \
	"								long parentNodeIndex, 						// argument 6										\n" \
	"								long roundCharacters, 						// argument 7										\n" \
	"								int intTagState, 							// argument 8										\n" \
	"								long nodeID,								// argument 9										\n" \
	"								__global fpoint* root_cache		)			// argument 10										\n" \
	"{																														    	\n" \
	"   int siteGlobal = get_global_id(0); // a unique global ID for each parentcharacter in the whole node's analysis 	   	\n" \
    "   int siteLocal = get_local_id(0); // a local ID unique within this set of parentcharacters in the site.		    	\n" \
	"	__local double childScratch[64];																							\n" \
	"	long site = parentCharGlobal/roundCharacters;																				\n" \
	"	long parentCharacter = parentCharGlobal & (roundCharacters-1);																\n" \
    "   int parentCharacterIndex = parentNodeIndex*sites*roundCharacters + site*roundCharacters + parentCharacter; 		            \n" \
    "   fpoint privateParentScratch = 1.0; 		        																		    \n" \
	"	if (intTagState == 1) 																										\n" \
	"		privateParentScratch = node_cache[parentCharacterIndex];																\n" \
	"	childScratch[parentCharLocal] = node_cache[childNodeIndex*sites*roundCharacters + parentCharGlobal];						\n" \
	"	barrier(CLK_LOCAL_MEM_FENCE);																						    	\n" \
	"	fpoint sum = 0.;																											\n" \
	"	long myChar;																												\n" \
	"	for (myChar = 0; myChar < characters; myChar++)																				\n" \
	"	{																															\n" \
    "  	 	sum += childScratch[myChar] * model[nodeID*roundCharacters*roundCharacters + myChar*roundCharacters + parentCharacter]; \n" \
	"	}																															\n" \
	"	privateParentScratch *= sum;																								\n" \
	"	node_cache[parentCharacterIndex] = privateParentScratch;																	\n" \
	"	root_cache[site*roundCharacters+parentCharacter] = privateParentScratch;													\n" \
	"}																													    		\n" \
	"\n";
    
    
    //cpProgram = clCreateProgramWithSource(cxGPUContext, 1, (const char**)&program_source,
    //                                      NULL, &ciErr1);
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
    
    //printf("clCreateProgramWithSource...\n"); 
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error in clCreateProgramWithSource, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(EXIT_FAILURE);
    }
    
    //ciErr1 = clBuildProgram(cpProgram, 1, &cdDevice, NULL, NULL, NULL);



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
    //printf("clBuildProgram...\n"); 
    
    // Shows the log
    char* build_log;
    size_t log_size;
    // First call to know the proper size
    //clGetProgramBuildInfo(cpProgram, cdDevice, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
    //build_log = new char[log_size+1];   
    clGetProgramBuildInfo(cpLeafProgram, cdDevice, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
    build_log = new char[log_size+1];   
    clGetProgramBuildInfo(cpInternalProgram, cdDevice, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
    build_log = new char[log_size+1];   
    clGetProgramBuildInfo(cpAmbigProgram, cdDevice, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
    build_log = new char[log_size+1];   
    // Second call to get the log
    //clGetProgramBuildInfo(cpProgram, cdDevice, CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL);
    //build_log[log_size] = '\0';
    clGetProgramBuildInfo(cpLeafProgram, cdDevice, CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL);
    build_log[log_size] = '\0';
    clGetProgramBuildInfo(cpInternalProgram, cdDevice, CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL);
    build_log[log_size] = '\0';
    clGetProgramBuildInfo(cpAmbigProgram, cdDevice, CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL);
    build_log[log_size] = '\0';
    //printf("%s", build_log);
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
    ckLeafKernel = clCreateKernel(cpLeafProgram, "LeafKernel", &ciErr1);
    printf("clCreateKernel (LeafKernel)...\n"); 
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error in clCreateKernel, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(EXIT_FAILURE);
    }
    ckAmbigKernel = clCreateKernel(cpAmbigProgram, "InternalKernel", &ciErr1);
    printf("clCreateKernel (InternalKernel)...\n"); 
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
    
    long tempLeafState = 1;
    long tempSiteCount = siteCount;
    long tempCharCount = alphabetDimension;
	long tempChildNodeIndex = 0;
	long tempParentNodeIndex = 0;
	long tempRoundCharCount = localMemorySize;
	int tempTagIntState = 0;
	long tempNodeID = 0;
	int tempDivisor = divisor;
/*
    // Set the Argument values
	ciErr1 = clSetKernelArg(ckKernel, 0, sizeof(cl_mem), (void*)&cmNode_cache);
	ciErr1 |= clSetKernelArg(ckKernel, 1, sizeof(cl_mem), (void*)&cmModel_cache);
	ciErr1 |= clSetKernelArg(ckKernel, 2, sizeof(cl_mem), (void*)&cmNodRes_cache);
	ciErr1 |= clSetKernelArg(ckKernel, 3, sizeof(cl_mem), (void*)&cmNodFlag_cache);
	ciErr1 |= clSetKernelArg(ckKernel, 4, sizeof(fpoint) * roundCharacters, NULL); // Child
	//ciErr1 |= clSetKernelArg(ckKernel, 5, sizeof(fpoint) * roundCharacters*roundCharacters/divisor, NULL); // Model
	ciErr1 |= clSetKernelArg(ckKernel, 5, sizeof(fpoint) * roundCharacters*roundCharacters/divisor, NULL); // Model
	ciErr1 |= clSetKernelArg(ckKernel, 6, sizeof(fpoint) * roundCharacters/divisor, NULL); // Parent
	ciErr1 |= clSetKernelArg(ckKernel, 7, sizeof(cl_long), (void*)&tempLeafState); // reset this in the loop
	ciErr1 |= clSetKernelArg(ckKernel, 8, sizeof(cl_long), (void*)&tempSiteCount);
	ciErr1 |= clSetKernelArg(ckKernel, 9, sizeof(cl_long), (void*)&tempCharCount);
	ciErr1 |= clSetKernelArg(ckKernel, 10, sizeof(cl_long), (void*)&tempChildNodeIndex); // reset this in the loop
	ciErr1 |= clSetKernelArg(ckKernel, 11, sizeof(cl_long), (void*)&tempParentNodeIndex); // reset this in the loop
	ciErr1 |= clSetKernelArg(ckKernel, 12, sizeof(cl_long), (void*)&tempRoundCharCount); 
	ciErr1 |= clSetKernelArg(ckKernel, 13, sizeof(cl_int), (void*)&tempTagIntState); // reset this in the loop
	ciErr1 |= clSetKernelArg(ckKernel, 14, sizeof(cl_long), (void*)&tempNodeID); // reset this in the loop
	ciErr1 |= clSetKernelArg(ckKernel, 15, sizeof(cl_int), (void*)&tempDivisor);
	ciErr1 |= clSetKernelArg(ckKernel, 16, sizeof(cl_mem), (void*)&cmroot_cache);
*/

	ciErr1 = clSetKernelArg(ckLeafKernel, 0, sizeof(cl_mem), (void*)&cmNode_cache);
	ciErr1 |= clSetKernelArg(ckLeafKernel, 1, sizeof(cl_mem), (void*)&cmModel_cache);
	ciErr1 |= clSetKernelArg(ckLeafKernel, 2, sizeof(cl_mem), (void*)&cmNodRes_cache);
	ciErr1 |= clSetKernelArg(ckLeafKernel, 3, sizeof(cl_mem), (void*)&cmNodFlag_cache);
	ciErr1 |= clSetKernelArg(ckLeafKernel, 4, sizeof(cl_long), (void*)&tempSiteCount);
	ciErr1 |= clSetKernelArg(ckLeafKernel, 5, sizeof(cl_long), (void*)&tempCharCount);
	ciErr1 |= clSetKernelArg(ckLeafKernel, 6, sizeof(cl_long), (void*)&tempChildNodeIndex); // reset this in the loop
	ciErr1 |= clSetKernelArg(ckLeafKernel, 7, sizeof(cl_long), (void*)&tempParentNodeIndex); // reset this in the loop
	ciErr1 |= clSetKernelArg(ckLeafKernel, 8, sizeof(cl_long), (void*)&tempRoundCharCount); 
	ciErr1 |= clSetKernelArg(ckLeafKernel, 9, sizeof(cl_int), (void*)&tempTagIntState); // reset this in the loop
	ciErr1 |= clSetKernelArg(ckLeafKernel, 10, sizeof(cl_long), (void*)&tempNodeID); // reset this in the loop

	ciErr1 |= clSetKernelArg(ckAmbigKernel, 0, sizeof(cl_mem), (void*)&cmNode_cache);
	ciErr1 |= clSetKernelArg(ckAmbigKernel, 1, sizeof(cl_mem), (void*)&cmModel_cache);
	ciErr1 |= clSetKernelArg(ckAmbigKernel, 2, sizeof(cl_mem), (void*)&cmNodRes_cache);
	ciErr1 |= clSetKernelArg(ckAmbigKernel, 3, sizeof(cl_mem), (void*)&cmNodFlag_cache);
	ciErr1 |= clSetKernelArg(ckAmbigKernel, 4, sizeof(cl_long), (void*)&tempSiteCount);
	ciErr1 |= clSetKernelArg(ckAmbigKernel, 5, sizeof(cl_long), (void*)&tempCharCount);
	ciErr1 |= clSetKernelArg(ckAmbigKernel, 6, sizeof(cl_long), (void*)&tempChildNodeIndex); // reset this in the loop
	ciErr1 |= clSetKernelArg(ckAmbigKernel, 7, sizeof(cl_long), (void*)&tempParentNodeIndex); // reset this in the loop
	ciErr1 |= clSetKernelArg(ckAmbigKernel, 8, sizeof(cl_long), (void*)&tempRoundCharCount); 
	ciErr1 |= clSetKernelArg(ckAmbigKernel, 9, sizeof(cl_int), (void*)&tempTagIntState); // reset this in the loop
	ciErr1 |= clSetKernelArg(ckAmbigKernel, 10, sizeof(cl_long), (void*)&tempNodeID); // reset this in the loop

	ciErr1 |= clSetKernelArg(ckInternalKernel, 0, sizeof(cl_mem), (void*)&cmNode_cache);
	ciErr1 |= clSetKernelArg(ckInternalKernel, 1, sizeof(cl_mem), (void*)&cmModel_cache);
	ciErr1 |= clSetKernelArg(ckInternalKernel, 2, sizeof(cl_mem), (void*)&cmNodRes_cache);
	ciErr1 |= clSetKernelArg(ckInternalKernel, 3, sizeof(cl_long), (void*)&tempSiteCount);
	ciErr1 |= clSetKernelArg(ckInternalKernel, 4, sizeof(cl_long), (void*)&tempCharCount);
	ciErr1 |= clSetKernelArg(ckInternalKernel, 5, sizeof(cl_long), (void*)&tempChildNodeIndex); // reset this in the loop
	ciErr1 |= clSetKernelArg(ckInternalKernel, 6, sizeof(cl_long), (void*)&tempParentNodeIndex); // reset this in the loop
	ciErr1 |= clSetKernelArg(ckInternalKernel, 7, sizeof(cl_long), (void*)&tempRoundCharCount); 
	ciErr1 |= clSetKernelArg(ckInternalKernel, 8, sizeof(cl_int), (void*)&tempTagIntState); // reset this in the loop
	ciErr1 |= clSetKernelArg(ckInternalKernel, 9, sizeof(cl_long), (void*)&tempNodeID); // reset this in the loop
	ciErr1 |= clSetKernelArg(ckInternalKernel, 10, sizeof(cl_mem), (void*)&cmroot_cache);
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
                sizeof(clfp)*roundCharacters*siteCount*flatNodes.lLength, node_cache, 
                0, NULL, NULL);


    ciErr1 |= clEnqueueWriteBuffer(cqCommandQueue, cmNodRes_cache, CL_FALSE, 0,
                sizeof(clfp)*roundUpToNextPowerOfTwo(nodeResCount), nodRes_cache, 0, NULL, NULL);

    ciErr1 |= clEnqueueWriteBuffer(cqCommandQueue, cmNodFlag_cache, CL_FALSE, 0,
                sizeof(cl_long)*roundUpToNextPowerOfTwo(nodeFlagCount), nodFlag_cache, 0, NULL, NULL);
	
 */   
	ciErr1 |= clEnqueueWriteBuffer(cqCommandQueue, cmroot_cache, CL_FALSE, 0,
				sizeof(clfp)*siteCount*roundCharacters, root_cache, 0, NULL, NULL);
    printf("clEnqueueWriteBuffer (root_cache, etc.)...\n"); 
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error in clEnqueueWriteBuffer, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(EXIT_FAILURE);
    }
}	

double _OCLEvaluator::oclmain(void)
{
	// so far this wholebuffer rebuild takes almost no time at all. Perhaps not true re:queue
	// Fix the model cache
	clock_gettime(CLOCK_MONOTONIC, &bufferStart);
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
	#pragma omp parallel for default(none) shared(updateNodes, flatParents, flatLeaves, flatCLeaves, flatTree, alphabetDimension, model, roundCharacters) private(nodeCode, parentCode, isLeaf, tMatrix, a1, a2)
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
                //((fpoint*)model)[nodeID*roundCharacters*roundCharacters+a1*roundCharacters+a2] =
                //   (fpoint)(tMatrix[a1*alphabetDimension+a2]);
                ((fpoint*)model)[nodeID*roundCharacters*roundCharacters+a2*roundCharacters+a1] =
                   (fpoint)(tMatrix[a1*alphabetDimension+a2]);
            }
        }
	}
	
	// enqueueing the read and write buffers takes 1/2 the time, the kernel takes the other 1/2.
	// with no queueing, however, we still only see ~700lf/s, which isn't much better than the threaded CPU code.
    ciErr1 |= clEnqueueWriteBuffer(cqCommandQueue, cmModel_cache, CL_FALSE, 0,
                sizeof(clfp)*roundCharacters*roundCharacters*updateNodes.lLength,
                model, 0, NULL, NULL);
	clFinish(cqCommandQueue);
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error in clEnqueueWriteBuffer, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(EXIT_FAILURE);
    }
	clock_gettime(CLOCK_MONOTONIC, &bufferEnd);
	buffSecs += (bufferEnd.tv_sec - bufferStart.tv_sec)+(bufferEnd.tv_nsec - bufferStart.tv_nsec)/BILLION;

	clock_gettime(CLOCK_MONOTONIC, &queueStart);
	//printf("Finished writing the model stuff\n");
    // Launch kernel
    for (int nodeIndex = 0; nodeIndex < updateNodes.lLength; nodeIndex++)
    {

		long 	nodeCode = updateNodes.lData[nodeIndex],
				parentCode = flatParents.lData[nodeCode];

        bool isLeaf = nodeCode < flatLeaves.lLength;

		if (isLeaf)
		{
			long nodeCodeTemp = nodeCode;
			int tempIntTagState = taggedInternals.lData[parentCode];
			if (((double*)nodFlag_cache)[nodeCode*siteCount] >= 0)
			{	
				ciErr1 |= clSetKernelArg(ckLeafKernel, 6, sizeof(cl_long), (void*)&nodeCodeTemp);
				ciErr1 |= clSetKernelArg(ckLeafKernel, 7, sizeof(cl_long), (void*)&parentCode);
				ciErr1 |= clSetKernelArg(ckLeafKernel, 9, sizeof(cl_int), (void*)&tempIntTagState);
				ciErr1 |= clSetKernelArg(ckLeafKernel, 10, sizeof(cl_long), (void*)&nodeIndex);
				taggedInternals.lData[parentCode] = 1;
	
				ciErr1 = clEnqueueNDRangeKernel(cqCommandQueue, ckLeafKernel, 2, NULL, 
												&szGlobalWorkSize, &szLocalWorkSize, 0, NULL, NULL);
			}
			else
			{
				ciErr1 |= clSetKernelArg(ckAmbigKernel, 6, sizeof(cl_long), (void*)&nodeCodeTemp);
				ciErr1 |= clSetKernelArg(ckAmbigKernel, 7, sizeof(cl_long), (void*)&parentCode);
				ciErr1 |= clSetKernelArg(ckAmbigKernel, 9, sizeof(cl_int), (void*)&tempIntTagState);
				ciErr1 |= clSetKernelArg(ckAmbigKernel, 10, sizeof(cl_long), (void*)&nodeIndex);
				taggedInternals.lData[parentCode] = 1;
	
				ciErr1 = clEnqueueNDRangeKernel(cqCommandQueue, ckAmbigKernel, 2, NULL, 
												&szGlobalWorkSize, &szLocalWorkSize, 0, NULL, NULL);
			}
			ciErr1 |= clFlush(cqCommandQueue);
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
			ciErr1 |= clSetKernelArg(ckInternalKernel, 9, sizeof(cl_long), (void*)&nodeIndex);
			taggedInternals.lData[parentCode] = 1;
			ciErr1 = clEnqueueNDRangeKernel(cqCommandQueue, ckInternalKernel, 2, NULL, 
											&szInternalGlobalWorkSize, &szInternalLocalWorkSize, 0, NULL, NULL);

			ciErr1 |= clFlush(cqCommandQueue);
		}
/*
		long tempLeafState = 1;
        if (!isLeaf) 
		{
			nodeCode -= flatLeaves.lLength;
			tempLeafState = 0;
		}

		long nodeCodeTemp = nodeCode;
		int tempIntTagState = taggedInternals.lData[parentCode];
		ciErr1 |= clSetKernelArg(ckKernel, 7, sizeof(cl_long), (void*)&tempLeafState);
		ciErr1 |= clSetKernelArg(ckKernel, 10, sizeof(cl_long), (void*)&nodeCodeTemp);
		ciErr1 |= clSetKernelArg(ckKernel, 11, sizeof(cl_long), (void*)&parentCode);
		ciErr1 |= clSetKernelArg(ckKernel, 13, sizeof(cl_int), (void*)&tempIntTagState);
		ciErr1 |= clSetKernelArg(ckKernel, 14, sizeof(cl_long), (void*)&nodeIndex);
		taggedInternals.lData[parentCode] = 1;

		ciErr1 = clEnqueueNDRangeKernel(cqCommandQueue, ckKernel, 1, NULL, 
										&szGlobalWorkSize, &szLocalWorkSize, 0, NULL, NULL);
		ciErr1 |= clFlush(cqCommandQueue);
        
*/		
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
    }
    
    // Synchronous/blocking read of results, and check accumulated errors
    //ciErr1 = clEnqueueReadBuffer(cqCommandQueue, cmNode_cache, CL_TRUE, 0,
    //        sizeof(clfp)*roundCharacters*siteCount*(flatNodes.lLength), node_cache, 0,
    //        NULL, NULL);
    //ciErr1 = clEnqueueReadBuffer(cqCommandQueue, cmroot_cache, CL_TRUE, 0,
     //       sizeof(clfp)*roundCharacters*siteCount, root_cache, 0,
      //      NULL, NULL);
    ciErr1 = clEnqueueReadBuffer(cqCommandQueue, cmroot_cache, CL_FALSE, 0,
            sizeof(clfp)*roundCharacters*siteCount, root_cache, 0,
            NULL, NULL);
//    printf("clEnqueueReadBuffer...\n\n"); 
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

	clock_gettime(CLOCK_MONOTONIC, &queueEnd);
	queueSecs += (queueEnd.tv_sec - queueStart.tv_sec)+(queueEnd.tv_nsec - queueStart.tv_nsec)/BILLION;
//    printf("%f seconds on device\n", difftime(time(NULL), dtimer));
	
// Everything after this point takes a total of about two seconds.
/*
	time(&mainTimer);
	int alphaI = 0;
    for (int i = 0; i < (flatNodes.lLength)*siteCount*roundCharacters; i++)
    {
		if (i%roundCharacters < alphabetDimension)
		{
       		iNodeCache[alphaI] = ((_Parameter*)node_cache)[i];
			alphaI++;
   		}
	 }
 */   
	clock_gettime(CLOCK_MONOTONIC, &mainStart);
	double rootVals[alphabetDimension*siteCount];
	int alphaI = 0;
    for (int i = 0; i < siteCount*roundCharacters; i++)
    {
		if (i%roundCharacters < alphabetDimension)
		{
       		rootVals[alphaI] = ((double*)root_cache)[i];
			alphaI++;
   		}
	 }
	// Verify the node cache TESTING
/*
	printf("NodeCache: ");
    for (int i = 0; i < (flatNodes.lLength)*alphabetDimension*siteCount; i++)
    {
		if (i % (alphabetDimension*siteCount) == 0) printf("NEWNODE\n");
		printf(" %g", iNodeCache[i]);
    }
	printf("\n");
*/

//	double* rootConditionals = iNodeCache + alphabetDimension * ((flatTree.lLength-1)*siteCount);
	double* rootConditionals = rootVals;
	double result = 0.0;
//	printf("Rootconditionals: ");
	long p = 0;
	long siteID = 0;
	double accumulator = 0.;
	for (siteID = 0; siteID < siteCount; siteID++)
	{
		accumulator = 0.;
//		printf("%g ", *rootConditionals);
		for (p = 0; p < alphabetDimension; p++, rootConditionals++)
		{
			accumulator += *rootConditionals * theProbs[p];
		}
		result += log(accumulator) * theFrequencies[siteID];
	}
	// this bit takes .25sec total on the i7
	clock_gettime(CLOCK_MONOTONIC, &mainEnd);
	mainSecs += (mainEnd.tv_sec - mainStart.tv_sec)+(mainEnd.tv_nsec - mainStart.tv_nsec)/BILLION;
    
//	printf("\n");
    return result;
}


double _OCLEvaluator::launchmdsocl(	_SimpleList& eupdateNodes,
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
        // build now, move onto GPU all at once, move a chunk into memory in each kernel. 
    
    //printf("Made it to the pass-off Function!");
	
	clock_gettime(CLOCK_MONOTONIC, &mainStart);

    
    updateNodes = eupdateNodes;
	theProbs = etheProbs;
	theFrequencies = etheFrequencies;
	taggedInternals = etaggedInternals;


	if (!contextSet)
	{
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

	// setupContext is taking .75sec total on the i7
	clock_gettime(CLOCK_MONOTONIC, &mainEnd);
	mainSecs += (mainEnd.tv_sec - mainStart.tv_sec)+(mainEnd.tv_nsec - mainStart.tv_nsec)/BILLION;

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
		if(cPathAndName)free(cPathAndName);
		//if(ckKernel)clReleaseKernel(ckKernel);  
		if(ckLeafKernel)clReleaseKernel(ckLeafKernel);  
		if(ckInternalKernel)clReleaseKernel(ckInternalKernel);  
		//if(cpProgram)clReleaseProgram(cpProgram);
		if(cpLeafProgram)clReleaseProgram(cpLeafProgram);
		if(cpInternalProgram)clReleaseProgram(cpInternalProgram);
		if(cqCommandQueue)clReleaseCommandQueue(cqCommandQueue);
		printf("Halfway...\n\n");
		if(cxGPUContext)clReleaseContext(cxGPUContext);
		if(cmNode_cache)clReleaseMemObject(cmNode_cache);
		if(cmModel_cache)clReleaseMemObject(cmModel_cache);
		if(cmNodRes_cache)clReleaseMemObject(cmNodRes_cache);
		if(cmNodFlag_cache)clReleaseMemObject(cmNodFlag_cache);
		if(cmroot_cache)clReleaseMemObject(cmroot_cache);
		printf("Done with ocl stuff...\n\n");
		// Free host memory
		free(node_cache); 
		//free(model);
		free(nodRes_cache);
		free(nodFlag_cache);
		printf("Done!\n\n");
		clean = true;
		
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
