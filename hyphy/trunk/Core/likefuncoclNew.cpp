// *********************************************************************
// OpenCL likelihood function Notes:  
//
// Runs computations with OpenCL on the GPU device and then checks results 
// against basic host CPU/C++ computation.
// 
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

// OpenCL Vars
cl_context cxGPUContext;        // OpenCL context
cl_command_queue cqCommandQueue;// OpenCL command que
cl_platform_id cpPlatform;      // OpenCL platform
cl_device_id cdDevice;          // OpenCL device
cl_program cpProgram;           // OpenCL program
cl_kernel ckKernel;             // OpenCL kernel
size_t szGlobalWorkSize;        // 1D var for Total # of work items
size_t szLocalWorkSize;         // 1D var for # of work items in the work group 
size_t localMemorySize;         // size of local memory buffer for kernel scratch
size_t szParmDataBytes;         // Byte size of context information
size_t szKernelLength;          // Byte size of kernel code
cl_int ciErr1, ciErr2;          // Error code var
char* cPathAndName = NULL;      // var for full paths to data, src, etc.

cl_mem cmNode_cache;
cl_mem cmModel_cache;
cl_mem cmNodRes_cache;
double* parent_results;
long siteCount, alphabetDimension; 
double nodeCount;
_SimpleList&    updateNodes, 
                flatParents,
                flatNodes,
                flatCLeaves,
                flatTree;
_Parameter* iNodeCache;
_SimpleList taggedInternals;
_GrowingVector* lNodeResolutions;

void *model, *node_cache, *nodRes_cache;

// Forward Declarations
// *********************************************************************
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


// Main function 
// *********************************************************************
int oclmain()
{
    // TODO: finish accounting for all of the functions in the original method
    // (some of which will have to be done on the host, some will have to be
    // done on the Device)
    
    // Make transitionMatrixArray, do other host stuff:
    model = (void*)malloc
        (sizeof(clfp)*alphabetDimension*alphabetDimension*nodeCount);
    node_cache = (void*)malloc
        (sizeof(clfp)*alphabetDimension*siteCount*nodeCount);
    // FIXED: fix nodRes_cache size
    nodRes_cache = (void*)malloc
        (sizeof(clfp)*lNodeResolutions.lLength);
    for (long nodeID = 0; nodeID < updateNodes.lLength; nodeID++)
    {
        long    nodeCode = updateNodes.lData[nodeID],
                parentCode = flatParents.lData[nodeCode];

        bool isLeaf = nodeCode < flatLeaves.lLength;

        if (!isLeaf)
            nodeCode -= flatLeaves.lLength;

       _Parameter * parentConditionals = iNodeCache +	(parentCode  * siteCount) * alphabetDimension;

		if (taggedInternals.lData[parentCode] == 0)
		// mark the parent for update and clear its conditionals if needed
        {
            taggedInternals.lData[parentCode]	  = 1; // only do this once
            for (long k = 0, k3 = 0; k < siteCount; k++)
                for (long k2 = 0; k2 < alphabetDimension; k2++)
                    parentConditionals [k3++] = 1.0;
        }
		

						
		_Parameter  *		tMatrix = (isLeaf? ((_CalcNode*) flatCLeaves (nodeCode)):
                                               ((_CalcNode*) flatTree    (nodeCode)))->GetCompExp(0)->theData;

        for (int a1 = 0; a1 < alphabetDimension; a1++)
        {
            for (int a2 = 0; a2 < alphabetDimension; a2++)
            {
                ((fpoint*)model)[nodeID*alphabetDimension*alphabetDimension+a1*alphabetDimension+a2] =
                    tMatrix[a1*alphabetDimension+a2];
            }
        }
    }
    for (int i = 0; i < nodeCount*siteCount*alphabetDimension; i++)
    {
       ((fpoint*)node_cache)[i] = iNodeCache[i];
    }
    for (int i = 0; i < lNodeResolutions.lLength; i++)
    {
        ((fpoint*)nodRes_cache)[i] = lNodeResolutions[i];
    }
    // alright, by now taggedInternals have been taken care of, and model has
    // been filled with all of the transition matrices. 

    // time stuff:
    time_t dtimer;
    time_t htimer;
    
    // set and log Global and Local work size dimensions
    
    szLocalWorkSize = roundDoubleUpToNextPowerOfTwo(alphabetDimension);
    szGlobalWorkSize = roundDoubleUpToNextPowerOfTwo(siteCount) *
        roundDoubleUpToNextPowerOfTwo(alphabetDimension);
    localMemorySize = roundDoubleUpToNextPowerOfTwo(alphabetDimension);
    printf("Global Work Size \t\t= %d\nLocal Work Size \t\t= %d\n# of Work Groups \t\t= %d\n\n", 
           szGlobalWorkSize, szLocalWorkSize, 
           (szGlobalWorkSize % szLocalWorkSize + szGlobalWorkSize/szLocalWorkSize)); 
    
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


    // Allocate the OpenCL buffer memory objects for the input and output on the
    // device GMEM
    cmNode_cache = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE,
                    sizeof(clfp)*alphabetDimension*siteCount*nodeCount, NULL,
                    &ciErr1);
    cmModel_cache = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY,
                    sizeof(clfp)*alphabetDimension*alphabetDimension*nodeCount, 
                    NULL, &ciErr2);
    ciErr1 |= ciErr2;
    cmNodRes_cache = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY,
                    sizeof(clfp)*lNodeResolutions.lLength, NULL, &ciErr2);
    ciErr1 |= ciErr2;
    
    print("clCreateBuffer...\n");
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error in clCreateBuffer, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(EXIT_FAILURE);
    }

    
    // Create the program
    // Read the OpenCL kernel in from source file
    // TODO: account for if it is a leaf or not, make sure all of the arguments are passed properly
    // TODO: figure out where the parent result should be written out to. 
    // My options for figuring out the proper parent and child character indices for aquisition and insertion of data are as follows:
    //      -- pass the conversion arrays and use the globalID and localID to figure out the site number offset from the childNodeIndex
    //          and parentNodeIndex offsets respectively, using the localID as the final character offset.
    //      -- the siteNumber is easy, it is the globalID divided by the alphabet dimension. We're padding the alphabet dimension, however, 
    //          so we have to use the roundCharacters variable I pass in. Thus XNodeIndex*sites*characters + (globalID/roundCharacters)*
    //          characters + localID is the corrected character index in the iNodeCache array. 
    // This question brings about a comparison. What are the local and global ID's used for in the toy implementation, and are these 
    //      necessary/advantageous in the real implementation. 
    // So the global ID for a given execution of the kernel is unique for all parent characters in the entire node's analysis. 
    // The local ID is this same parent character's unique ID within a site. This has consequences with character # rounding as well as 
    //      with local caching. 
    // Just as a side note however, I still need the nodeIndex for accessing the correct node's portion of the iNodeCache.
    // Another note, you cannot just pass isLeaf. You have to look it up. Which means putting flatLeaves.lLength on the GPU

    // So dealing with the leaf/not leaf issues. If it is a leaf and the node is ambiguous, lNodeResolutions is a flat array of length
    //      (count(ambiguous nodes in the analysis)*alphabetDimension). So instead of pointing nodeScratch to a section of iNodeCache
    //      we just point it to a section of lNodeResolutions.
    // For non-ambiguous leaves there is no point in looping through the possible child characters for each parent character. Rather
    //      we multiply each parent character by the value in the transition matrix that corresponds to that parent character at the 
    //      child character that has a non-zero value. 
	const char *program_source = "\n" \
	"#pragma OPENCL EXTENSION cl_khr_fp64: enable																			    	\n" \
	"" FLOATPREC                                                                                                                        \
	"__kernel void FirstLoop(__global fpoint* node_cache, __global const fpoint* model, __global const fpoint* nodRes_cache,    	\n" \
    "    __local fpoint* nodeScratch, __local fpoint * modelScratch, int nodes, int sites, int characters,	int childNodeIndex,     \n" \
    "    int parentNodeIndex, int leafLen, int roundCharacters)	                                                                    \n" \
	"{																														    	\n" \
	"   int parentCharGlobal = get_global_id(0); // a unique global ID for each parentcharacter in the whole node's analysis    	\n" \
    "   int parentCharLocal = get_local_id(0); // a local ID unique within the site.										    	\n" \
	"	if ((parentCharGlobal/characters) >= sites) return;	// filter out those parent characters that were added to round the      \n" \
    "   // number of sites to a power of two                                                                                        \n" \
	"	if (parentCharLocal >= characters) return; // that won't catch all the characters, as some were inserted into each site     \n" \
    "   // to round the number of characters up to a power of two.                                                                  \n" \
    "   bool isLeaf = childNodeIndex < leafLen;                                                                                          \n" \
    "   int siteNumber = parentCharGlobal/roundCharacters);                                                                         \n" \
    "   int parentCharacterIndex = parentNodeIndex*sites*characters + siteNumber*characters + parentCharLocal;                      \n" \
    "   int childCharacterIndex = childNodeIndex*sites*characters + siteNumber*characters + parentCharLocal;                        \n" \
    "   if (!isLeaf)
	"       nodeScratch[parentCharLocal] = node_cache[childCharacterIndex];				                            			    	\n" \
    "   else
    "       int siteState = lNodeFlags[childNodeIndex*sites + siteNumber];
    "       nodeScratch[parentCharLocal] = nodRes_cache[characters*(-siteState-1)];
    "   modelScratch[parentCharLocal] = model[childNodeIndex*characters*characters + parentCharLocal * characters + parentCharLocal];	\n" \
	"	barrier(CLK_LOCAL_MEM_FENCE);																						    	\n" \
	"   fpoint sum = 0.;																									    	\n" \
    "   long myChar;																										    	\n" \
    "   for (myChar = 0; myChar < characters; myChar++)																	    		\n" \
    "   {																													     	\n" \
    "       sum += nodeScratch[myChar] * modelScratch[myChar];																    	\n" \
    "   }																														    \n" \
    "   barrier(CLK_LOCAL_MEM_FENCE);																				    			\n" \
	"   parent_cache[parentCharGlobal] *= sum;																		     			\n" \
	"}																													    		\n" \
	"\n";
    
    
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
    
    // Set the Argument values
    // TODO: set the arguments
    // nodeIndex
    // 
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
                sizeof(clfp)*alphabetDimension*siteCount*nodeCount, node_cache, 
                0, NULL, NULL);
    ciErr1 |= clEnqueueWriteBuffer(cqCommandQueue, cmModel_cache, CL_FALSE, 0,
                sizeof(clfp)*alphabetDimension*alphabetDimension*nodeCount,
                model_cache, 0, NULL, NULL);
    // FIXED: fix size of resNod_cache
    ciErr1 |= clEnqueueWriteBuffer(cqCommandQueue, cmNodRes_cache, CL_FALSE, 0,
                sizeof(clfp)*lNodeResolutions, nodRes_cache, 0, NULL, NULL);
    
    printf("clEnqueueWriteBuffer (node_cache, parent_cache and model)...\n"); 
    if (ciErr1 != CL_SUCCESS)
    {
        printf("Error in clEnqueueWriteBuffer, Line %u in file %s !!!\n\n", __LINE__, __FILE__);
        Cleanup(EXIT_FAILURE);
    }
    
    // Launch kernel
    // TODO: node loop, argument setting, launch kernel, manage output
    // TODO: deal with pointer arithmetic. Pass as argument or do with
    // global/localID's?
    
    for (int nodeIndex = 0; nodeIndex < nodeCount; nodeIndex++)
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
		
    }
    
    // Synchronous/blocking read of results, and check accumulated errors
    ciErr1 = clEnqueueReadBuffer(cqCommandQueue, cmNode_cache, CL_TRUE, 0,
            sizeof(clfp)*alphabetDimension*siteCount*nodeCount, node_cache, 0,
            NULL, NULL);
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
    htimer = time(NULL);

    for (int i = 0; i < nodeCount*siteCount*alphabetDimension; i++)
    {
       iNodeCache[i] = node_cache[i];
    }
    
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
    this->parent_results = parent_results;
    this->siteCount = siteCount;
    this->nodeCount = nodeCount;
    this->alphabetDimension = alphabetDimension;
    this->updateNodes = updateNodes;
    this->flatParents = flatParents;
    this->flatNodes = flatNodes;
    this->flatCLeaves = flatCLeaves;
    this->flatTree = flatTree;
    this->iNodeCache = iNodeCache;
    this->lNodeFlags = lNodeFlags;
    this->taggedInternals = taggedInternals;
    this->lNodeResolutions = lNodeResolutions;
     
    

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
