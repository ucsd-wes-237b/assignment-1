#include <stdio.h>
#include <stdlib.h>

#include "device.h"
#include "kernel.h"
#include "matrix.h"

#define CHECK_ERR(err, msg)                           \
    if (err != CL_SUCCESS)                            \
    {                                                 \
        fprintf(stderr, "%s failed: %d\n", msg, err); \
        exit(EXIT_FAILURE);                           \
    }

#define KERNEL_PATH "kernel.cl"

int main(int argc, char *argv[])
{
    if (argc != 5)
    {
        fprintf(stderr, "Usage: %s <input_file_0> <input_file_1> <answer_file> <output_file>\n", argv[0]);
        return -1;
    }

    const char *input_file_a = argv[1];
    const char *input_file_b = argv[2];
    const char *input_file_c = argv[3];
    const char *input_file_d = argv[4];

    // Load external OpenCL kernel code
    char *kernel_source = OclLoadKernel(KERNEL_PATH); // Load kernel source

    // Host input and output vectors and sizes
    Matrix host_a, host_b, host_c, answer;

    // Device input and output buffers
    cl_mem device_a, device_b, device_c;

    size_t global_item_size, local_item_size;
    cl_int err;

    cl_device_id device_id;    // device ID
    cl_context context;        // context
    cl_command_queue queue;    // command queue
    cl_program program;        // program
    cl_kernel kernel;          // kernel

    err = LoadMatrix(input_file_a, &host_a);
    CHECK_ERR(err, "LoadMatrix");printf("...LoadMatrix input_file_a\n");

    err = LoadMatrix(input_file_b, &host_b);
    CHECK_ERR(err, "LoadMatrix");printf("...LoadMatrix input_file_b\n");
  
    err = LoadMatrix(input_file_c, &answer);
    CHECK_ERR(err, "LoadMatrix");

    // Allocate the memory for the target.
    host_c.shape[0] = host_a.shape[0];
    host_c.shape[1] = host_a.shape[1];
    host_c.data = (float *)malloc(sizeof(float) * host_c.shape[0] * host_c.shape[1]);

    // Find platforms and devices
    OclPlatformProp *platforms = NULL;
    cl_uint num_platforms;

    err = OclFindPlatforms((const OclPlatformProp **)&platforms, &num_platforms);
    CHECK_ERR(err, "OclFindPlatforms");

    // Get ID for first device on first platform
    device_id = platforms[0].devices[0].device_id;

    // Create a context                                         -OK
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    CHECK_ERR(err, "clCreateContext");

    // Create a command queue                                   -OK
    queue = clCreateCommandQueueWithProperties(context, device_id, 0, &err);
    CHECK_ERR(err, "clCreateCommandQueueWithProperties");

    // Create the program from the source buffer                ~OK
    program = clCreateProgramWithSource(context, 1, (const char **)&kernel_source, NULL, &err);
    CHECK_ERR(err, "clCreateProgramWithSource");

    // Build the program executable                              -OK
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    CHECK_ERR(err, "clBuildProgram");

    // Create the compute kernel in the program we wish to run   -OK
    kernel = clCreateKernel(program, "vectorAdd", &err);
    CHECK_ERR(err, "clCreateKernel");

    //@@ Allocate GPU memory here
    unsigned int ARRAY_SIZE;
    ARRAY_SIZE = host_a.shape[0] * host_a.shape[1];printf("...ARRAY_SIZE = %u\n ",ARRAY_SIZE);
    device_a = clCreateBuffer(context, CL_MEM_WRITE_ONLY, ARRAY_SIZE*sizeof(float), NULL, &err);
    device_b = clCreateBuffer(context, CL_MEM_WRITE_ONLY, ARRAY_SIZE*sizeof(float), NULL, &err);
    device_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, ARRAY_SIZE*sizeof(float), NULL, &err);

    //@@ Copy memory to the GPU here
        err = clEnqueueWriteBuffer(queue, device_a, CL_TRUE, 0, ARRAY_SIZE*sizeof(float), host_a.data, 0, NULL, NULL);//host_a, 0, NULL, NULL);
        err = clEnqueueWriteBuffer(queue, device_b, CL_TRUE, 0, ARRAY_SIZE*sizeof(float), host_b.data, 0, NULL, NULL);
    
    // Set the arguments to our compute kernel
        //unsigned int size_a = (unsigned int)host_a.shape[0] * (unsigned int)host_a.shape[1];
        unsigned int size_a = host_a.shape[0] * host_a.shape[1];
        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&device_a);  //~err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&device_a);
        CHECK_ERR(err, "clSetKernelArg 0");
        err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&device_b);
        CHECK_ERR(err, "clSetKernelArg 1");
        err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&device_c);
        CHECK_ERR(err, "clSetKernelArg 2");
        err |= clSetKernelArg(kernel, 3, sizeof(unsigned int), &size_a);
        CHECK_ERR(err, "clSetKernelArg 3");

    //@@ Initialize the global size and local size here
        global_item_size = size_a;
        local_item_size = 1;

    //@@ Launch the GPU Kernel here
        err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL); 
        clFinish(queue);

    //@@ Copy the GPU memory back to the CPU here    
     clEnqueueReadBuffer(queue, device_c, CL_TRUE, 0, size_a*sizeof(float), host_c.data, 0, NULL, NULL);//~host_c, 0, NULL, NULL);

    // Prints the results
    printf("...Ready to Prints the results  [%u, %u] \n", (unsigned int)host_c.shape[0], (unsigned int)host_c.shape[1]);
    for (unsigned int i = 0; i < host_c.shape[0] * host_c.shape[1]; i++)
    {
        printf("C[%u]: %f == %f\n", i, host_c.data[i], host_a.data[i] + host_b.data[i]);
    }
    printf("...Prints the results...Done\n");

    CheckMatrix(&answer, &host_c);
    // Save the result
    err = SaveMatrix(input_file_d, &host_c);
    CHECK_ERR(err, "SaveMatrix");
  
    //@@ Free the GPU memory here
        cl_int ret;
        clFlush(queue);
        clFinish(queue);
        clReleaseCommandQueue(queue);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseContext(context);
      
    // Release host memory
        free(host_a.data);
        free(host_b.data);
        free(host_c.data);
        free(kernel_source);
        free(platforms);
  
    return 0;
}