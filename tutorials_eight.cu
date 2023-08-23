

/*
    
    我的教程优势:一系列且强逻辑顺序教程，附有源码，实战性很强。

    只有核心cuda处理代码，隐藏在教程中，我将不开源，毕竟已开源很多cuda教程代码，也为本次教程付出很多汗水。

    因此，核心代码于yolo部署cuda代码和整个文字解释教程需要有一定补偿，望理解。

    可以保证，认真学完教程，cuda编程毫无压力。

    详情请链接:http://t.csdn.cn/NaCZ5

    @Description : 内存应用篇
    @Author      : tangjun
    @Date        :
*/


#include <iostream>
#include <time.h>
#include "opencv2/highgui.hpp"  //实际上在/usr/include下
#include "opencv2/opencv.hpp"
#include "device_launch_parameters.h"
#include <cuda_runtime_api.h>
#include <device_functions.h>
#include <texture_fetch_functions.h>
using namespace cv;
using namespace std;



#define checkRuntime(op)  __check_cuda_runtime((op), #op, __FILE__, __LINE__)











int inquire_GPU_info() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    int dev;
    for (dev = 0; dev < deviceCount; dev++)
    {
        int driver_version(0), runtime_version(0);
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
        if (dev == 0)
            if (deviceProp.minor = 9999 && deviceProp.major == 9999)
                printf("\n");
        printf("\nDevice%d:\"%s\"\n", dev, deviceProp.name);
        cudaDriverGetVersion(&driver_version);
        printf("CUDA驱动版本:                                         %d.%d\n", driver_version / 1000, (driver_version % 1000) / 10);
        cudaRuntimeGetVersion(&runtime_version);
        printf("CUDA运行时版本:                                       %d.%d\n", runtime_version / 1000, (runtime_version % 1000) / 10);
        printf("设备计算能力:                                         %d.%d\n", deviceProp.major, deviceProp.minor);
        printf("设备全局内存总量 Global Memory:                       %u M\n", deviceProp.totalGlobalMem / (1024 * 1024));
        printf("Number of SMs:                                        %d\n", deviceProp.multiProcessorCount);
        printf("常量内存 Constant Memory:                             %u K\n", deviceProp.totalConstMem / 1024);
        printf("每个block的共享内存 Shared Memory:                    %u K\n", deviceProp.sharedMemPerBlock / 1024);
        printf("每个block的寄存器 registers :                         %d\n", deviceProp.regsPerBlock);
        printf("线程束Warp size:                                      %d\n", deviceProp.warpSize);
        printf("每个SM的最大线程数 threads per SM:                    %d\n", deviceProp.maxThreadsPerMultiProcessor);
        printf("每个block的最大线程数 threads per block:              %d\n", deviceProp.maxThreadsPerBlock);
        printf("每个block的最大维度 each dimension of a block:        %d x %d x %d\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
        printf("每个grid的最大维度 dimension of a grid:               %d x %d x %d\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
        printf("Maximum memory pitch:                                 %u bytes\n", deviceProp.memPitch);
        printf("Texture alignmemt:                                    %u bytes\n", deviceProp.texturePitchAlignment);
        printf("Clock rate:                                           %.2f GHz\n", deviceProp.clockRate * 1e-6f);
        printf("Memory Clock rate:                                    %.0f MHz\n", deviceProp.memoryClockRate * 1e-3f);
        printf("Memory Bus Width:                                     %d-bit\n", deviceProp.memoryBusWidth);
    }

    return 0;
}


bool __check_cuda_runtime(cudaError_t code, const char* op, const char* file, int line)
{
    if (code != cudaSuccess)
    {
        const char* err_name = cudaGetErrorName(code);
        const char* err_message = cudaGetErrorString(code);
        printf("runtime error %s:%d  %s failed. \n  code = %s, message = %s\n", file, line, op, err_name, err_message);
        return false;
    }
    return true;
}

__global__  void show_value(float* v) {
    int idx = threadIdx.x;
    if (idx < 10) {
        printf("value_%d: \t%.2f\n", idx, v[idx]);  //只读取前10个数
    }
}

int memory_appy1()
{

    int device_id = 0;
    checkRuntime(cudaSetDevice(device_id));
    std::cout << "设置gpu id为：\t" << device_id << std::endl;


    std::cout << "设置全局内存" << std::endl;
    float* memory_device = nullptr; // Global Memory
    cudaMalloc((void**)&memory_device, 100 * sizeof(float)); // pointer to device


    std::cout << "设置new(malloc)可分页内存" << std::endl;
    float* memory_host = new float[100]; // Pageable Memory
    for (int i = 0; i < 100; i++) { memory_host[i] = i * 100; }
    checkRuntime(cudaMemcpy(memory_device, memory_host, sizeof(float) * 100, cudaMemcpyHostToDevice)); // 返回的地址是开辟的device地址，存放在memory_device
    show_value << <dim3(1), dim3(100) >> > (memory_device);




    std::cout << "设置页锁定内存" << std::endl;
    float* memory_page_locked = nullptr; // Pinned Memory
    checkRuntime(cudaMallocHost((void**)&memory_page_locked, 100 * sizeof(float))); // 返回的地址是被开辟的pin memory的地址，存放在memory_page_locked
    checkRuntime(cudaMemcpy(memory_page_locked, memory_device, sizeof(float) * 100, cudaMemcpyDeviceToHost)); // 将其返回host内存




    //printf("%f\n", memory_page_locked[2]);
    checkRuntime(cudaFreeHost(memory_page_locked));
    delete[] memory_host;
    checkRuntime(cudaFree(memory_device));

    return 0;
}



__device__ float dev_array[10];
__global__ void my_device_function(float* ptr) {
    // Use the device variable
    int idx = threadIdx.x;

    // Do something with the value
    ptr[idx] = dev_array[idx] + 0.9f;

}

void memory_appy2() {
    // Allocate memory on the device
    float* dev_ptr;
    cudaMalloc((void**)&dev_ptr, 10 * sizeof(float));

    // Copy data from host to device
    float host_array[10] = { 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f };
    cudaMemcpyToSymbol(dev_array, host_array, 10 * sizeof(float)); //赋值

    // Call a device function that uses the device variable
    my_device_function << <1, 10 >> > (dev_ptr);

    float* host_ptr = nullptr;
    cudaMallocHost((void**)&host_ptr, sizeof(int) * 10);
    cudaMemcpy(host_ptr, dev_ptr, sizeof(float) * 10, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 10; i++) { std::cout << host_ptr[i] << endl; }

    // Free memory on the device
    cudaFree(dev_ptr);
}







__global__ void static_sharedMemKernel(int* input, int* output)
{
    // 定义共享内存
    __shared__ int sharedMem[1024];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // 将数据从全局内存拷贝到共享内存
    sharedMem[tid] = input[i];
    // 等待所有线程都将数据拷贝到共享内存中
    __syncthreads();
    // 对共享内存中的数据进行处理
    sharedMem[tid] *= 2;
    sharedMem[tid] = sharedMem[tid] + 30;
    // 等待所有线程都完成数据处理
    __syncthreads();
    // 将结果从共享内存拷贝回全局内存
    output[i] = sharedMem[tid];
}


__global__ void dynamic_sharedMemKernel(int* input, int* output)
{
    // 定义共享内存
    extern __shared__ int sharedMem[];   //使用extern非常重要
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // 将数据从全局内存拷贝到共享内存
    sharedMem[tid] = input[i];
    // 等待所有线程都将数据拷贝到共享内存中
    __syncthreads();
    // 对共享内存中的数据进行处理
    sharedMem[tid] *= 2;
    sharedMem[tid] = sharedMem[tid] + 30;
    // 等待所有线程都完成数据处理
    __syncthreads();
    // 将结果从共享内存拷贝回全局内存
    output[i] = sharedMem[tid];
}







int memory_appy3()
{
    const int n = 9900;
    const int thrid = 1024;
    int input[n];
    int output[n];
    int* d_input, * d_output;
    for (int i = 0; i < n; i++) { input[i] = i; }
    // 分配设备内存
    cudaMalloc((void**)&d_input, n * sizeof(int));
    cudaMalloc((void**)&d_output, n * sizeof(int));
    // 将输入数据拷贝到设备内存中
    cudaMemcpy(d_input, input, n * sizeof(int), cudaMemcpyHostToDevice);

    // 调用核函数
    unsigned int grid = (n + thrid - 1) / thrid;  //列写给x
    dim3 gridperblock(grid);

    auto start = std::chrono::system_clock::now();  //时间函数
    static_sharedMemKernel << <gridperblock, thrid >> > (d_input, d_output);
    auto end = std::chrono::system_clock::now();
    float time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    cout << "\n计算时间:" << time << endl;

    // 将结果拷贝回主机内存中
    cudaMemcpy(output, d_output, n * sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "输出静态共享内存结果output前5位数:" << endl;
    for (int i = 0; i < 5; i++) { cout << output[i] << endl; }

    dynamic_sharedMemKernel << <gridperblock, thrid, 1024 * sizeof(int) >> > (d_input, d_output);
    cudaMemcpy(output, d_output, n * sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "输出动态共享内存结果output前5位数:" << endl;
    for (int i = 0; i < 5; i++) { cout << output[i] << endl; }

    // 释放设备内存
    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}




//声明纹理，用来绑定纹理，其实也就是个纹理标识  
texture<unsigned int, 1, cudaReadModeElementType> texone;


//核心代码，在gpu端执行的kernel，  
__global__ void Textureone(unsigned int* listTarget, int size)
{
    unsigned int texvalue = 0;
    int index = blockIdx.x * blockDim.x + threadIdx.x; //通过线程ID得到数组下标 
    if (index < size)
        texvalue = tex1Dfetch(texone, index) * 100; //通过索引获得纹理值再乘100 
    listTarget[index] = texvalue;
}


void memory_appy4_one()
{
    const int _length = 100;
    unsigned int* listSource = new unsigned int[_length];
    unsigned int* listTarget = new unsigned int[_length];

    //赋值  
    for (int i = 0; i < _length; i++) { listSource[i] = i; }

    unsigned int* dev_Source;
    unsigned int* dev_Target;

    //在设备上申请显存空间  
    cudaMalloc((void**)&dev_Source, _length * sizeof(unsigned int));
    cudaMalloc((void**)&dev_Target, _length * sizeof(unsigned int));
    //将host端的数据拷贝到device端  
    cudaMemcpy(dev_Source, listSource, _length * sizeof(unsigned int), cudaMemcpyHostToDevice);


    //绑定纹理，绑定的纹理标识对应的数据   
    cudaBindTexture(0, texone, dev_Source);

    //调用kernel  
    Textureone << < ceil(_length / 10), 10 >> > (dev_Target, _length);

    //将结果拷贝到host端 ☆host就是CPU  
    cudaMemcpy(listTarget, dev_Target, _length * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    //取消绑定  
    cudaUnbindTexture(texone);

    //释放内存空间  
    cudaFree(dev_Source);
    cudaFree(dev_Target);


    cout << "原始数据： " << endl;
    for (int i = 0; i < _length; i++) { cout << listSource[i] << " "; }

    cout << endl << endl << "运算结果： " << endl;
    for (int i = 0; i < _length; i++) { cout << listTarget[i] << " "; }
    getchar();
}






texture<uchar4, 2, cudaReadModeElementType> textwo;


__global__ void my_kernel(uchar3* output, int width, int height)
{
    uchar3 img_v;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    //float u = (float)x / (float)width;
    //float v = (float)y / (float)height;

    uchar4 value = tex2D(textwo, x, y);

    uchar4 swapped_value = make_uchar4(value.z, value.y, value.x, value.w);
    img_v.x = value.x;
    img_v.y = value.y;
    img_v.z = value.z;

    //printf("\n%.2f", (float)value.y);
    output[x + y * width] = img_v;

}


void show(Mat img, string name = "image") {
    cv::imshow(name, img);
    cv::waitKey(1000);
    cv::destroyAllWindows();

}

int memory_appy4_two()
{
    // 读取图像数据
    cv::Mat image = cv::imread("image.jpg", cv::IMREAD_COLOR);
    show(image, "img_ori");



    // 申请二维纹理内存
    cudaArray* cuArray;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
    cudaMallocArray(&cuArray, &channelDesc, image.cols, image.rows);
    cudaMemcpy2DToArray(cuArray, 0, 0, image.data, image.step, image.cols * sizeof(uchar4), image.rows, cudaMemcpyHostToDevice);

    // 绑定纹理对象和二维纹理内存
    cudaBindTextureToArray(textwo, cuArray);

    // 调用核函数
    dim3 block(16, 16);
    dim3 grid((image.cols + block.x - 1) / block.x, (image.rows + block.y - 1) / block.y);

    cv::Mat outimg = Mat(Size(image.cols, image.rows), CV_8UC1);
    uchar3* output;
    cudaMallocHost(&output, image.cols * image.rows * sizeof(uchar3));

    //cudaMallocHost(&outimg.data, image.cols * image.rows * sizeof(uchar3));
    my_kernel << <grid, block >> > (output, image.cols, image.rows);

    //cudaMemcpy(outimg.data, output, image.cols * image.rows * sizeof(uchar4), cudaMemcpyDeviceToHost);

    // 解绑纹理对象和二维纹理内存
    cudaUnbindTexture(textwo);




    //show(outimg,"outimg");

    // 释放内存
    cudaFree(output);
    cudaFreeArray(cuArray);
    cout << "ok" << endl;
    return 0;
}










texture<float, 2, cudaReadModeElementType> texRef;

__global__ void kernel(float* output, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) { output[y * width + x] = tex2D(texRef, x, y); }
}

int memory_appy5()
{
    int width = 512;
    int height = 512;
    int size = width * height * sizeof(float);

    float* input = (float*)malloc(size);
    float* output = (float*)malloc(size);

    // 初始化输入数据
    for (int i = 0; i < width * height; i++) { input[i] = (float)i; }

    // 定义CUDA数组
    cudaArray* cuArray;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaMallocArray(&cuArray, &channelDesc, width, height);

    // 将输入数据拷贝到CUDA数组中
    cudaMemcpyToArray(cuArray, 0, 0, input, size, cudaMemcpyHostToDevice);

    // 设置纹理内存参数
    texRef.addressMode[0] = cudaAddressModeWrap;
    texRef.addressMode[1] = cudaAddressModeWrap;
    texRef.filterMode = cudaFilterModeLinear;
    texRef.normalized = false;

    // 绑定纹理内存到CUDA数组
    cudaBindTextureToArray(texRef, cuArray);

    // 定义CUDA核函数的线程块和线程格
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    // 调用CUDA核函数
    kernel << <grid, block >> > (output, width, height);

    // 将输出数据从设备拷贝到主机
    cudaMemcpy(output, output, size, cudaMemcpyDeviceToHost);

    // 输出结果
    for (int i = 0; i < width * height; i++)
    {
        printf("%f ", output[i]);
    }

    // 解绑纹理内存
    cudaUnbindTexture(texRef);

    // 释放CUDA数组和内存
    cudaFreeArray(cuArray);
    free(input);
    free(output);

    return 0;
}











void main_eight() {
    inquire_GPU_info();
    memory_appy1();
    memory_appy2();
    memory_appy3();
    memory_appy4_one();
    memory_appy4_two();



}








