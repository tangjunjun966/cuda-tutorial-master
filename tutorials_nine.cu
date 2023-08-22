


/*!
    @Description : 原子操作篇
    @Author      : tangjun
    @Date        : 2023-8-21
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










extern "C" __global__ void kernel_func_error(int* counter, int* data_0)
{
    // 计算线程号
    unsigned int block_index = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    unsigned int thread_index = block_index * blockDim.x * blockDim.y * blockDim.z + \
        threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;

    // 统计结果
    int value = data_0[thread_index];
    //printf("%d\n", value);
    counter[value] ++;
}

extern "C" __global__ void kernel_func_correct(int* counter, int* data_0)
{
    // 计算线程号
    unsigned int block_index = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    unsigned int thread_index = block_index * blockDim.x * blockDim.y * blockDim.z + \
        threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;

    // 统计结果
    int value = data_0[thread_index];
    atomicAdd(&counter[value], 1);
}




int atomic_apply1() {

    const int N = 32;
    int* gpu_buffer;
    int* host_data = new int[N];

    for (int i = 0; i < N; i++) { 
        if (i%2==0) {
            host_data[i] = 1;
        }
        else {
            host_data[i] = 0;
        }
    
    }

    std::cout << "打印输入数据" << endl;
    for (int i = 0; i < N; i++) { std::cout << host_data[i] << "\t"; }


    cudaMalloc((void**)&gpu_buffer, N * sizeof(int));

    
    cudaMemcpy(gpu_buffer, host_data, N * sizeof(int), cudaMemcpyHostToDevice);
    
    int* count = nullptr;
    cudaMalloc((void**)&count, 2 * sizeof(int));
    int* host_count = nullptr;
    cudaMallocHost((void**)&host_count,2*sizeof(int));
    host_count[0] = 0;
    host_count[1] = 0;
    cudaMemcpy(count, host_count, 2 * sizeof(int), cudaMemcpyHostToDevice);


    //kernel_func_error << <N, 1 >> > (count, gpu_buffer);
    //kernel_func_correct << <N, 1 >> > (count, gpu_buffer);


    auto T0 = std::chrono::system_clock::now();  //时间函数
    int num = 10000;
    int num_k = 60;
 
    for (int k = 0; k < num_k; k++) {
        for (int j = 0; j < num; j++) {
            //cudaMemcpy(count, host_count, 2 * sizeof(int), cudaMemcpyHostToDevice);
            kernel_func_error << <N, 1 >> > (count, gpu_buffer);
            //kernel_func_correct << <N, 1 >> > (count, gpu_buffer);

        }
    
    }
    
    auto T1 = std::chrono::system_clock::now();
    float time_kernel = std::chrono::duration_cast<std::chrono::milliseconds>(T1 - T0).count();

    std::cout << "\n\n推理时间:\t " <<  time_kernel/ num_k << "ms\n\n" << endl;

    cudaMemcpy(host_count, count, 2 * sizeof(int), cudaMemcpyDeviceToHost);
    
    std::cout << "\n打印输出结果\n\n" <<"说明：\t若为偶数，0与1数量相等，否则相差1个数\n\n" << "0的计数量：" << host_count[0]<<"\n1的计数量："<< host_count[1]<< endl;
    

    


    return 0;
}







//十进制转二进制


void printbinary(const unsigned int val)
{
    for (int i = 16; i >= 0; i--)
    {
        if (val & (1 << i))
            cout << "1";
        else
            cout << "0";
    }
}

int atomic_apply2()
{
    int a = 6;
    int b = 4;
    std::cout << "\n打印变量a的二进制\n" << a << ":\t";
    printbinary(a);
    std::cout << "\n打印变量b的二进制\n" <<b << ":\t";
    printbinary(b);

    std::cout << "\n打印与、或、异或结果：" << endl;

    printf("\n与结果：\t%d",a&b);
    printf("\n或结果：\t%d", a | b);

    std::cout << "\n打印与的二进制\n";
    printbinary(a & b);
    std::cout << "\n打印或的二进制\n" ;
    printbinary(a|b);


    return 0;
}


















__global__ void kernel(int* data, int* gpu_output) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // 对共享内存中的数据执行原子加操作
    int index = atomicAdd(&data[tid], 1);
    //printf("%d\n", index);

    if (index > 50) {
        return;
    }
    gpu_output[index] = data[index] * 100;


}




int atomic_apply3() {

    const int N = 50;
    int* gpu_buffer[2];

    int* host_data = new int[N];


    for (int i = 0; i < N; i++) { host_data[i] = i + 6; }
    std::cout << "\n打印初始化" << endl;
    for (int i = 0; i < N; i++) { std::cout << host_data[i] << "\t"; }



    cudaMalloc((void**)&gpu_buffer[0], N * sizeof(int));

    cudaMalloc((void**)&gpu_buffer[1], N * sizeof(int));


    cudaMemcpy(gpu_buffer[0], host_data, N * sizeof(int), cudaMemcpyHostToDevice);
    kernel << <1, N >> > (gpu_buffer[0], gpu_buffer[1]);


    int* cpu_output = new int[N];

    cudaMemcpy(cpu_output, gpu_buffer[1], N * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "\n打印输出结果" << endl;
    for (int i = 0; i < N; i++) { std::cout << cpu_output[i] << "\t"; }



    return 0;
}





__global__ void kernel2(int* data, int* gpu_output, int N) {

    int count = data[0];
    //printf("count:%d\n", count);

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if ((tid + 1) % 2 != 0) { return; } 


    int index = atomicAdd(gpu_output, 1); 

    //printf("index:%d\n ", (index));

    if (index >= N / 2) return;

    gpu_output[index] = data[tid];


}


int atomic_apply4() {
  

    const int N = 50; //使用偶数验证
    int* gpu_buffer[2];

    int* host_data = new int[N];


    for (int i = 0; i < N; i++) { host_data[i] = i + 1; }
    std::cout << "\n打印初始化" << endl;
    for (int i = 0; i < N; i++) { std::cout << host_data[i] << "\t"; }



    cudaMalloc((void**)&gpu_buffer[0], N * sizeof(int));

    cudaMalloc((void**)&gpu_buffer[1], (N / 2) * sizeof(int));


    cudaMemcpy(gpu_buffer[0], host_data, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaStream_t stream;
    cudaStreamCreate(&stream); //stream初始化

    //cudaMemsetAsync(gpu_buffer[1], 0, sizeof(int)  * N/2,stream);
    kernel2 << <1, N, 0, stream >> > (gpu_buffer[0], gpu_buffer[1], N);


    int* cpu_output = new int[N / 2];

    cudaMemcpy(cpu_output, gpu_buffer[1], N / 2 * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "\n打印输出结果" << endl;
    for (int i = 0; i < N / 2; i++) { std::cout << cpu_output[i] << "\t"; }



    return 0;
}







//使用原子操作，计数

__global__ void countValues(float* list, int* count, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        atomicAdd(count, 1);

    }
}

int atomic_apply5()
{
    // 假设你已经有一个包含n个数值的列表list

    const int n = 32;

    float* d_list = nullptr;



    int count = 0;
    cudaMallocHost((void**)&d_list, n * sizeof(float));
    for (int i = 0; i < n; i++) { d_list[i] = i + 1; }
    std::cout << "d_list:" << endl;
    for (int i = 0; i < n; i++) { std::cout << d_list[i] << "\t"; }


    int* d_count = nullptr;
    cudaMalloc((void**)&d_count, sizeof(int));

    cudaMemcpy(d_count, &count, sizeof(int), cudaMemcpyHostToDevice);

    // 定义块和线程的数量
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    // 调用核函数
    countValues << <numBlocks, blockSize >> > (d_list, d_count, n);

    // 将计数器的值从设备端复制回主机端
    cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost);

    // 输出结果
    printf("Number of non-zero values: %d\n", count);

    // 释放内存
    cudaFree(d_list);
    cudaFree(d_count);

    return 0;
}




__global__ void kernel(int* data) {
    int tid = threadIdx.x;
    atomicAnd(&data[tid], 0x0F);
}

int atomic_apply6() {
    int data[16] = { 0 };
    int* d_data;
    cudaMalloc(&d_data, sizeof(int) * 16);
    cudaMemcpy(d_data, data, sizeof(int) * 16, cudaMemcpyHostToDevice);

    kernel << <1, 16 >> > (d_data);

    cudaMemcpy(data, d_data, sizeof(int) * 16, cudaMemcpyDeviceToHost);
    cudaFree(d_data);

    for (int i = 0; i < 16; ++i) {
        printf("%d ", data[i]);
    }
    printf("\n");

    return 0;
}








int  main_nine() {


    //atomic_apply1();
    atomic_apply2();
    //atomic_apply3();
    //atomic_apply4();
    //atomic_apply5();
    //atomic_apply6();
    return 0;
}








