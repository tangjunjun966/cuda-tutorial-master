

/*!
    我的教程优势:一系列且强逻辑顺序教程，附有源码，实战性很强。

    只有核心cuda处理代码，隐藏在教程中，我将不开源，毕竟已开源很多cuda教程代码，也为本次教程付出很多汗水。

    因此，核心代码于yolo部署cuda代码和整个文字解释教程需要有一定补偿，望理解。

    可以保证，认真学完教程，cuda编程毫无压力。

    详情请链接:http://t.csdn.cn/NaCZ5


 
    @Description : CUDA函数基础篇
    @Author      : tangjun
    @Date        :
*/



#include <iostream>
#include <time.h>
#include "device_launch_parameters.h"
#include <cuda_runtime_api.h>
#include <device_functions.h>

using namespace std;







/*************************************第四节-CUDA函数基础篇**********************************************/

float sigmoid_host(float x) {
    float y = 1 / (1 + exp(-x));
    return y;
}

__device__  float sigmoid(float x) {
    float y = 1 / (1 + exp(-x));
    //float y = sigmoid_host(x);
    return y;
}

__global__ void test_kernel(float* a, float* c) {

    int idx = threadIdx.x;
    c[idx] = sigmoid(a[idx]); //正确方式
    //c[idx] = sigmoid_host(a[idx]);//绝对错误，无法调用，即：global函数无法调用host函数，只能调用devices函数


}

void Print_dim(float* ptr, int N) {
    for (int i = 0; i < N; i++)
    {
        std::cout << "value:\t" << ptr[i] << std::endl;

    }
}

void init_variables_float(float* a, int m, int n) {

    //初始化变量
    std::cout << "value of a:" << endl;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            a[i * n + j] = rand() / 4089;
            std::cout << "\t" << a[i * n + j];

        }
        std::cout << "\n";
    }


}

void global2device() {
    const int m = 4;
    const int n = 2;
    //分配host内存
    float* a, * c;
    cudaMallocHost((void**)&a, sizeof(float) * m * n);
    cudaMallocHost((void**)&c, sizeof(float) * m * n);
    //变量初始化
    init_variables_float(a, m, n);
    // 分配gpu内存并将host值复制到gpu变量中
    float* g_a;
    cudaMalloc((void**)&g_a, sizeof(float) * m * n);
    cudaMemcpy(g_a, a, sizeof(float) * m * n, cudaMemcpyHostToDevice);
    float* g_c;
    cudaMalloc((void**)&g_c, sizeof(float) * m * n);
    test_kernel << <dim3(1), dim3(m * n), 0, nullptr >> > (g_a, g_c);
    cudaMemcpy(c, g_c, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
    Print_dim(c, m * n);

}

__device__ __host__  float sigmoid_device_host(float x) {
    float y = 1 / (1 + exp(-x));
    return y;
}

void host2device() {
    float y = sigmoid_device_host(1.25);
    std::cout << y << endl;
    std::cout << "success：host calling  device+host  " << endl;
    //以下执行失败   
    try {
        float y = sigmoid_host(1.25);
        throw std::runtime_error("error: fail");
    }
    catch (std::runtime_error err) {
        std::cout << "fail：host calling device" << endl;

    }

}


void main_four() {

    global2device();//host<--global<--device
    host2device();



}






