
/*!
    @Description : CUDA矩阵的加减乘除
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
using namespace cv;
using namespace std;





void Print_2dim(int* ptr, int m, int n) {
    std::cout << "result:\n";
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::cout << "\t" << ptr[i * n + j];
        }
        std::cout << "\n";
    }
}

__global__ void gpu_matrix_plus_thread(int* a, int* b, int* c)
{
    //方法一：通过id方式计算
    //grid为2维度，block为2维度,使用公式id=blocksize * blockid + threadid
    int blocksize = blockDim.x * blockDim.y;
    int blockid = gridDim.x * blockIdx.y + blockIdx.x;
    int threadid = blockDim.x * threadIdx.y + threadIdx.x;
    int id = blocksize * blockid + threadid;

    c[id] = a[id] + b[id];

}


__global__ void gpu_matrix_plus1(int* a, int* b, int* c, int m, int n)
{   //方法二：通过row与col的方式计算-->通过变换列给出id
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    c[row * n + col] = a[row * n + col] + b[row * n + col];
}


__global__ void gpu_matrix_plus2(int* a, int* b, int* c, int m, int n)
{   //方法三：通过row与col的方式计算-->通过变换行给出id
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    c[row + col * m] = a[row + col * m] + b[row + col * m];
}


void init_variables(int* a, int* b, int m, int n) {

    //初始化变量
    std::cout << "value of a:" << endl;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            a[i * n + j] = rand() % 256;
            std::cout << "\t" << a[i * n + j];

        }
        std::cout << "\n";
    }
    std::cout << "value of b:" << endl;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            b[i * n + j] = rand() % 256;
            std::cout << "\t" << b[i * n + j];
        }
        std::cout << "\n";
    }
    std::cout << "value of a+b:" << endl;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {

            std::cout << "\t" << a[i * n + j] + b[i * n + j];
        }
        std::cout << "\n";
    }


}

int kernel_plus()
{
    /*
    matrix a[m,n], matrix b[m,n]
    a[m,n]+b[m,n]=[m,n]
    */

    const int  BLOCK_SIZE = 2;
    int m = 8;    //行
    int n = 10;   //列
    int* a, * b;
    //分配host内存
    cudaMallocHost((void**)&a, sizeof(int) * m * n);
    cudaMallocHost((void**)&b, sizeof(int) * m * n);

    init_variables(a, b, m, n);//随机初始化变量

    int* g_a, * g_b;
    //分配gpu内存
    cudaMalloc((void**)&g_a, sizeof(int) * m * n);
    cudaMalloc((void**)&g_b, sizeof(int) * m * n);
    cudaMemcpy(g_a, a, sizeof(int) * m * n, cudaMemcpyHostToDevice);
    cudaMemcpy(g_b, b, sizeof(int) * m * n, cudaMemcpyHostToDevice);

    unsigned int grid_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;  //行写给y
    unsigned int grid_cols = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;  //列写给x
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);


    std::cout << "gridDIM.x:" << grid_cols << "\tgridDIM.y:" << grid_rows << endl;
    std::cout << "blockDIM.x:" << BLOCK_SIZE << "\tblockDIM.y:" << BLOCK_SIZE << endl;




    int* c1, * g_c;
    cudaMalloc((void**)&g_c, sizeof(int) * m * n);
    cudaMallocHost((void**)&c1, sizeof(int) * m * n);
    gpu_matrix_plus_thread << <dimGrid, dimBlock >> > (g_a, g_b, g_c);
    cudaMemcpy(c1, g_c, sizeof(int) * m * n, cudaMemcpyDeviceToHost);
    Print_2dim(c1, m, n);


    int* c2, * g_c2;
    cudaMallocHost((void**)&c2, sizeof(int) * m * n);
    cudaMalloc((void**)&g_c2, sizeof(int) * m * n);
    gpu_matrix_plus1 << <dimGrid, dimBlock >> > (g_a, g_b, g_c2, m, n);
    cudaMemcpy(c2, g_c2, sizeof(int) * m * n, cudaMemcpyDeviceToHost); //将device端转host端
    Print_2dim(c2, m, n);

    int* c3, * g_c3;
    cudaMallocHost((void**)&c3, sizeof(int) * m * n);
    cudaMalloc((void**)&g_c3, sizeof(int) * m * n);
    gpu_matrix_plus2 << <dimGrid, dimBlock >> > (g_a, g_b, g_c3, m, n);
    cudaMemcpy(c3, g_c3, sizeof(int) * m * n, cudaMemcpyDeviceToHost); //将device端转host端
    Print_2dim(c3, m, n);

    //释放内存
    cudaFree(g_a);
    cudaFree(g_b);
    cudaFree(g_c);
    cudaFreeHost(a);
    cudaFreeHost(b);
    cudaFreeHost(c1);

    return 0;
}


__global__ void gpu_matrix_mult(int* a, int* b, int* c, int m, int n, int k)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; // 行线程 y
    int col = blockIdx.x * blockDim.x + threadIdx.x; // 列线程 x
    int sum = 0;
    if (col < k && row < m) {
        for (int i = 0; i < n; i++) {
            sum += a[row * n + i] * b[i * k + col];  //看出row与col不动的方式计算
        }
        c[row * k + col] = sum;
    }
}

__global__ void gpu_matrix_multiply_thread(int* a, int* b, int* c, int m, int n, int k)
{
    // [m*n]* [n*k]<---m n
    //方法一：通过id方式计算
    //grid为2维度，block为2维度,使用公式id=blocksize * blockid + threadid
    int blocksize = blockDim.x * blockDim.y;
    int blockid = gridDim.x * blockIdx.y + blockIdx.x;
    int threadid = blockDim.x * threadIdx.y + threadIdx.x;
    int id = blocksize * blockid + threadid;

    int row = id / k;
    int col = id % k;
    int sum = 0;

    for (int i = 0; i < n; i++) {
        sum += a[row * n + i] * b[i * k + col];

    }
    c[row * k + col] = sum;


}

int kernel_multiply()
{
    /*
    matrix a[m,n], matrix b[n,k]
    a[m,n]*b[n,k]=[m,k]
    */

    const int  BLOCK_SIZE = 2;
    int m = 8;    //行
    int n = 4;   //中间变量
    int k = 10;    //列
    int* a, * b;

    // 初始化 a与b
    cudaMallocHost((void**)&a, sizeof(int) * m * n);
    cudaMallocHost((void**)&b, sizeof(int) * n * k);

    std::cout << "value of a:" << endl;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            a[i * n + j] = rand() % 6;
            std::cout << "\t" << a[i * n + j];
        }
        std::cout << "\n";
    }

    std::cout << "value of b:" << endl;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < k; j++) {
            b[i * k + j] = rand() % 10;
            std::cout << "\t" << b[i * k + j];
        }
        std::cout << "\n";
    }

    //a*b相乘
    std::cout << "value of a*b:" << endl;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            int tmp = 0;
            for (int h = 0; h < n; h++) {
                tmp += a[i * n + h] * b[h * k + j];
            }
            //c[i * k + j] = tmp;
            std::cout << "\t" << tmp;
        }
        std::cout << "\n";
    }




    int* g_a, * g_b;

    cudaMalloc((void**)&g_a, sizeof(int) * m * n);
    cudaMalloc((void**)&g_b, sizeof(int) * n * k);
    cudaMemcpy(g_a, a, sizeof(int) * m * n, cudaMemcpyHostToDevice);
    cudaMemcpy(g_b, b, sizeof(int) * k * n, cudaMemcpyHostToDevice);
    unsigned int grid_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;  //行写给y
    unsigned int grid_cols = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;  //列写给x
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    std::cout << "gridDIM.x:" << grid_cols << "\tgridDIM.y:" << grid_rows << endl;
    std::cout << "blockDIM.x:" << BLOCK_SIZE << "\tblockDIM.y:" << BLOCK_SIZE << endl;

    // 使用row与col计算
    int* c1, * g_c;
    cudaMalloc((void**)&g_c, sizeof(int) * m * k);
    cudaMallocHost((void**)&c1, sizeof(int) * m * k);
    gpu_matrix_mult << <dimGrid, dimBlock >> > (g_a, g_b, g_c, m, n, k);
    cudaMemcpy(c1, g_c, sizeof(int) * m * k, cudaMemcpyDeviceToHost);
    Print_2dim(c1, m, k);
    cudaFree(g_c);
    cudaFreeHost(c1);

    //使用id计算
    int* c2, * g_c2;
    cudaMalloc((void**)&g_c2, sizeof(int) * m * k);
    cudaMallocHost((void**)&c2, sizeof(int) * m * k);
    gpu_matrix_multiply_thread << <dimGrid, dimBlock >> > (g_a, g_b, g_c2, m, n, k);
    cudaMemcpy(c2, g_c2, sizeof(int) * m * k, cudaMemcpyDeviceToHost);
    Print_2dim(c2, m, k);
    cudaFree(g_c2);
    cudaFreeHost(c2);

    //释放内存
    cudaFree(g_a);
    cudaFree(g_b);
    cudaFreeHost(a);
    cudaFreeHost(b);


    return 0;
}

void main_six() {
    kernel_plus();
    kernel_multiply();

}




