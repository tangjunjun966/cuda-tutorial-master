/*!
    
   我的教程优势:一系列且强逻辑顺序教程，附有源码，实战性很强。

    只有核心cuda处理代码，隐藏在教程中，我将不开源，毕竟已开源很多cuda教程代码，也为本次教程付出很多汗水。

    因此，核心代码于yolo部署cuda代码和整个文字解释教程需要有一定补偿，望理解。

    可以保证，认真学完教程，cuda编程毫无压力。

    详情请链接:http://t.csdn.cn/NaCZ5


    @Description : CUDA kenel计算应用示例篇
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







__global__ void hello_from_gpu()
{
    const int blockid = blockIdx.x;
    const int  threadid = threadIdx.x;
    printf("block index %d and thread idex %d!\n", blockid, threadid);
}
int kernel_apply1(void)
{
    hello_from_gpu << <6, 5 >> > ();
    cudaDeviceSynchronize();
    return 0;
}



__global__ void VecAdd1(int* A, int* B, int* C)
{
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}

int kernel_apply2()
{
    int m = 8;
    int* a, * b, * c;
    //分配host内存
    cudaMallocHost((void**)&a, sizeof(int) * m);
    cudaMallocHost((void**)&b, sizeof(int) * m);
    cudaMallocHost((void**)&c, sizeof(int) * m);

    std::cout << "value of a:" << endl;
    for (int i = 0; i < m; i++) {
        a[i] = rand() % 256;
        std::cout << a[i] << "\t";
    }
    std::cout << "\nvalue of b:" << endl;
    for (int i = 0; i < m; i++) {
        b[i] = rand() % 260;
        std::cout << b[i] << "\t";
    }

    int* g_a, * g_b, * g_c;
    //分配gpu内存
    cudaMalloc((void**)&g_a, sizeof(int) * m);
    cudaMalloc((void**)&g_b, sizeof(int) * m);
    cudaMalloc((void**)&g_c, sizeof(int) * m);
    // 赋值
    cudaMemcpy(g_a, a, sizeof(int) * m, cudaMemcpyHostToDevice);
    cudaMemcpy(g_b, b, sizeof(int) * m, cudaMemcpyHostToDevice);

    dim3 dimGrid(1);
    dim3 dimBlock(m);

    //应用grid只有x方向一个block，block只有x方向m个third
    VecAdd1 << <dimGrid, dimBlock >> > (g_a, g_b, g_c);
    //VecAdd1 << <dim3(1), dim3(m) >> > (g_a, g_b, g_c);
    //VecAdd1 << <1, m >> > (g_a, g_b, g_c);

    //将g_c赋值给c
    cudaMemcpy(c, g_c, sizeof(int) * m, cudaMemcpyDeviceToHost);
    //打印
    std::cout << "\nvalue of c:" << endl;
    for (int i = 0; i < m; i++) {
        std::cout << c[i] << "\t";
    }





    //释放内存
    cudaFree(g_a);
    cudaFree(g_b);
    cudaFree(g_c);
    cudaFreeHost(a);
    cudaFreeHost(b);
    cudaFreeHost(c);

    return 0;
}




__global__ void MatAdd2(int A[8], int B[8], int C[8])
{
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
    printf("\ni=%i", i);

    //std::cout <<"核函数："  << std::endl;
}


int kernel_apply3()
{
    const int m = 8;
    int a[m], b[m], c[m];
    //int *a, *b, *c;
    //int* a, * b, c[m];

    //分配host内存
    cudaMallocHost((void**)&a, sizeof(int) * m * m);
    cudaMallocHost((void**)&b, sizeof(int) * m);
    cudaMallocHost((void**)&c, sizeof(int) * m);

    std::cout << "value of a:" << endl;
    for (int i = 0; i < m; i++) {
        a[i] = rand() % 69;
        std::cout << a[i] << "\t";
    }

    std::cout << "value of b:" << endl;
    for (int j = 0; j < m; j++) {
        b[j] = rand() % 25;
        std::cout << b[j] << "\t";
    }

    int* g_a, * g_b, * g_c;

    //分配gpu内存
    cudaMalloc((void**)&g_a, sizeof(int) * m * m);
    cudaMalloc((void**)&g_b, sizeof(int) * m);
    cudaMalloc((void**)&g_c, sizeof(int) * m);
    // 赋值
    cudaMemcpy(g_a, a, sizeof(int) * m, cudaMemcpyHostToDevice);
    cudaMemcpy(g_b, b, sizeof(int) * m, cudaMemcpyHostToDevice);


    MatAdd2 << <1, m >> > (g_a, g_b, g_c);
    //cudaDeviceSynchronize();

    cudaMemcpy(c, g_c, sizeof(int) * m, cudaMemcpyDeviceToHost);

    std::cout << "value of c:" << endl;
    for (int j = 0; j < m; j++) {
        std::cout << c[j] << "\t";
    }
    return 0;
}











//用于CV读取图片BGR通道将其改为RGB方法
__global__ void rgb2grayincuda(uchar3* const d_in, unsigned char* const d_out,
    uint imgheight, uint imgwidth)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;  //w
    const unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;  //h

    if (idx < imgwidth && idy < imgheight)  //有的线程会跑到图像外面去，不执行即可
    {
        uchar3 rgb = d_in[idy * imgwidth + idx];
        d_out[idy * imgwidth + idx] = 0.299f * rgb.x + 0.587f * rgb.y + 0.114f * rgb.z;

    }
}


void show_img(Mat img) {
    cv::imshow("Image", img);
    cv::waitKey(1000);
    cv::destroyAllWindows();

}


void kernel_apply4() {
    Mat srcImage = imread("image.jpg");
    show_img(srcImage);

    const uint imgheight = srcImage.rows;
    const uint imgwidth = srcImage.cols;

    Mat grayImage(imgheight, imgwidth, CV_8UC1, Scalar(0));



    uchar3* d_in;   //向量类型，3个uchar
    unsigned char* d_out;

    cudaMalloc((void**)&d_in, imgheight * imgwidth * sizeof(uchar3));
    cudaMalloc((void**)&d_out, imgheight * imgwidth * sizeof(unsigned char));


    cudaMemcpy(d_in, srcImage.data, imgheight * imgwidth * sizeof(uchar3), cudaMemcpyHostToDevice);

    //说明：(imgwidth + threadsPerBlock.x - 1) / threadsPerBlock.x表示x方向
    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((imgwidth + threadsPerBlock.x - 1) / threadsPerBlock.x, (imgheight + threadsPerBlock.y - 1) / threadsPerBlock.y);


    //启动内核
    rgb2grayincuda << <blocksPerGrid, threadsPerBlock >> > (d_in, d_out, imgheight, imgwidth);

    //执行内核是一个异步操作，因此需要同步以测量准确时间
    cudaDeviceSynchronize();




    //拷贝回来数据
    cudaMemcpy(grayImage.data, d_out, imgheight * imgwidth * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    //释放显存
    cudaFree(d_in);
    cudaFree(d_out);

    imshow("grayImage", grayImage);
    cv::waitKey(1000);
    cv::destroyAllWindows();


}




typedef struct {
    int width;
    int height;
    float* elements;

}Matrix;









void main_seven()
{

    //kernel_apply1();
    //kernel_apply2();
    //kernel_apply2();
    kernel_apply4();



}









