


/*!
    @Description : cuda nms计算方法
    @Author      : tangjun
    @Date        : 2023-08-07
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




#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>

using namespace std;

// 定义矩形框的结构体
struct nms_box {
    float x1, y1, x2, y2;
    float score;
    int cls_id;
};

// 定义CUDA核函数，用于计算两个矩形框之间的IOU值
__device__ float iou(nms_box a, nms_box b)
{
    float x1 = fmaxf(a.x1, b.x1);
    float y1 = fmaxf(a.y1, b.y1);
    float x2 = fminf(a.x2, b.x2);
    float y2 = fminf(a.y2, b.y2);
    float intersection = fmaxf(0.0f, x2 - x1) * fmaxf(0.0f, y2 - y1);
    float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    float union_ = area_a + area_b - intersection;
    return intersection / union_;
}

// 定义CUDA核函数，用于执行NMS算法
__global__ void nms_kernel(nms_box* boxes, int* indices, int* num_indices, float nms_thr)
{
    /*
    boxes:输入nms信息，为结构体
    indices:输入为列表序列，记录所有box，如[0,1,2,3,4,5,...]，后续将不需要会变成-1。
    num_indices:记录有多少个box数量
    float nms_thr:nms的阈值，实际为iou阈值
    */

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= *num_indices) { return; }

    int index = indices[i];

    if (index == -1) { return; }

    nms_box box = boxes[index];


    for (int j = i + 1; j < *num_indices; j++) {
        int other_index = indices[j];
        if (other_index == -1) { continue; }

        nms_box other_box = boxes[other_index];
        float iou_value = iou(box, other_box);
        printf("iou value:%f\n", iou_value);
        if (iou_value > nms_thr) { indices[j] = -1; }

    }
}

vector<nms_box> nms(vector<nms_box> boxes, float threshold)
{
    int num_boxes = boxes.size();

    // 将矩形框转换为CUDA中的Box结构体
    nms_box* d_boxes = nullptr;
    cudaMalloc(&d_boxes, num_boxes * sizeof(nms_box));
    cudaMemcpy(d_boxes, boxes.data(), num_boxes * sizeof(nms_box), cudaMemcpyHostToDevice);




    // 创建一个索引数组，用于标记哪些矩形框应该被保留
    int* d_indices;
    cudaMallocHost(&d_indices, num_boxes * sizeof(int));
    for (int i = 0; i < num_boxes; i++) { d_indices[i] = i; }



    // 在CUDA设备上执行NMS算法
    int num_indices = num_boxes;
    int* d_num_indices = nullptr;
    cudaMalloc(&d_num_indices, sizeof(int));
    cudaMemcpy(d_num_indices, &num_indices, sizeof(int), cudaMemcpyHostToDevice);





    int blockSize = 256;
    int numBlocks = (num_boxes + blockSize - 1) / blockSize;
    nms_kernel << <numBlocks, blockSize >> > (d_boxes, d_indices, d_num_indices, threshold);
    //




    // 将保留的矩形框复制回主机端
    cudaMemcpy(&num_indices, d_num_indices, sizeof(int), cudaMemcpyDeviceToHost);



    int* h_indices = new int[num_indices];

    cudaMemcpy(h_indices, d_indices, num_indices * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "打印需要保存box的索引值:" << endl;
    for (int i = 0; i < num_indices; i++) {
        std::cout << "keep indices:" << h_indices[i] << endl;
    }


    vector<nms_box> kept_boxes(num_indices);
    for (int i = 0; i < num_indices; i++) {
        if (h_indices[i] > -1) {
            kept_boxes[i] = boxes[h_indices[i]];
        }
    }


    // 释放内存
    cudaFree(d_boxes);
    cudaFree(d_indices);
    cudaFree(d_num_indices);
    delete[] h_indices;

    return kept_boxes;
}

int main_nms()
{
    // 创建一组矩形框
    vector<nms_box> boxes = {
        {367.0, 38.0, 677.0, 318.0, 0.9,1},
        {502.0, 38.0, 731.0, 318.0, 0.8,2},
        {303.0, 378.0, 831.0, 1071.0, 0.8,2},
        {193.0, 435.0, 831.0, 931.0, 0.7,3},
        {1039.0, 147.0, 1471.0, 557.0, 0.6,4},
        {1339, 1.0,1571.0, 209.0, 0.5,5}
    };





    // 执行NMS算法
    vector<nms_box> kept_boxes = nms(boxes, 0.2);



    // 输出结果
    cv::Mat image = cv::imread("image.jpg");

    for (nms_box box : kept_boxes) {
        cout << box.x1 << ", " << box.y1 << ", " << box.x2 << ", " << box.y2 << ", " << box.score << endl;

        cv::Point p1(box.x1, box.y1);
        cv::Point p2(box.x2, box.y2);
        cv::rectangle(image, p1, p2, cv::Scalar(0, 255, 0), 4, 1, 0);//矩形的两个顶点，两个顶点都包括在矩形内部
    }

    cv::resize(image, image, cv::Size(600, 400), 0, 0, cv::INTER_NEAREST);

    cv::imshow("www", image);
    cv::waitKey(100000);
    cv::destroyAllWindows();

    return 0;
}




/*

在这个示例中，我们定义了一个名为 iou 的函数，用于计算两个矩形框之间的 IOU（交并比）。然后，我们定义了一个名为 nms_kernel 的核函数，用于执行 NMS 算法。在 nms_kernel 中，我们首先获取当前线程的索引 tid，并获取该线程对应的矩形框 box。然后，我们遍历所有矩形框，并计算当前矩形框与其他矩形框之间的 IOU 值。如果 IOU 值大于阈值 iou_threshold，则将该矩形框标记为不保留。最后，我们将结果存储在 indices 中。

在 nms 函数中，我们计算适当的块和网格大小，并调用 nms_kernel 核函数。

请注意，这个示例只是一个简单的实现，并且可能不适用于所有情况。在实际应用中，您可能需要根据具体情况进行修改和优化。
*/














