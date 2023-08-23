/*!
    
   我的教程优势:一系列且强逻辑顺序教程，附有源码，实战性很强。

    只有核心cuda处理代码，隐藏在教程中，我将不开源，毕竟已开源很多cuda教程代码，也为本次教程付出很多汗水。

    因此，核心代码于yolo部署cuda代码和整个文字解释教程需要有一定补偿，望理解。

    可以保证，认真学完教程，cuda编程毫无压力。

    详情请链接:http://t.csdn.cn/NaCZ5


    
    
    @Description : 模拟yolo输出整个cuda的后处理-->最终为box、conf、cls_id，本部分只给出了框架。
    @Author      : tangjun
    @Date        : 2023-08-10
*/



#include <iostream>
#include <time.h>
#include "opencv2/highgui.hpp"  //实际上在/usr/include下
#include "opencv2/opencv.hpp"
#include "device_launch_parameters.h"
#include <cuda_runtime_api.h>
#include <device_functions.h>
#include <texture_fetch_functions.h>
#include<math.h>
using namespace cv;
using namespace std;


#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>



struct nms_box {
    float x1, y1, x2, y2;
    float score;
    int cls_id;
};















void imitate_yolo_part_postprocess() {
    /*

    该代码模拟yolo后处理方法

    */

    float conf_thr = 0.3;
    float nms_thr = 0.1;  //iou>nms_thr则排除
    const int max_object = 6;


    int cls_num = 3;

    int anchor_output_num = 21; //模型输出目标数量，类似25200
    int N_obj = anchor_output_num * (cls_num + 5);//模型输出x,y,w,h+conf+cls_num




    /***********************#########################模仿构建yolo输出结果########################***********************/
    float* input_data = nullptr;
    cudaMallocHost((void**)&input_data, sizeof(float) * N_obj);

    //赋值
    for (int i = 0; i < N_obj; i++) {
        //input_data[i] = (float)(i+1) ;
        float value = rand() / float(RAND_MAX);
        input_data[i] = round(value * 10000) / 10000;
    }
    //更新赋值
    for (int i = 0; i < 6; i++) {
        int idx = i * (cls_num + 5);
        if (idx == 0) {
            input_data[idx] = 367.0;
            input_data[idx + 1] = 38.0;
            input_data[idx + 2] = 677.0;
            input_data[idx + 3] = 318.0;
            input_data[idx + 4] = 1.0;
        }
        else if (idx == 1 * (cls_num + 5)) {
            input_data[idx] = 502.0;
            input_data[idx + 1] = 38.0;
            input_data[idx + 2] = 731.0;
            input_data[idx + 3] = 318.0;
            input_data[idx + 4] = 1.0;

        }
        else if (idx == 2 * (cls_num + 5)) {
            input_data[idx] = 303.0;
            input_data[idx + 1] = 377.0;
            input_data[idx + 2] = 831.0;
            input_data[idx + 3] = 1071.0;
            input_data[idx + 4] = 1.0;

        }
        else if (idx == 3 * (cls_num + 5)) {
            input_data[idx] = 193.0;
            input_data[idx + 1] = 435.0;
            input_data[idx + 2] = 831.0;
            input_data[idx + 3] = 931.0;
            input_data[idx + 4] = 1.0;

        }
        else if (idx == 4 * (cls_num + 5)) {
            input_data[idx] = 1039.0;
            input_data[idx + 1] = 147.0;
            input_data[idx + 2] = 1471.0;
            input_data[idx + 3] = 557.0;
            input_data[idx + 4] = 1.0;

        }
        else if (idx == 5 * (cls_num + 5)) {
            input_data[idx] = 1339.0;
            input_data[idx + 1] = 1.0;
            input_data[idx + 2] = 1571.0;
            input_data[idx + 3] = 209.0;
            input_data[idx + 4] = 1.0;

        }



    }

    //打印显示
    std::cout << "原始数据赋值+打印:" << endl;
    for (int i = 0; i < N_obj; i++) {

        if (i % (cls_num + 5) == 0) { std::cout << endl; }
        std::cout << input_data[i] << "\t\t";

    }

    float* gpu_input = nullptr;
    cudaMalloc((void**)&gpu_input, sizeof(float) * N_obj);
    cudaMemcpy(gpu_input, input_data, sizeof(float) * N_obj, cudaMemcpyHostToDevice);

    /***********************#########################模仿构建yolo输出结果########################***********************/





     /***********************#########################cuda相关变量和内存分配-初始化########################***********************/

    float* gpu_output = nullptr;
    cudaMalloc((void**)&gpu_output, sizeof(float) * max_object * 6);// 保存处理后的yolo输出结果，格式为[max_boject,   [x1,y1,x2,y2,conf,cls_id]]

    nms_box* d_boxes = nullptr;
    cudaMalloc(&d_boxes, anchor_output_num * sizeof(nms_box)); // gpu设备保存，gpu_output数据纯粹格式转换为nms_box结构体的格式
    nms_box* h_boxes = nullptr;
    cudaMallocHost(&h_boxes, anchor_output_num * sizeof(nms_box)); //同理，host端保存，


    int* h_nms_indices_init;
    cudaMallocHost(&h_nms_indices_init, max_object * sizeof(int)); //nms处理的索引赋值初始化变量，恒定不变
    for (int i = 0; i < max_object; i++) { h_nms_indices_init[i] = i; } //赋值操作

    int* d_nms_indices;
    cudaMalloc(&d_nms_indices, max_object * sizeof(int)); //gpu设备处理后索引值，-1为排除目标，>-1为保存目标，需h_nms_indices_init为为其赋值
    int* h_nms_indices;
    cudaMallocHost(&h_nms_indices, max_object * sizeof(int)); //host端保存d_nms_indices赋值，以此决定保留nms目标


    int h_count = 0; //host端记录gpu_output保存有效obj数量，值来源d_conut
    int* d_count = nullptr;
    cudaMalloc((void**)&d_count, sizeof(int)); //gpu端通过原子操作，记录gpu_output保存有效obj数量


    /***********************#########################cuda相关变量和内存分配-初始化########################***********************/



    const int block = 32;



    /*************************************************开始cuda计算***********************************************/

    cudaStream_t stream;
    cudaStreamCreate(&stream);



    h_count = 0;
    cudaMemcpy(d_count, &h_count, sizeof(int), cudaMemcpyHostToDevice);  //初始化记录有效变量d_count与h_count
    int grid = (anchor_output_num + block - 1) / block;
    //调用yolo输出结果处理的核函数
    //decode_yolo_kernel << < grid, block, 0, stream >> > (gpu_input, gpu_output, max_object, cls_num, conf_thr, d_count);
    

    cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    if (h_count > max_object) { h_count = max_object; };




    /****************************************打印模型输出输出数据结果--》通过置信度已过滤不满足要求和给出类别**********************************/

    float* host_decode = nullptr; // 保存gpu处理的变量 
    cudaMallocHost((void**)&host_decode, sizeof(float) * max_object * 6);
    cudaMemcpy(host_decode, gpu_output, sizeof(float) * max_object * 6, cudaMemcpyDeviceToHost);
    std::cout << "\n\n打印输出结果-gpu_output\n" << endl;
    if (h_count == 0) { std::cout << "\n无检测结果" << endl; }
    for (int i = 0; i < h_count; i++) {
        int idx = i * 6;
        std::cout << "x1:" << host_decode[idx] << "\ty1:" << host_decode[idx + 1] << "\tx2:" << host_decode[idx + 2]
            << "\ty2:" << host_decode[idx + 3] << "\tconf:" << host_decode[idx + 4] << "\tclass_id:" << host_decode[idx + 5] << endl;

    }
    /******************************************************************************************************************************/




    //这里是我对decode_yolo_kernel核函数处理后的数据进一步转换格式，在gpu中完成
    int grid_max = (max_object + block - 1) / block;
    //data_format_convert << < grid_max, block, 0, stream >> > (d_boxes, gpu_output, h_count); // gpu_output格式为[x1,y1,conf,cls_id]






    /****************************************将数据转换为带有nms_box格式数据******************************************************/
    nms_box* h_boxes_format = nullptr;
    cudaMallocHost(&h_boxes_format, anchor_output_num * sizeof(nms_box));
    cudaMemcpy(h_boxes_format, d_boxes, anchor_output_num * sizeof(nms_box), cudaMemcpyDeviceToHost);
    std::cout << "\n\n打印格式转换输出-h_boxes_format\n" << endl;
    if (h_count == 0) { std::cout << "\n无检测结果" << endl; }
    for (int i = 0; i < h_count; i++) {
        nms_box bb = h_boxes_format[i];
        std::cout << "x1:" << bb.x1 << "\ty1:" << bb.y1 << "\tx2:" << bb.x2 << "\ty2:" << bb.y2 << "\tconf:" << bb.score << "\tclass_id:" << bb.cls_id << endl;
    }
    /******************************************************************************************************************************/





    cudaMemcpy(d_nms_indices, h_nms_indices_init, max_object * sizeof(int), cudaMemcpyHostToDevice); //初始化nms处理的索引-->很重要




    /****************************************查看d_nms_indices数据******************************************************/
    int* d_nms_indices_visual = nullptr;
    cudaMallocHost(&d_nms_indices_visual, max_object * sizeof(int));
    cudaMemcpy(d_nms_indices_visual, d_nms_indices, max_object * sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "\n\nd_nms_indices:\n" << endl;
    for (int i = 0; i < max_object; i++) { std::cout << "\t" << d_nms_indices_visual[i] << endl; }

    /******************************************************************************************************************************/




    //这一步是将转换后的数据做nms处理，也是在gpu上进行cuda处理
    //nms_yolo_kernel << <grid_max, block >> > (d_boxes, d_nms_indices, h_count, nms_thr);



    /*******将yolo的gpu上结果转host端，然后保存结果处理-->最终结果保存在keep_boxes中**********/
    cudaMemcpy(h_boxes, d_boxes, anchor_output_num * sizeof(nms_box), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_nms_indices, d_nms_indices, max_object * sizeof(int), cudaMemcpyDeviceToHost);  //保存处理后的indice

    vector<nms_box> keep_boxes(h_count);
    for (int i = 0; i < h_count; i++) {
        if (h_nms_indices[i] > -1) {
            keep_boxes[i] = h_boxes[i];
        }
    }



    /****************************************查看nms处理后的-d_nms_indices******************************************************/
    std::cout << "nms处理后，保留box索引，-1表示排除obj，>-1表示保存obj" << endl;
    for (int i = 0; i < max_object; i++) { std::cout << h_nms_indices[i] << "\t"; }
    /**********************************************************************************************/



    /****************************************随便一张图为背景-显示结果于图上******************************************************/
    cv::Mat image = cv::imread("image.jpg");

    for (nms_box box : keep_boxes) {

        cv::Point p1(box.x1, box.y1);
        cv::Point p2(box.x2, box.y2);
        cv::rectangle(image, p1, p2, cv::Scalar(0, 255, 0), 4, 1, 0);//矩形的两个顶点，两个顶点都包括在矩形内部
    }


    cv::resize(image, image, cv::Size(600, 400), 0, 0, cv::INTER_NEAREST);
    cv::imshow("www", image);
    cv::waitKey(100000);
    cv::destroyAllWindows();
    /**********************************************************************************************/



}







void main() {


  
    imitate_yolo_part_postprocess();



}

