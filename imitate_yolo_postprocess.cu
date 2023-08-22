/*!
    @Description : 模拟yolo输出使用cuda处理
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
#include<math.h>
using namespace cv;
using namespace std;




#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>





__global__ void decode_yolo(float* prob, float* parray, int max_objects, int cls_num, float conf_thr, int* d_count) {
    //gpu_input输入格式为anchor_output_num个[x1,y1,x2,y2,conf,cls1_score,cls2_score,...]
    
    int idx = blockDim.x * blockIdx.x + threadIdx.x; //idx设置为对应输出数量
    int tmp_idx = idx * (cls_num + 5); //cls+x,y,w,h+conf

    //假设左上右下为以下对应数据
    float left = prob[tmp_idx + 0]; //center_x center_y w h
    float top = prob[tmp_idx + 1];
    float right = prob[tmp_idx + 2];
    float bottom = prob[tmp_idx + 3];
    float conf = prob[tmp_idx + 4];  //是为目标的置信度

    float class_score = prob[tmp_idx + 5];

    float tmp_conf = conf * class_score;
    int class_id = 0;
    for (int j = 0; j < cls_num; j++) {

        int cls_idx = tmp_idx + 5 + j;
        if (tmp_conf < conf * prob[cls_idx]) {
            class_id = j;
            tmp_conf = conf * prob[cls_idx];
        }

    }

    if (tmp_conf < conf_thr) { return; }

    int index = atomicAdd(d_count, 1); //当作指针，相当于每次会走一次
    if (index >= max_objects) { return; } //不能超过最大目标

    int out_index = index * 6; //6=x1,y1,x2,y2,conf,class_id


    //printf("\out_index:%d\n", out_index);

    parray[out_index + 0] = left;   //1
    parray[out_index + 1] = top;    //2
    parray[out_index + 2] = right;  //3
    parray[out_index + 3] = bottom;     //
    parray[out_index + 4] = tmp_conf;   //  置信度
    parray[out_index + 5] = class_id;   //  类别





}



void imitate_yolo_postprocess() {
    /*

    该代码模拟yolo后处理方法

    */


    const int max_object = 6;


    int cls_num = 2;

    int N = 21; //模型输出目标数量，类似25200
    int N_obj = N * (cls_num + 5);//模型输出cls+x,y,w,h+conf


    float* input_data = nullptr;
    cudaMallocHost((void**)&input_data, sizeof(float) * N_obj);
    std::cout << "原始数据赋值+打印:" << endl;
    for (int i = 0; i < N_obj; i++) {
        //input_data[i] = (float)(i+1) ;
        float value = rand() / float(RAND_MAX);
        input_data[i] = round(value * 10000) / 10000;
        if (i % (cls_num + 5) == 0) { std::cout << endl; }
        std::cout << input_data[i] << "\t\t";

    }

    float* gpu_input = nullptr;
    cudaMalloc((void**)&gpu_input, sizeof(float) * N_obj);
    cudaMemcpy(gpu_input, input_data, sizeof(float) * N_obj, cudaMemcpyHostToDevice);



    float* gpu_output = nullptr; // 保存gpu处理的变量 
    cudaMalloc((void**)&gpu_output, sizeof(float) * max_object * 6);




    int count = 0;
    int* d_count = nullptr;
    cudaMalloc((void**)&d_count, sizeof(int));
    cudaMemcpy(d_count, &count, sizeof(int), cudaMemcpyHostToDevice);





    cudaStream_t stream;
    cudaStreamCreate(&stream);

    int block = 32;

    int grid = (N + block - 1) / block;


    float conf_thr = 0.45;

    decode_yolo << < grid, block, 0, stream >> > (gpu_input, gpu_output, max_object, cls_num, conf_thr, d_count);






    float* host_decode = nullptr; // 保存gpu处理的变量 
    cudaMallocHost((void**)&host_decode, sizeof(float) * max_object * 6);


    cudaMemcpy(host_decode, gpu_output, sizeof(float) * max_object * 6, cudaMemcpyDeviceToHost);

    cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost);




    std::cout << "\n\n打印输出结果\n" << endl;
    if (count == 0) { std::cout << "\n无检测结果" << endl; }
    if (count > max_object) { count = max_object; };

    for (int i = 0; i < count; i++) {

        int idx = i * 6;
        std::cout << "x1:" << host_decode[idx] << "\ty1:" << host_decode[idx + 1] << "\tx2:" << host_decode[idx + 2]
            << "\ty2:" << host_decode[idx + 3] << "\tconf:" << host_decode[idx + 4] << "\tclass_id:" << host_decode[idx + 5] << endl;

    }




}



void main_yolo() {
    imitate_yolo_postprocess();

}










