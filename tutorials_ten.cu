/*!

   我的教程优势:一系列且强逻辑顺序教程，附有源码，实战性很强。

	只有核心cuda处理代码，隐藏在教程中，我将不开源，毕竟已开源很多cuda教程代码，也为本次教程付出很多汗水。

	因此，核心代码于yolo部署cuda代码和整个文字解释教程需要有一定补偿，望理解。

	可以保证，认真学完教程，cuda编程毫无压力。

	详情请链接:http://t.csdn.cn/NaCZ5




	@Description : stream
	@Author      : tangjun
	@Date        : 2023-08-23
*/




#include "cuda_runtime.h"  
#include <iostream>
#include <stdio.h>  
#include <math.h>  
using namespace std;







__global__ void kernel_one(int* a, int* b, int* c)
{
	int threadID = blockIdx.x * blockDim.x + threadIdx.x;
	//printf("threadID:%d\n", threadID);

	c[threadID] = a[threadID] + b[threadID];
	
}

int stream_apply1()
{
	int N = 32;
	const int FULL_DATA_SIZE = N * 2;
	//获取设备属性
	cudaDeviceProp prop;
	int deviceID;
	cudaGetDevice(&deviceID);
	cudaGetDeviceProperties(&prop, deviceID);
	//检查设备是否支持重叠功能
	if (!prop.deviceOverlap)
	{
		printf("No device will handle overlaps. so no speed up from stream.\n");
		return 0;
	}

	//启动计时器
	cudaEvent_t start, stop;
	float elapsedTime;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	//创建一个CUDA流
	cudaStream_t stream;
	cudaStreamCreate(&stream);

	int* host_a, * host_b, * host_c;
	int* dev_a, * dev_b, * dev_c;

	//在GPU上分配内存
	cudaMalloc((void**)&dev_a, N * sizeof(int));
	cudaMalloc((void**)&dev_b, N * sizeof(int));
	cudaMalloc((void**)&dev_c, N * sizeof(int));

	//在CPU上分配页锁定内存
	cudaHostAlloc((void**)&host_a, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault);
	cudaHostAlloc((void**)&host_b, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault);
	cudaHostAlloc((void**)&host_c, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault);

	//主机上的内存赋值
	for (int i = 0; i < FULL_DATA_SIZE; i++)
	{
		host_a[i] = i;
		host_b[i] = 10000 * i;
	}

	for (int i = 0; i < FULL_DATA_SIZE; i += N)
	{
		cudaMemcpyAsync(dev_a, host_a + i, N * sizeof(int), cudaMemcpyHostToDevice, stream);
		cudaMemcpyAsync(dev_b, host_b + i, N * sizeof(int), cudaMemcpyHostToDevice, stream);


		kernel_one << <FULL_DATA_SIZE / 32, 32, 0, stream >> > (dev_a, dev_b, dev_c);

		cudaMemcpyAsync(host_c + i, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost, stream);
	}

	// wait until gpu execution finish  
	cudaStreamSynchronize(stream);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);

	std::cout << "消耗时间： " << elapsedTime << std::endl;



	cout << "输入数据host_a" << endl;
	for (int i = 0; i < FULL_DATA_SIZE; i++) { std::cout << host_a[i] << "\t"; }
	cout << "\n输入数据host_b" << endl;
	for (int i = 0; i < FULL_DATA_SIZE; i++) { std::cout << host_b[i] << "\t"; }

	cout << "\n输出结果host_c" << endl;
	for (int i = 0; i < FULL_DATA_SIZE; i++)	{std::cout << host_c[i] << "\t"; }

	getchar();

	// free stream and mem  
	cudaFreeHost(host_a);
	cudaFreeHost(host_b);
	cudaFreeHost(host_c);

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	cudaStreamDestroy(stream);
	return 0;
}

int stream_apply2()
{
	const int NS = 4;
	const int ND = 32;

	//创建CUDA流与初始化
	cudaStream_t streams[NS];
	for (int i = 0; i < NS; i++) { cudaStreamCreate(&streams[i]); }
	

	int* host_a, * host_b, * host_c;
	int* dev_a, * dev_b, * dev_c;

	//在GPU上分配内存
	//cudaMalloc((void**)&dev_a, ND * sizeof(int));
	//cudaMalloc((void**)&dev_b, ND * sizeof(int));
	//cudaMalloc((void**)&dev_c, ND * sizeof(int));


	cudaMalloc((void**)&dev_a, ND * NS * sizeof(int));
	cudaMalloc((void**)&dev_b, ND * NS * sizeof(int));
	cudaMalloc((void**)&dev_c, ND * NS * sizeof(int));

	//在CPU上分配页锁定内存
	cudaHostAlloc((void**)&host_a, ND*NS * sizeof(int), cudaHostAllocDefault);
	cudaHostAlloc((void**)&host_b, ND*NS * sizeof(int), cudaHostAllocDefault);
	cudaHostAlloc((void**)&host_c, ND*NS * sizeof(int), cudaHostAllocDefault);

	//主机上的内存赋值
	for (int i = 0; i < ND * NS; i++)	{
		host_a[i] = i;
		host_b[i] = 10000 * i;	}

	for (int i = 0; i < NS; i++)	{
		cudaMemcpyAsync(dev_a + i * ND, host_a + i * ND, ND * sizeof(int), cudaMemcpyHostToDevice, streams[i]);
		cudaMemcpyAsync(dev_b + i * ND, host_b + i * ND, ND * sizeof(int), cudaMemcpyHostToDevice, streams[i]);
		kernel_one << <ND / 32, 32, 0, streams[i] >> > (dev_a + i * ND, dev_b + i * ND, dev_c + i * ND);
		cudaMemcpyAsync(host_c + i * ND, dev_c + i * ND, ND * sizeof(int), cudaMemcpyDeviceToHost, streams[i]);

	}

	// wait until gpu execution finish  
	cudaDeviceSynchronize();

	cout << "输入数据host_a" << endl;
	for (int i = 0; i < ND * NS; i++) { std::cout << host_a[i] << "\t"; }
	cout << "\n输入数据host_b" << endl;
	for (int i = 0; i < ND * NS; i++) { std::cout << host_b[i] << "\t"; }
	cout << "\n输出结果host_c" << endl;
	for (int i = 0; i < ND * NS; i++) { std::cout << host_c[i] << "\t"; }
	// free stream and mem  
	cudaFreeHost(host_a);
	cudaFreeHost(host_b);
	cudaFreeHost(host_c);

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	for (int i = 0; i < NS; i++) { cudaStreamDestroy(streams[i]); }
	return 0;
}




int stream_apply3()
{
	const int NS = 4; //流个数
	const int ND = 32; //每个流分配负责多少个数据

	cudaStream_t streams[NS]; //创建多个cuda流
	for (int i = 0; i < NS; i++) { cudaStreamCreate(&streams[i]); } //每个流初始化

	int* host_a, * host_b, * host_c; //host端变量
	int* dev_a, * dev_b, * dev_c; //gpu端变量

	//在GPU上分配内存
	cudaMalloc((void**)&dev_a, ND * sizeof(int));
	cudaMalloc((void**)&dev_b, ND * sizeof(int));
	cudaMalloc((void**)&dev_c, ND * sizeof(int));

	
	//在CPU上分配页锁定内存，必须使用cudaHostAlloc方法
	cudaHostAlloc((void**)&host_a, ND * NS * sizeof(int), cudaHostAllocDefault);
	cudaHostAlloc((void**)&host_b, ND * NS * sizeof(int), cudaHostAllocDefault);
	cudaHostAlloc((void**)&host_c, ND * NS * sizeof(int), cudaHostAllocDefault);

	//主机上的内存赋值
	for (int i = 0; i < ND * NS; i++) {
		host_a[i] = i;
		host_b[i] = 10000 * i;
	}
	//循环流，为每个流分配数据赋值与kernel操作过程
	for (int i = 0; i < NS; i++) {
		cudaMemcpyAsync(dev_a , host_a + i * ND, ND * sizeof(int), cudaMemcpyHostToDevice, streams[i]);
		cudaMemcpyAsync(dev_b , host_b + i * ND, ND * sizeof(int), cudaMemcpyHostToDevice, streams[i]);
		kernel_one << <ND / 32, 32, 0, streams[i] >> > (dev_a , dev_b , dev_c );
		cudaMemcpyAsync(host_c + i * ND, dev_c , ND * sizeof(int), cudaMemcpyDeviceToHost, streams[i]);

	}

	// wait until gpu execution finish  
	cudaDeviceSynchronize();//等待所有异步执行完，cpu才操作
	//打印输出结果
	cout << "输入数据host_a" << endl;
	for (int i = 0; i < ND * NS; i++) { std::cout << host_a[i] << "\t"; }
	cout << "\n输入数据host_b" << endl;
	for (int i = 0; i < ND * NS; i++) { std::cout << host_b[i] << "\t"; }
	cout << "\n输出结果host_c" << endl;
	for (int i = 0; i < ND * NS; i++) { std::cout << host_c[i] << "\t"; }
	// free stream and mem  
	cudaFreeHost(host_a);
	cudaFreeHost(host_b);
	cudaFreeHost(host_c);

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	for (int i = 0; i < NS; i++) { cudaStreamDestroy(streams[i]); }
	return 0;
}













int main_ten() {


	//stream_apply1();
	stream_apply2();
	stream_apply3();

	return 0;

}

