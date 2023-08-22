#include "cuda_runtime.h"  
#include <iostream>
#include <stdio.h>  
#include <math.h>  
using namespace std;








__global__ void addKernel(int* c, int* a, int* b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}

void testStream() {
	int const N = 100;
	int p_data_a[N] = { 0 };
	int p_data_b[N] = { 0 };
	int p_data_c[N] = { 0 };

	for (int i = 0; i < N; i++) {
		p_data_a[i] = i;
		p_data_b[i] = 1000 + i;
		
	}


	cout << "\n输入数据：\n" << "p_data_a:" << endl;

	for (int i = 0; i < N; i++) { cout << p_data_a[i] << "\t"; }
	cout << "\np_data_b:" << endl;
	for (int i = 0; i < N; i++) { cout << p_data_b[i] << "\t"; }
	


	int* dev_a = nullptr;
	int* dev_b = nullptr;
	int* dev_c = nullptr;
	cudaMalloc(&dev_a, sizeof(int) * N);
	cudaMalloc(&dev_b, sizeof(int) * N);
	cudaMalloc(&dev_c, sizeof(int) * N);

	//拷贝到内存
	cudaMemcpy(dev_a, p_data_a, sizeof(int) * N, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, p_data_b, sizeof(int) * N, cudaMemcpyHostToDevice);

	cudaStream_t streams[N];
	for (int i = 0; i < N; ++i) { cudaStreamCreate(streams + i); }

	for (int i = 0; i < N; ++i) { addKernel << <1, 1, 0 >> > (dev_c + i, dev_a + i, dev_b + i); }
	cudaDeviceSynchronize();
	
	cudaMemcpy(p_data_c, dev_c, sizeof(int) * N, cudaMemcpyDeviceToHost);
	
	
	cout << "\n查看输出结果：" << endl;
	
	for (int i = 0; i < N; i++) {		cout << p_data_c[i] << "\t";	}
	
	
	cout << endl;
	cout << "over" << endl;
	
	
	
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
}


int stream_apply1() {
	testStream();
	getchar();
	return 0;
}


__global__ void kernel_one(int* a, int* b, int* c, int N)
{
	int threadID = blockIdx.x * blockDim.x + threadIdx.x;
	//printf("threadID:%d\n", threadID);
	if (threadID < N)
	{
		c[threadID] = (a[threadID] + b[threadID]) / 2;
	}
}

int stream_apply2()
{

	int N = 1024 * 1024;
	const int FULL_DATA_SIZE = N * 20;


	//启动计时器
	cudaEvent_t start, stop;
	float elapsedTime;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	int* host_a, * host_b, * host_c;
	int* dev_a, * dev_b, * dev_c;

	//在GPU上分配内存
	cudaMalloc((void**)&dev_a, FULL_DATA_SIZE * sizeof(int));
	cudaMalloc((void**)&dev_b, FULL_DATA_SIZE * sizeof(int));
	cudaMalloc((void**)&dev_c, FULL_DATA_SIZE * sizeof(int));

	//在CPU上分配可分页内存
	host_a = (int*)malloc(FULL_DATA_SIZE * sizeof(int));
	host_b = (int*)malloc(FULL_DATA_SIZE * sizeof(int));
	host_c = (int*)malloc(FULL_DATA_SIZE * sizeof(int));

	//主机上的内存赋值
	for (int i = 0; i < FULL_DATA_SIZE; i++)
	{
		host_a[i] = i;
		host_b[i] = FULL_DATA_SIZE - i;
	}

	//从主机到设备复制数据
	cudaMemcpy(dev_a, host_a, FULL_DATA_SIZE * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, host_b, FULL_DATA_SIZE * sizeof(int), cudaMemcpyHostToDevice);

	kernel_one << <FULL_DATA_SIZE / 1024, 1024 >> > (dev_a, dev_b, dev_c, N);

	//数据拷贝回主机
	cudaMemcpy(host_c, dev_c, FULL_DATA_SIZE * sizeof(int), cudaMemcpyDeviceToHost);

	//计时结束
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);

	std::cout << "消耗时间： " << elapsedTime << std::endl;

	//输出前10个结果
	for (int i = 0; i < 10; i++)
	{
		std::cout << host_c[i] << std::endl;
	}

	getchar();

	cudaFreeHost(host_a);
	cudaFreeHost(host_b);
	cudaFreeHost(host_c);

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	return 0;
}

int stream_apply()
{


	int N = 1024 * 1024;
	const int FULL_DATA_SIZE = N * 20;


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
		host_b[i] = FULL_DATA_SIZE - i;
	}

	for (int i = 0; i < FULL_DATA_SIZE; i += N)
	{
		cudaMemcpyAsync(dev_a, host_a + i, N * sizeof(int), cudaMemcpyHostToDevice, stream);
		cudaMemcpyAsync(dev_b, host_b + i, N * sizeof(int), cudaMemcpyHostToDevice, stream);

		kernel_one << <N / 1024, 1024, 0, stream >> > (dev_a, dev_b, dev_c, N);

		cudaMemcpyAsync(host_c + i, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost, stream);
	}

	// wait until gpu execution finish  
	cudaStreamSynchronize(stream);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);

	std::cout << "消耗时间： " << elapsedTime << std::endl;

	//输出前10个结果
	for (int i = 0; i < 10; i++)
	{
		std::cout << host_c[i] << std::endl;
	}

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





int main() {


	stream_apply1();

	return 0;

}

