


#include <iostream>
#include <time.h>

using namespace std;




/*************************************第一节-指针篇**********************************************/
void Print_Pointers(int* ptr, int N) {
    for (int i = 0; i < N; i++)
    {
        std::cout << "order:\t" << i << "\tptr_value:\t" << *ptr << "\tphysical address:" << ptr << std::endl;
        ptr++;
    }
}

void pointer_1() {
    /*  探索指针赋值方法 */
    const int N = 6;
    int arr[N];
    for (int i = 0; i < N; i++) arr[i] = i + 1; //数组赋值
    //指针第一种赋值方法
    int* ptr = nullptr;
    ptr = arr;
    //指针第二种赋值方法
    int* ptr2 = arr;

    std::cout << "output ptr1 " << std::endl;
    Print_Pointers(ptr, N);
    std::cout << "\n\noutput ptr2 " << std::endl;
    Print_Pointers(ptr2, N);

    //单独变量赋值
    int a = 20;
    int* p = &a;
    std::cout << "\n\noutput p value: \t" << *p << "\tphysical address:\t" << p << std::endl;

}

void pointer_2() {
    const int N = 6;
    int arr[N];
    for (int i = 0; i < N; i++) arr[i] = i + 1; //数组赋值
    int* ptr = arr; //构建指针
    for (int i = 0; i < 5; i++)
    {
        std::cout << "ptr_value_" << i << ":\t" << *ptr << std::endl;;
        ptr++;
    }
}


void pointer_3() {
    int num = 4;
    int* p = &num;
    cout << "*p:\t" << *p << "\t p address:\t" << p << "\tnum value:\t" << num << "\tnum address:\t" << num << endl;

    *p = *p + 20; //通过指针更改地址的值
    cout << "*p:\t" << *p << "\t p address:\t" << p << "\tnum value:\t" << num << "\tnum address:\t" << num << endl;
    num = 30; //更改变量值
    cout << "*p:\t" << *p << "\t p address:\t" << p << "\tnum value:\t" << num << "\tnum address:\t" << num << endl;


}

void pointer_4() {
    int num = 4;
    int* p1 = &num;
    //指针的指针第一种赋值方法
    int** p2 = &p1;
    //指针的指针第二种赋值方法
    int** p3;
    p3 = &p1;

    cout << "num value:\t" << num << "\t num address:\t" << &num << endl;
    cout << "p1 value:\t" << *p1 << "\t p1 address:\t" << p1 << endl;
    cout << "p2 value:\t" << *p2 << "\t p2 address:\t" << p2 << endl;
    cout << "p3 value:\t" << *p3 << "\t p3 address:\t" << p3 << endl;

    cout << "p2 value:\t" << **p2 << "\t p2 address:\t" << *p2 << endl;
}

void main_first() {
    pointer_1();
    pointer_2();
    pointer_3();
    pointer_4();


}




