/*************************************************************************
	> File Name: helloworld.cpp
	> Author: wpx
	> Mail: wpx15673207315@gmail.com 
	> Created Time: 2020年11月12日 星期四 14时53分59秒
 ************************************************************************/

#include<iostream>
using namespace std;
inline auto area(float a, float b,float c=30);
auto area(float a, float b,float c)
{
	return (a*b*c);
}


int main()
{
	cout<<area(20,19);
	return 0;
}

