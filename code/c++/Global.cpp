/*************************************************************************
	> File Name: Global.cpp
	> Author: wpx
	> Mail: wpx15673207315@gmail.com 
	> Created Time: 2020年11月14日 星期六 18时38分21秒
 ************************************************************************/

#include<iostream>
using namespace std;

void convert();
float fahrenheit;
float celsius;

int main()
{
	cout<<"enter fahrenheit: ";
	cin>>fahrenheit;
	convert();
	cout<<"celsius is \t"<<celsius<<"\n";
	return 0;
}

void convert()
{
	celsius = fahrenheit*5;
}
