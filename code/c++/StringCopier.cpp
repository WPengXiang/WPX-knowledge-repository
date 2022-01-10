/*************************************************************************
	> File Name: StringCopier.cpp
	> Author: wpx
	> Mail: wpx15673207315@gmail.com 
	> Created Time: 2020年11月17日 星期二 16时05分34秒
 ************************************************************************/

#include<iostream>
#include<string.h>
using namespace std;
int main()
{
	char str1[] = "freeodicals!";
	char str2[80];
	strncpy(str2,str1);
	cout<<str2<<endl;
}
