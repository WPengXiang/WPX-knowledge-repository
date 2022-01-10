/*************************************************************************
	> File Name: advance-pointer.cpp
	> Author: wpx
	> Mail: wpx15673207315@gmail.com 
	> Created Time: 2020年11月20日 星期五 17时00分35秒
 ************************************************************************/

#include<iostream>
using namespace std;

class SimpletCat
{
public:
	SimpletCat();
	~SimpletCat(); 
private:
	int itsAge;
};

SimpletCat::SimpletCat()
{
	cout<<"new!!\n";
	itsAge =1;
}

SimpletCat::~SimpletCat()
{
	cout<<"destory!!\n";
}
   
int main()
{
	SimpletCat Cat;
	SimpletCat * pointer = new SimpletCat;
	delete pointer;

	//std::cout<<"ads\n";
	return 0;
}


