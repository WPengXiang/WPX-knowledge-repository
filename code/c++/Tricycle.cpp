/*************************************************************************
	> File Name: Tricycle.cpp
	> Author: wpx
	> Mail: wpx15673207315@gmail.com 
	> Created Time: 2020年11月18日 星期三 16时56分52秒
 ************************************************************************/

#include<iostream>
using namespace std;

class Tricycle 
{
	public:
		int speed;
	
	public:
	int getspeed()
	{
		return speed;
	}

	void setSpeed(int newspeed)
	{
		if (newspeed >=0)
		{
			speed = newspeed;
		}
	}
};

int main()
{
	Tricycle mycycle;
	mycycle.setSpeed(10);
	cout<<mycycle.speed;
}
