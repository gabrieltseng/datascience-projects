#include <iostream>
#include "constants.h"

int getTowerHeight()
{
	std::cout << "Enter the tower height in metres: ";
	int height;
	std::cin >> height;
	return height;
}


double getBallHeight(int towerHeight, int time)
{
	double distanceFallen(gravity * time * time / 2);
	double ballHeight(towerHeight - distanceFallen);
	if (ballHeight > 0)
		return ballHeight;
	else
	{
		double ground(0.0);
		return ground;
	}
}

int main(){
	using std::cout;
	using std::endl;

	int towerHeight(getTowerHeight());

	// initalize ball height at tower height
	double ballHeight(towerHeight);
	int numSeconds(0);

	while (ballHeight > 0)
	{
		ballHeight = getBallHeight(towerHeight, numSeconds);
		cout << "At " << numSeconds << "s, the ball is " << ballHeight << "m high" << endl;
		++numSeconds;
	}
	return 0;
}
