#include <iostream>
#include "constants.h"

int getTowerHeight()
{
	std::cout << "Enter the tower height in metres: ";
	int height;
	std::cin >> height;
	return height;
}

void getBallHeight(int towerHeight, int time)
{
	double distanceFallen(gravity * time * time / 2);
	double ballHeight(towerHeight - distanceFallen);
	if (ballHeight > 0)
		std::cout << "At " << time << "s, ball height is " << ballHeight << " metres" << std::endl;
	if (ballHeight <= 0)
		std::cout << "At " << time << "s, the ball is on the ground" << std::endl;
}

int main(){
	int towerHeight(getTowerHeight());
	getBallHeight(towerHeight, 0);
	getBallHeight(towerHeight, 1);
	getBallHeight(towerHeight, 2);
	getBallHeight(towerHeight, 3);
	getBallHeight(towerHeight, 4);
	getBallHeight(towerHeight, 5);
	return 0;
}
