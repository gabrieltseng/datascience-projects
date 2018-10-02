#include <iostream>

int readNumber();
int writeAnswer(int);

int main()
{
	int x;
	int y;
	std::cout << "Enter your first number" << std::endl;
	x = readNumber();
	std::cout << "Enter your second number" << std::endl;
	y = readNumber();
	writeAnswer(x + y);
	return 0;
}
