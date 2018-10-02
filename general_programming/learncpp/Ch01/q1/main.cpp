#include <iostream>

int readNumber()
{
	int x;
	std::cin >> x;
	return x;
}

void writeAnswer(int x)
{
	std::cout << "The numbers add to: " << x << std::endl;
}

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
