#include <iostream>

double readDoubleInput()
{
	std::cout << "Enter a double value: ";
	double x;
	std::cin >> x;
	return x;
}

int readCharInput()
{
	std::cout << "Enter one of the following: +, -, *, or /: ";
	char c;
	std::cin >> c;
	return c;
}

int main()
{
	double x(readDoubleInput());
	double y(readDoubleInput());
	char op(readCharInput());

	if (op == '*')
		std::cout << x << op << y << " is " << x * y << std::endl;
	if (op == '-')
		std::cout << x << op << y << " is " << x - y << std::endl;
	if (op == '+')
		std::cout << x << op << y << " is " << x + y << std::endl;
	if (op == '/')
		std::cout << x << op << y << " is " << x / y << std::endl;
}
