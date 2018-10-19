#include <iostream>
#include <string>
#include <stdexcept>


int readInt()
{
	std::cout << "Enter an integer: ";
	int x;
	std::cin >> x;
	return x;
}

char readOp()
{

	std::cout << "Enter a mathematical operation: ";
	char op;
	std::cin >> op;
	return op;
}

double performOperation(int x, int y, char op)
{
	// cast x to a double, so a double is returned
	double d_x = static_cast<double>(x);
	switch (op)
	{
		case '*':
			return d_x * y;
		case '/':
			return d_x / y;
		case '+':
			return d_x + y;
		case '-':
			return d_x - y;
		default:
			throw std::invalid_argument("Invalid operation!");
	}
}


int main()
{
	int x(readInt());
	int y(readInt());
	char op(readOp());

	double result(performOperation(x, y, op));

	std::cout << x << op << y << " = " << result << std::endl;
	return 0;
}
