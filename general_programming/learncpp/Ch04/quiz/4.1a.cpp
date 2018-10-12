# include <iostream>

int readNumber()
{
	int x;
	std::cin >> x;
	return x;
}

int main()
{
	std::cout << "Enter an integer" << std::endl;
	int x(readNumber());
	std::cout << "Enter a larger integer" << std::endl;
	int y(readNumber());

	// check x < y
	if (x > y)
	{
		std::cout << "Swapping values" << std::endl;
		int temp = x; // temp only defined in this block
		x = y;
		y = temp;
	}
	std::cout << "The larger value is " << y << std::endl;
	std::cout << "The smaller value is " << x << std::endl;
}
