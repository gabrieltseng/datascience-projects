#include <iostream>


int sumTo(int x)
{
	int sum(0);
	for(int counter(0); counter <= x; ++counter)
	{
		sum += counter;
	}
	return sum;
}


int main()
{
	std::cout << "Enter an integer: ";
	int x;
	std::cin >> x;

	int sum(sumTo(x));
	std::cout << "The sum of all the numbers from 1 to " << x << " is " << sum << std::endl;
	return 0;
}
