#include <iostream>
#include <string>
#include <stdexcept>


std::string intToBinary(int x)
{
	// initialize the string to be returned
	std::string binaryString("00000000");

	// Make sure x is in the right range
	if ((x > 255) || (x < 0))
	{
		throw std::invalid_argument("argument not in the range 0 to 255!");
	}
	int digitValue(128);
	int binaryIdx(0);
	while (x > 0)
	{
		if (x >= digitValue)
		{
			binaryString.at(binaryIdx) = '1';
			x -= digitValue;
		}
		digitValue /= 2;
		binaryIdx += 1;
	}
	return binaryString;
}


int readNumber()
{
	int x;
	std::cin >> x;
	return x;
}


int main()
{
	std::cout << "Enter an integer between 0 and 255" << std::endl;
	int x(readNumber());
	std::string binaryString(intToBinary(x));
	std::cout << x << " is " << binaryString << " in binary" << std::endl;
	return 0;
}
