#include <iostream>

namespace Animal
{
	enum Animal
	{
		CHICKEN,
		DOG,
		CAT,
		ELEPHANT,
		DUCK,
		SNAKE,
	};

	int animalLegs[] = {2, 4, 4, 4, 2, 0};
}

int main()
{
	Animal::Animal elephant(Animal::ELEPHANT);
	int numElephantLegs = Animal::animalLegs[elephant];

	std::cout << "An elephant has " << numElephantLegs << " legs" << std::endl;
}
