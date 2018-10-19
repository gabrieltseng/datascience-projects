#include <iostream>
#include <string>

enum Animal
{
	ANIMAL_PIG,
	ANIMAL_CHICKEN,
	ANIMAL_GOAT,
};


std::string getAnimalName(Animal animal)
{
	switch(animal)
	{
		case ANIMAL_PIG:
			return "pig";
		case ANIMAL_CHICKEN:
			return "chicken";
		case ANIMAL_GOAT:
			return "goat";
	}
}


int getNumLegs(Animal animal)
{
	switch(animal)
	{
		case ANIMAL_PIG:
			return 4;
		case ANIMAL_CHICKEN:
			return 2;
		case ANIMAL_GOAT:
			return 4;
	}
}


int main()
{
	using std::cout;
	using std::endl;
	Animal pig(ANIMAL_PIG);
	Animal chicken(ANIMAL_CHICKEN);

	cout << "A " << getAnimalName(pig) << " has " << getNumLegs(pig) << " legs." << endl;
	cout << "A " << getAnimalName(chicken) << " has " << getNumLegs(chicken) << " legs." << endl;
}
