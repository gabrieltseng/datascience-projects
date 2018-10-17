#include <iostream>
#include <string>


double yearsLivedPerLetter(std::string name, int age)
{
	// find how many letters are in the name
	double numLetters(static_cast<double>(name.length()));
	return age / numLetters;
}

int main()
{
	using std::cout;
	using std::getline;
	using std::string;

	cout << "Enter your full name: ";
	string name;
	getline(std::cin, name);

	cout << "Enter your age:";
	int age;
	std::cin >> age;

	double yearsPerLetter(yearsLivedPerLetter(name, age));
	cout << "You have lived " << yearsPerLetter << " years per letter in your name" << std::endl;
	return 0;
}
