#include <iostream>
#include <string>
#include <stdexcept>


enum monsterType
{
	TYPE_OGRE,
	TYPE_DRAGON,
	TYPE_GIANT_SPIDER,
	TYPE_SLIME,
};


struct Monster
{
	monsterType type;
	std::string name;
	double health;
};


std::string typeToName(monsterType type)
{
	if (type == TYPE_OGRE)
		return "Ogre";
	else if (type == TYPE_DRAGON)
		return "Dragon";
	else if (type == TYPE_GIANT_SPIDER)
		return "Giant Spider";
	else if (type == TYPE_SLIME)
		return "Slime";
	else
		throw std::invalid_argument("monster type invalid!");
}

void printStats(Monster monster)
{
	using std::cout;
	using std::endl;

	// This is to prevent the cout line from getting too long
	std::string type = typeToName(monster.type);
	std::string name = monster.name;
	double health = monster.health;

	cout << "This " << type << " is named " << name << " and has " << health << " health." << endl;
}

int main()
{
	// instantiate an ogre and a slime
	Monster ogre = {TYPE_OGRE, "Torg", 145};
	Monster slime = {TYPE_SLIME, "Blurp", 23};

	// print the stats
	printStats(ogre);
	printStats(slime);

	return 0;
}

