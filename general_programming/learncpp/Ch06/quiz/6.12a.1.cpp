# include <iostream>
# include <string>

std::string getName()
{
    std::cout << "Enter a name: ";
    std::string name;
    std::cin >> name;
    return name;
}

int main()
{
    std::string acceptableNames[8] = {"Alex", "Betty", "Caroline", "Dave", "Emily", "Fred", "Greg", "Holly"};
    std::string name(getName());

    for (const auto &accName: acceptableNames)
    {
        if (accName == name)
        {
            std::cout << name << " was found." << std::endl;
            return 0;
        }
    }
    std::cout << name << " was not found." << std::endl;
    return 0;
}
