# include <iostream>

enum class Items
{
    HEALTH_POTION,
    TORCH,
    ARROW
};


int countTotalItems(int (&player)[3])
{
    int totalItems = 0;
    for (auto item: player)
        totalItems = totalItems + item;
    return totalItems;
}


int main()
{
    // a player is an array of item values
    int playerOne[] = {2, 5, 10};

    int playerOneItems = countTotalItems(playerOne);
    std::cout << "Player one has " << playerOneItems << " items." << std::endl;
}
