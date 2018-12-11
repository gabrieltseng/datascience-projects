#include <iostream>
#include <array>
#include <cstdlib>
#include <ctime>
#include <string>
#include "q6.h"


bool hitMe()
{
    while(true)
    {
        std::cout << "Hit? (Y/N) ";
        std::string x;
        std::cin >> x;

        if (std::cin.fail())
        {
            std::cin.clear();
            std::cin.ignore(32767, '\n');
        }
        else
        {
            if (x == "Y")
            {
            return true;
            }
            if (x == "N")
            {
                return false;
            }
        }
    }
}


bool playBlackJack(std::array<Card, 52> deck)
{
    int dealerScore = 0;
    int playerScore = 0;

    Card *curCard = &deck[0];
    dealerScore += getCardValue(*curCard++);
    // player gets two cards
    playerScore += getCardValue(*curCard++);
    playerScore += getCardValue(*curCard++);

    // show the scores at the beginning
    std::cout << "Player score: " << playerScore << std::endl;
    std::cout << "Dealer score: " << dealerScore << std::endl;

    // if the player has stood once, they cannot
    // change their mind; this keeps track of that
    bool playerStayed = false;
    bool dealerStayed = false;

    // the game keeps going until both the player
    // and the dealer have stood
    while (!playerStayed && !dealerStayed) 
    {
        // player's turn first
        if (!playerStayed)
        {
            bool playerGoes = hitMe();
            if (!playerGoes)
            {
                playerStayed = true;
            }
            else
            {
                playerScore += getCardValue(*curCard++);
                std::cout << "Player score: " << playerScore << std::endl;

                if (playerScore > 21)
                {
                    std::cout << "Player is bust!" << std::endl;
                    return false;
                }
            }
        }
        // now, the dealer's turn
        if (dealerScore < 17)
        {
            dealerScore += getCardValue(*curCard++);
            std::cout << "Dealer score: " << dealerScore << std::endl;
            if (dealerScore > 21)
            {
                std::cout << "Dealer is bust!" << std::endl;
                return true;
            }
        }
        else
        {
            dealerStayed = true;
        }
    }

    if (playerScore > dealerScore)
        return true;
    else
        return false;
}


int main()
{
    // for shuffling
    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    std::array<Card, 52> fullDeck = makeDeck();
    shuffleDeck(fullDeck);

    bool playerWon = playBlackJack(fullDeck);

    if (playerWon)
        std::cout << "Congratulations! Player won";
    else
        std::cout << "Dealer won";
    std::cout << std::endl;
}
