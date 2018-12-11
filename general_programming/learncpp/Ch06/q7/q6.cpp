// code written for q6, useful for q7
// enums and structs moved into the header file

# include <iostream>
# include <array>
#include <cstdlib>
#include <ctime>
#include "q6.h"

void printCard(Card card)
{
    using std::cout; 

    switch (card.rank)
    {
        case two: {cout  << "2"; break;}
        case three: {cout << "3"; break;}
        case four: {cout << "4"; break;}
        case five: {cout << "5"; break;}
        case six: {cout << "6"; break;}
        case seven: {cout << "7"; break;}
        case eight: {cout << "8"; break;}
        case nine: {cout << "9"; break;}
        case ten: {cout << "10"; break;}
        case Jack: {cout << "J"; break;}
        case Queen: {cout << "Q"; break;}
        case King: {cout << "K"; break;}
        case Ace: {cout << "A"; break;}
    };
    switch (card.suit)
    {
        case clubs: {cout << "C"; break;}
        case diamonds: {cout << "D"; break;}
        case hearts: {cout << "H"; break;}
        case spades: {cout << "S"; break;}
    }
    cout << std::endl;
}

std::array<Card, 52> makeDeck()
{
    std::array<Card, 52> fullDeck;
    int deckIndex = 0;
    for (int rankInt = two; rankInt <= Ace; rankInt++)
    {
        for (int suitInt = clubs; suitInt <= spades; suitInt++)
        {
            fullDeck[deckIndex].rank = static_cast<Rank>(rankInt);
            fullDeck[deckIndex].suit = static_cast<Suit>(suitInt);
            deckIndex++;
        }
    }
    return fullDeck;
}

void printDeck(std::array<Card, 52> deck)
{
    for (auto card: deck)
        printCard(card);
}


void swapCards(Card &c1, Card &c2)
{
    Card temp = c1;
    c1 = c2;
    c2 = temp;
}

int generateRandomInt(int min = 0, int max = 52)
{
    // generate a random integer between min and max
    double fraction = 1.0 / (RAND_MAX + 1.0);

    return min + static_cast<int>((max - min + 1) * (std::rand() * fraction));
}

void shuffleDeck(std::array<Card, 52> &deck)
{
    for (int cardNum = 0; cardNum < 52; ++cardNum)
    {
        int swapNum = generateRandomInt();
        swapCards(deck[cardNum], deck[swapNum]);
    }
}

int getCardValue(Card card)
{
        switch (card.rank)
    {
        case two: {return 2;}
        case three: {return 3;}
        case four: {return 4;}
        case five: {return 5;}
        case six: {return 6;}
        case seven: {return 7;}
        case eight: {return 8;}
        case nine: {return 9;}
        case ten: {return 10;}
        case Jack: {return 10;}
        case Queen: {return 10;}
        case King: {return 10;}
        case Ace: {return 11;}
    };
}
