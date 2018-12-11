#ifndef ADD_H
#define ADD_H

enum Rank
{
    two, 
    three, 
    four, 
    five, 
    six, 
    seven, 
    eight, 
    nine, 
    ten, 
    Jack, 
    Queen, 
    King, 
    Ace,
};


enum Suit
{
    clubs,
    diamonds,
    hearts,
    spades,
};

struct Card
{
    Rank rank;
    Suit suit;
};

// functions
void printCard(Card);
std::array<Card, 52> makeDeck();

// variable name to show the header its a reference
void shuffleDeck(std::array<Card, 52> &deck);

void printDeck(std::array<Card, 52>);
int generateRandomInt(int, int);
int getCardValue(Card);

# endif
