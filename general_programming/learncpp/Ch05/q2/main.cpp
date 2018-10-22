#include <iostream>
#include <cstdlib>


int generateRandomInt(int min = 0, int max = 100)
{
	// generate a random integer between min and max
	double fraction = 1.0 / (RAND_MAX + 1.0);

	return min + static_cast<int>((max - min + 1) * (std::rand() * fraction));
}


bool evaluateGuess(int guess, int targetNumber)
{
	bool correctGuess(false);
	// returns True if the guess is right, and False otherwise
	if (guess == targetNumber)
	{
		std::cout << "Correct! You win!" << std::endl;
		correctGuess = true;
	}
	else if (guess > targetNumber)
		std::cout << "Your guess is too high" << std::endl;
	else
		std::cout << "Your guess is too low" << std::endl;
	return correctGuess;
}

int readGuess(int numGuesses)
{
	while(true)
	{
		std::cout << "Guess #" << numGuesses + 1 << ": ";
		int x;
		std::cin >> x;

		if (std::cin.fail())
		{
			std::cin.clear();
			std::cin.ignore(32767, '\n');
		}
		else
			return x;
	}
}


void playGame()
{
	std::cout << "Let's play a game. I am thinking of a number. You have "
			<< "7 tries to guess what it is." << std::endl;
	int targetNumber(generateRandomInt());
	// 7 guesses to guess the target number
	bool correctGuess(false);
	int guess;

	for (int numGuesses(0); (numGuesses < 7) && (not correctGuess); ++numGuesses)
	{
		guess = readGuess(numGuesses);
		correctGuess = evaluateGuess(guess, targetNumber);
	}
}

bool playAgain()
{
	while(true)
	{
		std::cout << "Would you like to play again (y/n)? ";
		char keepPlaying;
		std::cin >> keepPlaying;
		if (keepPlaying == 'y')
		{
			return true;
		};
		if (keepPlaying == 'n')
		{
			return false;
		};
	}
}

int main()
{
	// set an initial seed
	std::srand(42);
	bool keepPlaying(false);
	do
	{
		playGame();
		keepPlaying = playAgain();
	}
	while (keepPlaying);
	std::cout << "Thank you for playing" << std::endl;
	return 0;
}
