#include <iostream>


void printBinaryInt(unsigned int x){
	if (x >= 1){
		printBinaryInt(x / 2);
		std::cout << x % 2;
	}
}


int main(){
	printBinaryInt(-15);
	std::cout << std::endl;
}
