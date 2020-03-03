#include <string>
#include <iostream>
#include "repl.h"


int main() {
    std::cout << "Hello! This is the Monkey programming language \n";
    std::cout << "Feel free to type in commands. \n";

    repl::Start(std::cin, std::cout);
}
