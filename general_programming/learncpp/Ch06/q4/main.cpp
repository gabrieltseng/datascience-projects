# include <iostream>

void printString(char *s)
{
    while (*s != '\0')
    {
        std::cout << *s;
        s++;
    }
    std::cout << std::endl;
}

int main()
{
    char testString[] = "Hello World!";
    printString(testString);
}
