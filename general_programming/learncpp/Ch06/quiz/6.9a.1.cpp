# include <iostream>
# include <string>


void bubbleSort(std::string *array, int length)
{
    // For each iteration, we know the last element will be sorted
    // So we can loop through by one less each time
    int numIteration(0);
    for (int maxUnsorted = length - 1; maxUnsorted > 0; --maxUnsorted)
    {
        // If we go through the array without swapping any elements,
        // the array is sorted, and can exit the loop
        bool noSwap(true);

        // now we iterate through the array - up to maxUnsorted -
        // to do the swaps
        for (int curIndex = 0; curIndex < maxUnsorted; ++curIndex)
        {
            if (array[curIndex] > array[curIndex + 1])
            {
                noSwap = false;
                std::swap(array[curIndex], array[curIndex + 1]);
            }
        }
        ++numIteration;
        if (noSwap)
        {
            return;
        }
    }
}

int getNumNames()
{
    std::cout << "How many names will you enter?: ";
    int x;
    std::cin >> x;
    return x;
}

void getNames(std::string *nameArray, int numNames)
{

    for (int i = 0; i < numNames; ++ i)
    {
        std::cout << "Enter name #" << i + 1 << ": ";
        std::string temp;
        std::cin >> temp;
        nameArray[i] = temp;
    }
}

int main()
{
    int numNames(getNumNames());

    std::string *nameArray = new std::string[numNames];
    getNames(nameArray, numNames);
    bubbleSort(nameArray, numNames);

    std::cout << std::endl << "Here is your sorted list:" << std::endl;

    for (int i = 0; i < numNames; ++i)
    {
        std::cout << "Name #" << i + 1 << ": " << nameArray[i] << std::endl;
    }

}