# include <algorithm>
# include <iostream>

int main()
{
    // bubble sort

    // array to be sorted
    const int length(9);
    int array[length] = { 6, 3, 2, 9, 7, 1, 5, 4, 8 };

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
            std::cout << "Stopped after " << numIteration << " iterations." << std::endl;
            for (int i(0); i < length; ++i)
                std::cout << array[i] << " ";
            std::cout << std::endl;
            return 0;
        }
    }
 }
 