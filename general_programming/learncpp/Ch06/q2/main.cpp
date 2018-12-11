# include <iostream>
# include <string>


struct Student
{
    std::string name;
    int grade;
};

int getNumStudents()
{
    std::cout << "How many students will you enter?: ";
    int x;
    std::cin >> x;
    return x;
}


void getStudents(Student *scores, int numStudents)
{
    for (int i = 0; i < numStudents; ++i)
        {
            std::cout << "Enter a name: ";
            std::string temp_name;
            std::cin >> temp_name;
            std::cout << "Enter a score: ";
            int temp_score;
            std::cin >> temp_score;

            scores[i].name = temp_name;
            scores[i].grade = temp_score;
        }
}


void bubbleSort(Student *scores, int numStudents)
{
    // For each iteration, we know the last element will be sorted
    // So we can loop through by one less each time
    int numIteration(0);
    for (int maxUnsorted = numStudents - 1; maxUnsorted > 0; --maxUnsorted)
    {
        // If we go through the array without swapping any elements,
        // the array is sorted, and can exit the loop
        bool noSwap(true);

        // now we iterate through the array - up to maxUnsorted -
        // to do the swaps
        for (int curIndex = 0; curIndex < maxUnsorted; ++curIndex)
        {
            if (scores[curIndex].grade < scores[curIndex + 1].grade)
            {
                noSwap = false;
                std::swap(scores[curIndex], scores[curIndex + 1]);
            }
        }
        ++numIteration;
        if (noSwap)
        {
            return;
        }
    }
}


void printScores(Student *scores, int numStudents)
{
    for (int i = 0; i < numStudents; ++ i)
    {
        std::cout << scores[i].name << " got a grade of " << scores[i].grade << std::endl;
    }
}


int main()
{
    int numStudents = getNumStudents();

    Student *scores = new Student[numStudents];
    getStudents(scores, numStudents);
    bubbleSort(scores, numStudents);
    std::cout << std::endl;
    printScores(scores, numStudents);

}
