# include <iostream>

void swap(int &x, int &y)
{
    int temp = x;
    x = y;
    y = temp;
}

int main()
{
    int x = 1;
    int y = 2;

    std::cout << "Before swap, x: " << x << ", y: " << y << std::endl;

    swap(x, y);

    std::cout << "After swap, x: " << x << ", y: " << y << std::endl;

}
