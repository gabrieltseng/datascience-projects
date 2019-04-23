#include <iostream> 


int factorial(int x){
	if (x >= 1)
		return x * factorial(x - 1);
	else
		return 1;
}


int main(){
	int factorials[] = {1, 2, 3, 4, 5, 6, 7};

	for (const auto &facVal: factorials){
		std::cout << facVal << "! = " << factorial(facVal) << std::endl;
	}
}
