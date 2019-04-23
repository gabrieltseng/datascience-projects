#include <iostream>


int integerSum(int integers){
	if (integers > 10)
		return integers % 10 + integerSum(integers / 10);
	else
		return integers;
}


int main(){
	std::cout << "Sum of ints in 93427 is " << integerSum(93427) << std::endl;
}
