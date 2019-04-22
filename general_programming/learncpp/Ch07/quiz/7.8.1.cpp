
#include <iostream>

typedef int (*arithmeticFcn)(int, int);

int add(int x, int y){
	return x + y;
}

int subtract(int x, int y){
	return x - y;
}

int multiply(int x, int y){
	return x * y;
}

int divide(int x, int y){
	return x / y;
}


arithmeticFcn getArithmeticFunction(char op){
	switch(op){
		case '+':
			return add;
		case '-':
			return subtract;
		case '*':
			return multiply;
		case '/':
			return divide;
	}                                                                 
}

int getInt(){
	std::cout << "Enter an integer: ";
	int x;
	std::cin >> x;
	return x;
}

char getOperator(){
	char acceptableOps[4] = {'*', '+', '-', '/'};

	while(true){
		std::cout << "Enter an operation: ";
		char op;
		std::cin >> op;

		bool ok = false;
		for (const auto &possibleOp: acceptableOps){
			if (possibleOp == op)
				ok = true;
		}
		if (ok)
			return op;
		std::cout << op << "not acceptable. Try again!";
		std::cin.clear();
		std::cin.ignore(32767, '\n');
	}

}

int main(){

	int x(getInt());
	int y(getInt());
	char op(getOperator());

	arithmeticFcn opFcn(getArithmeticFunction(op));

	std::cout << x << op << y << " = " << opFcn(x, y);
}
