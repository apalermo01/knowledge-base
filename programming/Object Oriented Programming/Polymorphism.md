Breaking down the word, polymorphism means *many forms*. It is the idea that you can access objects of different types through the same interface. 

As an example, you can add integers and floats using the same interface (`+`) despite the fact that they're different kinds of entities (according to the computer, at least), hence they are polymorphic. Both integers and floats are instances of a number. 

Consider the following collection of python classes:
```python

class Vehicle:
	def wheels(self):
		 raise NotImplementedError
	 
	def drive(self):
		if about_to_crash(self):
			dont()

class Bicycle(Vehicle):
	def wheels(self):
		return 2


class Car(Vehicle):
	def wheels(self):
		return 4


class Truck(Vehicle):
	def wheels(self):
		return 18
```

`Vehicle` is a parent to `Bicycle`, `Car`, and `Truck`. Each of these 3 classes have a method called `drive` becuase it inherits that functionality from `Vehicle`- which calls some functionality that's defined elsewhere.

Additionally, each of these 3 kinds of vehicles (along with anything else that inherits `Vehicle`) needs to have its own function definition for wheels- since not doing so would raise the `NotImplementedError`\*. 

Every time you work with something that is a vehicle, you use the same interface (function) to get the number of wheels. Hence, we use the same interface to access the properties of many different objects, which is the definition of polymorphic.


\*This idiom in python tries to replicate what other programming languages would call an *abstract* method (e.g. java).  

**Function / Method overloading**

In many OOP languages (except for python), it is possible to write multiple functions with the same name but with different arguments. Take this example in C++:

```c++
#include <iostream>

int add(int a, int b){
	return a + b;
}

float add(float a, float b){
	return a + b;
}

float add(std::string a, std::string b){
	float a_float = std::stof(a);
	float b_float = str::stof(b);
	return a_float + b_float;
}

int main(){
	int a = 2;
	int b = 3;
	float c = 2.1;
	float d = 4.3;
	std::string e = "2.3534";
	std::string f = "5.234465";

	std::cout << "adding ints: " << add(a, b) << std::endl;
	std::cout << "adding floats: " << add(c, d) << std::endl;
	std::cout << "adding strings: " << add(e, f) << std::endl;
}
```

Here, the function `add` is defined 3 times- once when both arguments are integers, again when both arguments are floats, and again when both arguments are strings. When calling the function, the compiler will automatically know which function to run based on the types of the input arguments. 



## References
- https://stackify.com/oop-concept-polymorphism/
- https://stackoverflow.com/questions/1031273/what-is-polymorphism-what-is-it-for-and-how-is-it-used