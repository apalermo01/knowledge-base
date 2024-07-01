# Iterables
 
 - an **iterable** is anything you can iterate over (i.e. throw it in a for loop)

 - an **iterator** is a kind of abstraction of an iterable - instead of just looping over loops and other "container" datatypes, you can define your own datatypes (classes) that you can iterate over.

- To make an **iterator** you have to define 2 methods: `__iter__()` and `__next__()`

- `__iter__()` is called to initialize the iterator, must return an object. Typically you just return `self`

- `__next__()` should return the next piece of data in the sequence.


- [**generators**](generators.md) let you write functions that behave as iterators


- **Generator expressions** are like [[list comprehensions]], except that they return a generator instead of a fully-formed list:

```python

list_comp = [item for item in [1,2,3,4]]
generator_comp = (item for item in [1,2,3,4])
 ```


- big benefit of using iterators / generators over container data types: much more memory efficient. You don't need to worry about storing all the data in memory at the same time.
	- Decouples iteration from processing individual items

ex:
```python

def to_even(numbers):
	return (number for number in numbers if number % 2 == 0)

def to_square(numbers):
	return (number ** 2 for number in numbers)

def pipeline():
	list(to_square(to_even(range(20))))
```

- think of iterators / generators as "lazily" executed. You define what computations you want to happen ahead of time and only to the actual calculation when the value is required.

- drawback: you can't iterate over an iterator multiple times. Once you reach the end of the sequence its "used up"

## References
- https://realpython.com/python-iterators-iterables/
