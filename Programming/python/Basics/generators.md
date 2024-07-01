A generator is a function that behaves like an iterator.

To create a generator function, simple use the `yield` keyword

Consider an example where we're making a list of the first n integers:

```python

def first_n(n):
	num, nums = 0, []
	while num < n:
		nums.append(num)
		num += 1
	return nums
```

In this example, the list is created as soon as the function is called. This is not acceptable when you may be dealing with computationally complex cases or working with elements that use a lot of space in memory. 

Here is the same function as a generator:
```python
def firstn(n):
	num = 0
	while num < n:
		yield num
		num += 1
```

when you call this function, it returns a `generator object`. To get the elements, either loop over it:
```python
for elem in firstn(100):
	print(elem)
```

or wrap it in the list constructor:
```python
list(firstn(100))

# equivalent to 
first_n(100)
```

You can also turn a list comprehension into a generator by replacing `[]` wtih `()`
```python

# this is a list comprehension
listcomp = [n*2 for n in range(10)]

# this creates a generator
listgenerator = (n*2 for n in range(10Q))
```

# References 

- https://www.geeksforgeeks.org/generators-in-python/
- https://wiki.python.org/moin/Generators