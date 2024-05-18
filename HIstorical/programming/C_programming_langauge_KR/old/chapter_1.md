# Chapter 1 notes on C programming language

## 1.2

list of character format options:

- %d    -> print as a decimal integer
- %6d   -> print as decimal integer, at least 6 characters wide
- %f    -> print as a floating point
- %6f   -> print as a floating point, at least 6 characters wide
- %.2f  -> print as floating point, 2 characters after decimal point
- %6.2f -> print as floating point, at least 6 characters wide and 2 characters after decimal point
- %o    -> octal
- %x    -> hexidecimal
- %c    -> character
- %s    -> string
- %\%   -> % itself

## 1.4 symbolic constants

syntax: 

```c
#define <name> <replacement
```

- usually written in upper case
- replaces the raw text (i.e. NOT A VARIABLE)


## 1.5 character input and output
vv
- `getchar()` -> reads the next character of input
- `putchar()` -> prints a character


Note: in file copying example, we set c to be an integer to make sure it's large enough to hold anything that `getchar` returns.
- NOTE: in linux, to type EOF in terminal hit <C-d>
