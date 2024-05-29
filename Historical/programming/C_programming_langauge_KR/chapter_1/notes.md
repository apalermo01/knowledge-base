# Chapter 1

to compile:

```
cc <name of file>.c
```

This will generate an executable called `a.out`


To save the output to a specific file:

```
cc <name of file>.c -o <name of output>
```

## Character input and output

*text stream* - A sequence of characters divided into lines

`c = getchar()` - reads the next character from the input stream and stores in to the variable c

`putchar(c)` - writes the character c to the standard output


note: the EOF character can be sent to stdin using <ctrl>+D (at least that's the case on linux)

To run the file copying program, do `echo "input text here" | ./file_copying.out` or `cat input_file.txt | ./file_copying.out`


**character counting**

```c
    for (nc = 0; getchar() != EOF; ++nc)
        ;
```
The isolated `;` in the snippet is there becuase the body of the for loop is empty. This is called a null statement.

**line counting**

The standard library ensures that an input text stream appears as a sequence of lines, terminated by a newline character.

**character constants** - A character written between single quotes returns an integer of the ascii value of that character. For example, 'A' returns 65. So, '\n' is an integer, but "\n" is the newline character.