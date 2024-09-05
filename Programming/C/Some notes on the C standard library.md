# Input / Output


## printing to the screen

### printf
- writes to stdout (terminal screen)
### fprintf
- same as printf, but writes to a file stream (can be stdout, stderr, or some other file). This file is the first argument
### sprintf
- writes the character string to a buffer
- **WARNING** - this is vulnerable to memory issues. If the string to be written exceeds the size of the buffer, then the behavior is undefined.
### snprintf
- same as sprintf, but has a parameter specifying the size of the buffer to write. If the string to write exceeds bufsz, the the function will just truncate the string.
## reading input

### scanf
- reads data from stdin
### fscanf
- reads data from  a file stream (stream is specified by the first argument)
### sscanf
- reads data from a buffer until the null-terminator
- **WARNING** - there could be issues if the buffer is not properly null terminated.

## read / write  to / from stdout or files

### puts
- writes every character from null-terminated string to stdout
- success: non-negative value
- failure: returns EOF
### fputs
- same as puts, but you can specify the file stream
### gets
- reads stdin into the character array specified by the first argument until a newline or end of file
- **WARNING** - this is will be vulnerable to the case where the stdin stream is larger than the character array
### gets_s
- same as gets, but you can specify the max number of characters to read
### fgets
- reads characters from the given file stream and puts them in a char array. Specify the max number of characters to write using the second argument.

## file read / write operations

### fopen
- opens a file
### fclose
- closes a file
### fread
- read (up to some max) objects into a buffer from a file stream
### fwrite
- write a fixed number of objects from a buffer into a file stream.

## file positioning

### fseek
- sets the file position indicator for a file stream.
### ftell
- returns the file position indicator
### rewind
- moves the file position indicator to the beginning of the file stream

# String manipulation

# Memory management



# References
- https://devdocs.io/c/
