#include <stdio.h>

/*
write a program to copy its input to its output, replacing each tab by \t, each
backspace by \b, and each backslash by \\
*/


int main(){
    int c;

    while ((c = getchar()) != EOF){

        if (c == '\t'){
            putchar("\t");
        }

        else if (c == '\b'){
            putchar("\b");
        }

        else if (c == '\\'){
            putchar("\\");
        }

        else {
            putchar(c);
        }
    }
}