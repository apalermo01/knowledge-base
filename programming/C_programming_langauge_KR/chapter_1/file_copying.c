/*
copy a program's input to its output once character at a time
*/


#include <stdio.h>

int main_v1(){
    int c;

    c = getchar();
    while (c != EOF) {
        putchar(c);
        c = getchar();
    }
}

// a more concise version

int main() {
    int c;

    while ((c = getchar()) != EOF)
        putchar(c);
}