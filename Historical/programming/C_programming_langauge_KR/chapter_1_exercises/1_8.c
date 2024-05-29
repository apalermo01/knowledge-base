#include <stdio.h>

/*
Write a program to count blanks, tabs, and newlines
*/

int main(){
    int c;

    int blanks = 0;
    int tabs = 0;
    int newlines = 0;

    while ((c = getchar()) != EOF){
        if (c == '\n')
            newlines++;
        
        if (c == '\t')
            tabs ++;
        
        if (c == ' ')
            blanks ++;
    }

    printf("spaces: %d, tabs: %d, newlines %d\n", blanks, tabs, newlines);

}