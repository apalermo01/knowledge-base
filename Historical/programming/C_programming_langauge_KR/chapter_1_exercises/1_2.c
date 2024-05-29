/* 
Experiment to find out what happens when printf's argument string contains \c where c is a character not listed
*/

#include <stdio.h>

int main(){
    printf("test \c\n");
}

// outputs the character sequence \c