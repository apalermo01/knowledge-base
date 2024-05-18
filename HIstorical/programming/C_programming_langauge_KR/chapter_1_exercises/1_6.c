/*
Verify that the expression getchar() != EOF is 0 or 1

This also serves the same ask as Q 1_7
*/

#include <stdio.h>

int main(){
    printf("EOF is %d", getchar()!=EOF);
}