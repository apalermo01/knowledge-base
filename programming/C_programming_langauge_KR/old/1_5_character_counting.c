#include <stdio.h>

/* count characters in an input; 1st version */
int main_version1(){
	long nc;

	nc = 0;
	while (getchar() != EOF)
		++nc;
	printf("%ld\n", nc);
}

/* count characters in input; 2nd version */ 

main() {
	double nc;
	for (nc = 0; getchar() != EOF; ++nc)
		// for loop is empty because the definition of the loop does all the work
		// if there is no input, nc=0 still runs, so 0 is returned
		;
	printf("%.0f\n", nc);
}
