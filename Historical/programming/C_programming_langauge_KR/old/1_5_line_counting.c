#include <stdio.h>

/* count lines in the input*/


struct whitespace_counts {
	int newlines;
	int tabs;
	int spaces;
} ex_1_8_res;;

int exercsie_1_8();
int exercise_1_9();
int exercise_1_10();

main_old() {
	int c, nl;

	nl = 0;

	while ((c = getchar()) != EOF)
		// character written between single quotes is a
		// CHARACTER CONSTNANT
		// this means an integer value equal to the numerical value of the character
		// on the machine's character set
		// NOTE: '\n' is a single character and evaluates to an integer
		// "\n" is a string constant that contains one character
		if (c == '\n')
			++nl;
	printf("%d\n", nl);

}

int main() {
	// exercise_1_8();
	// printf("%d newlines, %d tabs, and %d spaces\n", ex_1_8_res.newlines, ex_1_8_res.tabs, ex_1_8_res.spaces);
	exercise_1_9();
	printf("success");
}

int exercsie_1_8() {
	int c;
	int spaces = 0;
	int tabs = 0;
	int nl = 0;

	while ((c = getchar()) != EOF) {
		printf("c = %d\n", c);
		if (c == ' ') {
			printf("space");
			spaces++;
		}
		if (c == '\t') {
			printf("tab");
			++tabs;
		}
		if (c == '\n') {
			printf("newline");
			++nl;
		}
	}
	ex_1_8_res.spaces = spaces;
	ex_1_8_res.newlines = nl;
	ex_1_8_res.tabs = tabs;
	return 0;
}

int exercise_1_9() {
	/* write a program to copy its input to its output, replacing each
	 * string of one or more blanks by a single blank*/

	int c;

	while ((c = getchar()) != EOF) {
		if (c == ' ') {
			while ((c = getchar()) == ' ') {;}
			putchar(' ');
		}
		putchar(c);
	}
	return 0;
}
