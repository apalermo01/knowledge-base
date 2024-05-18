#include <stdio.h>

/* copy input to output; 1st version */
int main_old1(){
	int c;

	c = getchar();
	while (c != EOF) {
		putchar(c);
		printf("===========");
		c = getchar();
	}

}

/* second version */
int main(){
	int c;
	printf("EOF is %d", EOF);
	printf("getchar() != EOF is %d", getchar() != EOF);
	while ((c=getchar()) != EOF)
		putchar(c);
}
