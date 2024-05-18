#include <stdio.h>

int main(){
	float fahr, celc;
	int lower, upper, step;

	lower = 0;
	upper = 300;
	step = 20;

	celc = lower;
	printf("celc\tfahr\n");
	printf("----\t----\n");

	while (fahr <= upper){
	    fahr = (9.0/5.0) * (celc + 32.0);
		printf("%4.0f\t%4.1f\n", celc, fahr);
		celc += step;
	}
}
