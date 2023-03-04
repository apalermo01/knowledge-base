#include <stdio.h>

/* print farenheit-celsius table */

/*
 * NOTE: we multiply by 5 and THEN divide by 9 instead of multiplying by 5/9
 * becuase any fractional part of integer division will truncate (i.e. decimal elements will be deleted
 */
int main_old()
{
	int fahr, celsius;
	int lower, upper, step;

	lower = 0; 				/* lower limit of temperature scale */
	upper = 300; 			/* upper limit of temperature scale */
	step = 20; 				/* step size */

	fahr = lower;
	while (fahr <= upper){
		celsius = 5 * (fahr-32) / 9;
		printf("%d\t%d\n", fahr, celsius);
		fahr = fahr + step;
	}
}


int main_old2()
{
	int fahr, celsius;
	int lower, upper, step;

	lower = 0; 				/* lower limit of temperature scale */
	upper = 300; 			/* upper limit of temperature scale */
	step = 20; 				/* step size */

	fahr = lower;
	while (fahr <= upper){
		celsius = 5 * (fahr-32) / 9;
		/* this ensures that the first number of each line in the field in 3 digits wide
		 * and the second field is 6 digits wide*/
		printf("%3d %6d\n", fahr, celsius);
		fahr = fahr + step;
	}
}

/* FLOATING POINT VERSION */
int main_old3()
{
	float fahr, celsius;
	int lower, upper, step;

	lower = 0; 				/* lower limit of temperature scale */
	upper = 300; 			/* upper limit of temperature scale */
	step = 20; 				/* step size */

	fahr = lower;
	printf("farenheit celsius\n");
	while (fahr <= upper){
		celsius = (5.0/9.0) * (fahr-32.0);
		/* don't forget to change the output format from decimal to integer */
		/* 3.0 means we want the number to be printed at least 3 characters wide, no decimal point and no fraction digits
		 * 6.1 means print the number at least 6 characters wide with 1 digit after the decimal point*/
		printf("%9.0f %7.1f\n", fahr, celsius);
		fahr = fahr + step;
	}
}

/* USING THE FOR STATEMENT */

int main() {
	int fahr;
	for (fahr = 0; fahr <= 300; fahr = fahr + 20)
		printf("%3d %6.1f\n", fahr, (5.0/9.0) * (fahr - 32));
}
