/* Test and timing harness program for developing a dense matrix
   multiplication routine for the CS3014 module */

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <assert.h>
//#include <omp.h>
#include <stdint.h>
#include <pthread.h>
#include <x86intrin.h>


/* the following two definitions of DEBUGGING control whether or not
   debugging information is written out. To put the program into
   debugging mode, uncomment the following line: */
/*#define DEBUGGING(_x) _x */
/* to stop the printing of debugging information, use the following line: */
#define DEBUGGING(_x)

struct complex {
	float real;
	float imag;
};

/* write matrix to stdout */
void write_out(struct complex ** a, int dim1, int dim2)
{
	int i, j;

	for ( i = 0; i < dim1; i++ ) {
		for ( j = 0; j < dim2 - 1; j++ ) {
			printf("%.3f + %.3fi ", a[i][j].real, a[i][j].imag);
		}
		printf("%.3f + %.3fi\n", a[i][dim2-1].real, a[i][dim2-1].imag);
	}
}


/* create new empty matrix */
struct complex ** new_empty_matrix(int dim1, int dim2)
{
	struct complex ** result = calloc(dim1, sizeof(struct complex *));
	struct complex * new_matrix = calloc(dim1 * dim2, sizeof(struct complex));
	int i;

	for ( i = 0; i < dim1; i++ ) {
		result[i] = &(new_matrix[i*dim2]);
	}

	return result;
}

void free_matrix(struct complex ** matrix) {
	free (matrix[0]); /* free the contents */
	free (matrix); /* free the header */
}

/* take a copy of the matrix and return in a newly allocated matrix */
struct complex ** copy_matrix(struct complex ** source_matrix, int dim1, int dim2)
{
	int i, j;
	struct complex ** result = new_empty_matrix(dim1, dim2);

	for ( i = 0; i < dim1; i++ ) {
		for ( j = 0; j < dim2; j++ ) {
			result[i][j] = source_matrix[i][j];
		}
	}

	return result;
}

/* create a matrix and fill it with random numbers */
struct complex ** gen_random_matrix(int dim1, int dim2)
{
	const int random_range = 512; // constant power of 2
	struct complex ** result;
	int i, j;
	struct timeval seedtime;
	int seed;

	result = new_empty_matrix(dim1, dim2);

	/* use the microsecond part of the current time as a pseudorandom seed */
	gettimeofday(&seedtime, NULL);
	seed = seedtime.tv_usec;
	srandom(seed);

	/* fill the matrix with random numbers */
	for ( i = 0; i < dim1; i++ ) {
		for ( j = 0; j < dim2; j++ ) {
			/* evenly generate values in the range [0, random_range-1)*/
			result[i][j].real = (float)(random() % random_range);
			result[i][j].imag = (float)(random() % random_range);

			/* at no loss of precision, negate the values sometimes */
			/* so the range is now (-(random_range-1), random_range-1)*/
			if (random() & 1) result[i][j].real = -result[i][j].real;
			if (random() & 1) result[i][j].imag = -result[i][j].imag;
		}
	}

	return result;
}

/* check the sum of absolute differences is within reasonable epsilon */
/* returns number of differing values */
void check_result(struct complex ** result, struct complex ** control, int dim1, int dim2)
{
	int i, j;
	double sum_abs_diff = 0.0;
	const double EPSILON = 0.0625;

	for ( i = 0; i < dim1; i++ ) {
		for ( j = 0; j < dim2; j++ ) {
			double diff;
			diff = abs(control[i][j].real - result[i][j].real);
			sum_abs_diff = sum_abs_diff + diff;

			diff = abs(control[i][j].imag - result[i][j].imag);
			sum_abs_diff = sum_abs_diff + diff;
		}
	}

	if ( sum_abs_diff > EPSILON ) {
		fprintf(stderr, "WARNING: sum of absolute differences (%f) > EPSILON (%f)\n",
				sum_abs_diff, EPSILON);
	}
}

/* multiply matrix A times matrix B and put result in matrix C */
void matmul(struct complex ** A, struct complex ** B, struct complex ** C, int a_dim1, int a_dim2, int b_dim2)
{
	int i, j, k;

	for ( i = 0; i < a_dim1; i++ ) {
		for( j = 0; j < b_dim2; j++ ) {
			struct complex sum;
			sum.real = 0.0;
			sum.imag = 0.0;
			for ( k = 0; k < a_dim2; k++ ) {
				// the following code does: sum += A[i][k] * B[k][j];
				struct complex product;
				product.real = A[i][k].real * B[k][j].real - A[i][k].imag * B[k][j].imag;
				product.imag = A[i][k].real * B[k][j].imag + A[i][k].imag * B[k][j].real;
				sum.real += product.real;
				sum.imag += product.imag;
			}
			C[i][j] = sum;
		}
	}
}

#define NUM_THREADS	56

struct complex **gA, **gB, **gC;
int ga_rows, ga_cols, gb_cols, g_size;

float **a_real, **a_imag, **b_real, **b_imag;

typedef union {
	__m128 vector;
	float access[4];
} Vector;

void *go(void *i) {
	int32_t j, k;
	long ii = (long)i;

	Vector add_real, add_imag;

	while (ii < ga_rows) {
		for (j = 0; j < gb_cols; j++) {
			for (k = 0; k < ga_cols-3; k += 4) {
				__m128 va_real = _mm_setr_ps(a_real[ii][k], a_real[ii][k+1], a_real[ii][k+2], a_real[ii][k+3]);
				__m128 va_imag = _mm_setr_ps(a_imag[ii][k], a_imag[ii][k+1], a_imag[ii][k+2], a_real[ii][k+3]);
				__m128 va_nmag = _mm_setr_ps(-a_imag[ii][k], -a_imag[ii][k+1], -a_imag[ii][k+2], -a_imag[ii][k+3]);
				__m128 vb_real = _mm_setr_ps(b_real[ii][k], b_real[ii][k+1], b_real[ii][k+2], b_real[ii][k+3]);
				__m128 vb_imag = _mm_setr_ps(b_imag[ii][k], b_imag[ii][k+1], b_imag[ii][k+2], b_imag[ii][k+3]);

				__m128 mul0 = _mm_mul_ps(va_real, vb_real);
				__m128 mul1 = _mm_mul_ps(va_nmag, vb_imag);
				__m128 mul2 = _mm_mul_ps(va_real, vb_imag);
				__m128 mul3 = _mm_mul_ps(va_imag, vb_real);

				add_real.vector = _mm_mul_ps(mul0, mul1);
				add_imag.vector = _mm_mul_ps(mul2, mul3);

				gC[ii][j].real += add_real.access[0]+add_real.access[1]+add_real.access[2]+add_real.access[3];
				gC[ii][j].imag += add_imag.access[0]+add_imag.access[1]+add_imag.access[2]+add_imag.access[3];
			}
		}
		ii += NUM_THREADS;
	}

	return NULL;
}


/*
void *go(void *i) {
	int32_t j, k;
	long ii = (long)i;

	Vector v00;
	Vector v01;
	Vector v10;
	Vector v11;
	Vector v20;
	Vector v21;
	Vector v30;
	Vector v31;


	while (ii < gb_cols) {
		for (j = 0; j < ga_rows-3; j += 4) {
			for (k = 0; k < ga_rows-3; k += 4) {
				struct complex temp_a00 = gA[j][k];
				struct complex temp_a01 = gA[j+1][k];
				struct complex temp_a02 = gA[j+2][k];
				struct complex temp_a03 = gA[j+3][k];
				struct complex temp_a10 = gA[j][k+1];
				struct complex temp_a11 = gA[j+1][k+1];
				struct complex temp_a12 = gA[j+2][k+1];
				struct complex temp_a13 = gA[j+3][k+1];
				struct complex temp_a20 = gA[j][k+2];
				struct complex temp_a21 = gA[j+1][k+2];
				struct complex temp_a22 = gA[j+2][k+2];
				struct complex temp_a23 = gA[j+3][k+2];
				struct complex temp_a30 = gA[j][k+3];
				struct complex temp_a31 = gA[j+1][k+3];
				struct complex temp_a32 = gA[j+2][k+3];
				struct complex temp_a33 = gA[j+3][k+3];

				struct complex temp_b0 = gB[k][ii];
				struct complex temp_b1 = gB[k+1][ii];
				struct complex temp_b2 = gB[k+2][ii];
				struct complex temp_b3 = gB[k+3][ii];

				__m128 a0_real = _mm_setr_ps(temp_a00.real, temp_a01.real, temp_a02.real, temp_a03.real);
				__m128 a0_imag = _mm_setr_ps(temp_a00.imag, temp_a01.imag, temp_a02.imag, temp_a03.imag);
				__m128 a0_nmag = _mm_setr_ps(-temp_a00.imag, -temp_a01.imag, -temp_a02.imag, -temp_a03.imag);

				__m128 a1_real = _mm_setr_ps(temp_a10.real, temp_a11.real, temp_a12.real, temp_a13.real);
				__m128 a1_imag = _mm_setr_ps(temp_a10.imag, temp_a11.imag, temp_a12.imag, temp_a13.imag);
				__m128 a1_nmag = _mm_setr_ps(-temp_a10.imag, -temp_a11.imag, -temp_a12.imag, -temp_a13.imag);

				__m128 a2_real = _mm_setr_ps(temp_a20.real, temp_a21.real, temp_a22.real, temp_a23.real);
				__m128 a2_imag = _mm_setr_ps(temp_a20.imag, temp_a21.imag, temp_a22.imag, temp_a23.imag);
				__m128 a2_nmag = _mm_setr_ps(-temp_a20.imag, -temp_a21.imag, -temp_a22.imag, -temp_a23.imag);

				__m128 a3_real = _mm_setr_ps(temp_a30.real, temp_a31.real, temp_a32.real, temp_a33.real);
				__m128 a3_imag = _mm_setr_ps(temp_a30.imag, temp_a31.imag, temp_a32.imag, temp_a33.imag);
				__m128 a3_nmag = _mm_setr_ps(-temp_a30.imag, -temp_a31.imag, -temp_a32.imag, -temp_a33.imag);


				__m128 b0_real = _mm_setr_ps(temp_b0.real, temp_b0.real, temp_b0.real, temp_b0.real);
				__m128 b0_imag = _mm_setr_ps(temp_b0.imag, temp_b0.imag, temp_b0.imag, temp_b0.imag);

				__m128 b1_real = _mm_setr_ps(temp_b1.real, temp_b1.real, temp_b1.real, temp_b1.real);
				__m128 b1_imag = _mm_setr_ps(temp_b1.imag, temp_b1.imag, temp_b1.imag, temp_b1.imag);

				__m128 b2_real = _mm_setr_ps(temp_b2.real, temp_b2.real, temp_b2.real, temp_b2.real);
				__m128 b2_imag = _mm_setr_ps(temp_b2.imag, temp_b2.imag, temp_b2.imag, temp_b2.imag);

				__m128 b3_real = _mm_setr_ps(temp_b3.real, temp_b3.real, temp_b3.real, temp_b3.real);
				__m128 b3_imag = _mm_setr_ps(temp_b3.imag, temp_b3.imag, temp_b3.imag, temp_b3.imag);


				__m128 mul00 = _mm_mul_ps(a0_real, b0_real);
				__m128 mul01 = _mm_mul_ps(a0_nmag, b0_imag);
				__m128 mul02 = _mm_mul_ps(a0_real, b0_imag);
				__m128 mul03 = _mm_mul_ps(a0_imag, b0_real);

				__m128 mul10 = _mm_mul_ps(a1_real, b1_real);
				__m128 mul11 = _mm_mul_ps(a1_nmag, b1_imag);
				__m128 mul12 = _mm_mul_ps(a1_real, b1_imag);
				__m128 mul13 = _mm_mul_ps(a1_imag, b1_real);

				__m128 mul20 = _mm_mul_ps(a2_real, b2_real);
				__m128 mul21 = _mm_mul_ps(a2_nmag, b2_imag);
				__m128 mul22 = _mm_mul_ps(a2_real, b2_imag);
				__m128 mul23 = _mm_mul_ps(a2_imag, b2_real);

				__m128 mul30 = _mm_mul_ps(a3_real, b3_real);
				__m128 mul31 = _mm_mul_ps(a3_nmag, b3_imag);
				__m128 mul32 = _mm_mul_ps(a3_real, b3_imag);
				__m128 mul33 = _mm_mul_ps(a3_imag, b3_real);



				v00.vector = _mm_add_ps(mul00, mul01);
				v01.vector = _mm_add_ps(mul02, mul03);

				v10.vector = _mm_add_ps(mul10, mul11);
				v11.vector = _mm_add_ps(mul12, mul13);

				v20.vector = _mm_add_ps(mul20, mul21);
				v21.vector = _mm_add_ps(mul22, mul23);

				v30.vector = _mm_add_ps(mul30, mul31);
				v31.vector = _mm_add_ps(mul32, mul33);


				gC[j][ii].real += v00.access[0] + v10.access[0] + v20.access[0] + v30.access[0];
				gC[j][ii].imag += v01.access[0] + v11.access[0] + v21.access[0] + v31.access[0];
				gC[j+1][ii].real += v00.access[1] + v10.access[1] + v20.access[1] + v30.access[1];
				gC[j+1][ii].imag += v01.access[1] + v11.access[1] + v21.access[1] + v31.access[1];
				gC[j+2][ii].real += v00.access[2] + v10.access[2] + v20.access[2] + v30.access[2];
				gC[j+2][ii].imag += v01.access[2] + v11.access[2] + v21.access[2] + v31.access[2];
				gC[j+3][ii].real += v00.access[3] + v10.access[3] + v20.access[3] + v30.access[3];
				gC[j+3][ii].imag += v01.access[3] + v11.access[3] + v21.access[3] + v31.access[3];
			}
		}

		ii += NUM_THREADS;
	}
	return NULL;
}*/



/* the fast version of matmul written by the team */
void team_matmul(struct complex ** A, struct complex ** B, struct complex ** C, int a_rows, int a_cols, int b_cols) {
	gA = A;
	gB = B;
	gC = C;
	ga_rows = a_rows;
	ga_cols = a_cols;
	gb_cols = b_cols;
	g_size = a_rows * b_cols;

	long i, j;

	// make a new array for the real and imaginary parts of a
	a_real = malloc(sizeof(float *) * a_rows);
	a_imag = malloc(sizeof(float *) * a_rows);
	// make a new array for the real and imaginary parts of b transpose
	b_real = malloc(sizeof(float *) * a_rows);
	b_imag = malloc(sizeof(float *) * a_rows);

	// populate a
	for (i = 0; i < a_rows; i++) {
		a_real[i] = malloc(sizeof(float) * a_cols);
		a_imag[i] = malloc(sizeof(float) * a_cols);
		for (j = 0; j < a_cols; j++) {
			a_real[i][j] = A[i][j].real;
			a_imag[i][j] = A[i][j].imag;
		}
	}

	// populate and transpose b
	for (i = 0; i < a_rows; i++) {
		b_real[i] = malloc(sizeof(float) * b_cols);
		b_imag[i] = malloc(sizeof(float) * b_cols);
		for (j = 0; j < b_cols; j++) {
			b_real[i][j] = B[j][i].real;
			b_imag[i][j] = B[j][i].imag;
		}
	}

/*	pthread_t t[NUM_THREADS];
	for (i = 0; i < NUM_THREADS; i++) {
		pthread_create(&t[i], NULL, go, (void *)i);
	}
	//go((void *)NUM_THREADS);
	for (i = 0; i < NUM_THREADS; i++) {
		pthread_join(t[i], NULL);
	}*/
}

long long time_diff(struct timeval * start, struct timeval * end) {
	return (end->tv_sec - start->tv_sec) * 1000000L + (end->tv_usec - start->tv_usec);
}

int main(int argc, char ** argv)
{
	struct complex ** A, ** B, ** C;
	struct complex ** control_matrix;
	long long control_time, mul_time;
	double speedup;
	int a_dim1, a_dim2, b_dim1, b_dim2, errs;
	struct timeval pre_time, start_time, stop_time;

	if ( argc != 5 ) {
		fprintf(stderr, "Usage: matmul-harness <A nrows> <A ncols> <B nrows> <B ncols>\n");
		exit(1);
	}
	else {
		a_dim1 = atoi(argv[1]);
		a_dim2 = atoi(argv[2]);
		b_dim1 = atoi(argv[3]);
		b_dim2 = atoi(argv[4]);
	}

	/* check the matrix sizes are compatible */
	if ( a_dim2 != b_dim1 ) {
		fprintf(stderr,
				"FATAL number of columns of A (%d) does not match number of rows of B (%d)\n",
				a_dim2, b_dim1);
		exit(1);
	}

	/* allocate the matrices */
	A = gen_random_matrix(a_dim1, a_dim2);
	B = gen_random_matrix(b_dim1, b_dim2);
	C = new_empty_matrix(a_dim1, b_dim2);
	control_matrix = new_empty_matrix(a_dim1, b_dim2);

	DEBUGGING( {
			printf("matrix A:\n");
			write_out(A, a_dim1, a_dim2);
			printf("\nmatrix B:\n");
			write_out(A, a_dim1, a_dim2);
			printf("\n");
			} )

	/* record control start time */
	gettimeofday(&pre_time, NULL);

	/* use a simple matmul routine to produce control result */
	matmul(A, B, control_matrix, a_dim1, a_dim2, b_dim2);

	/* record starting time */
	gettimeofday(&start_time, NULL);

	/* perform matrix multiplication */
	team_matmul(A, B, C, a_dim1, a_dim2, b_dim2);

	/* record finishing time */
	gettimeofday(&stop_time, NULL);

	/* compute elapsed times and speedup factor */
	control_time = time_diff(&pre_time, &start_time);
	mul_time = time_diff(&start_time, &stop_time);
	speedup = (float) control_time / mul_time;

	printf("Matmul time: %lld microseconds\n", mul_time);
	printf("control time : %lld microseconds\n", control_time);
	if (mul_time > 0 && control_time > 0) {
		printf("speedup: %.2fx\n", speedup);
	}

	/* now check that the team's matmul routine gives the same answer
	   as the known working version */
	check_result(C, control_matrix, a_dim1, b_dim2);

	/* free all matrices */
	free_matrix(A);
	free_matrix(B);
	free_matrix(C);
	free_matrix(control_matrix);

	return 0;
}

