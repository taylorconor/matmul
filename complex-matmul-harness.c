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

typedef union {
	__m128 vector;
	float access[4];
} Vector;

void *go(void *i) {
	int32_t j, k;
	long ii = (long)i;

    Vector v0;
    Vector v1;

	while (ii < gb_cols) {
		for (j = 0; j < ga_rows-3; j += 4) {
			//uint32_t result_row = ii / ga_rows, 
			//		 result_col = ii % ga_rows;

			for (k = 0; k < ga_rows-3; k += 4) {
				//struct complex temp_a = gA[j][k];
				//struct complex temp_b = gB[k][ii];

				// B array only requires one column
				struct complex temp_b0 = gB[k][ii];
				struct complex temp_b1 = gB[k+1][ii];
				struct complex temp_b2 = gB[k+2][ii];
				struct complex temp_b3 = gB[k+3][ii];
				__m128 b_real = _mm_setr_ps(temp_b0.real, temp_b1.real, temp_b2.real, temp_b3.real);
				__m128 b_imag = _mm_setr_ps(temp_b0.imag, temp_b1.imag, temp_b2.imag, temp_b3.imag);

				// A array requires 4 rows (since we unrolled the look into 4)
				struct complex temp_a00 = gA[j][k];
				struct complex temp_a01 = gA[j][k+1];
				struct complex temp_a02 = gA[j][k+2];
				struct complex temp_a03 = gA[j][k+3];
				__m128 a0_real = _mm_setr_ps(temp_a00.real, temp_a01.real, temp_a02.real, temp_a03.real);
				__m128 a0_imag = _mm_setr_ps(temp_a00.imag, temp_a01.imag, temp_a02.imag, temp_a03.imag);
				__m128 a0_nimag = _mm_setr_ps(-temp_a00.imag, -temp_a01.imag, -temp_a02.imag, -temp_a03.imag);

				struct complex temp_a10 = gA[j+1][k];
				struct complex temp_a11 = gA[j+1][k+1];
				struct complex temp_a12 = gA[j+1][k+2];
				struct complex temp_a13 = gA[j+1][k+3];
				__m128 a1_real = _mm_setr_ps(temp_a10.real, temp_a11.real, temp_a12.real, temp_a13.real);
				__m128 a1_imag = _mm_setr_ps(temp_a10.imag, temp_a11.imag, temp_a12.imag, temp_a13.imag);
				__m128 a1_nimag = _mm_setr_ps(-temp_a10.imag, -temp_a11.imag, -temp_a12.imag, -temp_a13.imag);

				struct complex temp_a20 = gA[j+2][k];
				struct complex temp_a21 = gA[j+2][k+1];
				struct complex temp_a22 = gA[j+2][k+2];
				struct complex temp_a23 = gA[j+2][k+3];
				__m128 a2_real = _mm_setr_ps(temp_a20.real, temp_a21.real, temp_a22.real, temp_a23.real);
				__m128 a2_imag = _mm_setr_ps(temp_a20.imag, temp_a21.imag, temp_a22.imag, temp_a23.imag);
				__m128 a2_nimag = _mm_setr_ps(-temp_a20.imag, -temp_a21.imag, -temp_a22.imag, -temp_a23.imag);

				struct complex temp_a30 = gA[j+3][k];
				struct complex temp_a31 = gA[j+3][k+1];
				struct complex temp_a32 = gA[j+3][k+2];
				struct complex temp_a33 = gA[j+3][k+3];
				__m128 a3_real = _mm_setr_ps(temp_a30.real, temp_a31.real, temp_a32.real, temp_a33.real);
				__m128 a3_imag = _mm_setr_ps(temp_a30.imag, temp_a31.imag, temp_a32.imag, temp_a33.imag);
				__m128 a3_nimag = _mm_setr_ps(-temp_a30.imag, -temp_a31.imag, -temp_a32.imag, -temp_a33.imag);

				
				// a.real * b.real mul
				__m128 mul0_pr0 = _mm_mul_ps(a0_real, b_real); 				
				__m128 mul0_pr1 = _mm_mul_ps(a1_real, b_real); 				
				__m128 mul0_pr2 = _mm_mul_ps(a2_real, b_real); 				
				__m128 mul0_pr3 = _mm_mul_ps(a3_real, b_real); 				
			
				// -a.imag * b.imag mul
				__m128 mul1_pr0 = _mm_mul_ps(a0_nimag, b_imag); 				
				__m128 mul1_pr1 = _mm_mul_ps(a1_nimag, b_imag); 				
				__m128 mul1_pr2 = _mm_mul_ps(a2_nimag, b_imag); 				
				__m128 mul1_pr3 = _mm_mul_ps(a3_nimag, b_imag); 				

				// a.real * b.real mul
				__m128 mul2_pr0 = _mm_mul_ps(a0_real, b_imag); 				
				__m128 mul2_pr1 = _mm_mul_ps(a1_real, b_imag); 				
				__m128 mul2_pr2 = _mm_mul_ps(a2_real, b_imag); 				
				__m128 mul2_pr3 = _mm_mul_ps(a3_real, b_imag); 				

				// a.imag * b.real mul
				__m128 mul3_pr0 = _mm_mul_ps(a0_imag, b_real); 				
				__m128 mul3_pr1 = _mm_mul_ps(a1_imag, b_real); 				
				__m128 mul3_pr2 = _mm_mul_ps(a2_imag, b_real); 				
				__m128 mul3_pr3 = _mm_mul_ps(a3_imag, b_real); 				

				
				// (a.real * b.real) + (-a.imag * b.imag) add	
				__m128 add0_pr0 = _mm_add_ps(mul0_pr0, mul1_pr0);
				__m128 add0_pr1 = _mm_add_ps(mul0_pr1, mul1_pr1);
				__m128 add0_pr2 = _mm_add_ps(mul0_pr2, mul1_pr2);
				__m128 add0_pr3 = _mm_add_ps(mul0_pr3, mul1_pr3);

				// (a.real * b.imag) + (a.imag * b.real) add
				__m128 add1_pr0 = _mm_add_ps(mul2_pr0, mul3_pr0);
				__m128 add1_pr1 = _mm_add_ps(mul2_pr1, mul3_pr1);
				__m128 add1_pr2 = _mm_add_ps(mul2_pr2, mul3_pr2);
				__m128 add1_pr3 = _mm_add_ps(mul2_pr3, mul3_pr3);

			/*	
				gC[j][ii].real += // horizontal add add0_pr0
				gC[j+1][ii].real += // horizontal add add0_pr1
				gC[j+2][ii].real += // horizontal add add0_pr2
				gC[j+3][ii].real += // horizontal add add0_pr3
	
				gC[j][ii].imag += // horizontal add add1_pr0
				gC[j+1][ii].imag += // horizontal add add1_pr1
				gC[j+2][ii].imag += // horizontal add add1_pr2
				gC[j+3][ii].imag += // horizontal add add1_pr3
			*/
				v0.vector = _mm_hadd_ps(add0_pr0, add0_pr0);
				v0.vector = _mm_hadd_ps(add0_pr0, add0_pr0);
				v1.vector = _mm_hadd_ps(add1_pr0, add1_pr0);
				v1.vector = _mm_hadd_ps(add1_pr0, add1_pr0);
				gC[j][ii].real += v0.access[0];
				gC[j][ii].imag += v1.access[0];
				
				v0.vector = _mm_hadd_ps(add0_pr1, add0_pr1);
				v0.vector = _mm_hadd_ps(add0_pr1, add0_pr1);
				v1.vector = _mm_hadd_ps(add1_pr1, add1_pr1);
				v1.vector = _mm_hadd_ps(add1_pr1, add1_pr1);
				gC[j+1][ii].real += v0.access[0];
				gC[j+1][ii].imag += v1.access[0];
				
				v0.vector = _mm_hadd_ps(add0_pr2, add0_pr2);
				v0.vector = _mm_hadd_ps(add0_pr2, add0_pr2);
				v1.vector = _mm_hadd_ps(add1_pr2, add1_pr2);
				v1.vector = _mm_hadd_ps(add1_pr2, add1_pr2);
				gC[j+2][ii].real += v0.access[0];
				gC[j+2][ii].imag += v1.access[0];
				
				v0.vector = _mm_hadd_ps(add0_pr3, add0_pr3);
				v0.vector = _mm_hadd_ps(add0_pr3, add0_pr3);
				v1.vector = _mm_hadd_ps(add1_pr3, add1_pr3);
				v1.vector = _mm_hadd_ps(add1_pr3, add1_pr3);
				gC[j+3][ii].real += v0.access[0];
				gC[j+3][ii].imag += v1.access[0];
				
				/*struct complex temp_a1 = gA[j][k+1];
				struct complex temp_b1 = gB[k+1][ii];


				__m128 mul0 = _mm_setr_ps(temp_a0.real, temp_a0.real, -temp_a0.imag, temp_a0.imag);
				__m128 mul1 = _mm_setr_ps(temp_b0.real, temp_b0.imag, temp_b0.imag, temp_b0.real);
            	__m128 mul2 = _mm_setr_ps(temp_a1.real, temp_a1.real, -temp_a1.imag, temp_a1.imag);
				__m128 mul3 = _mm_setr_ps(temp_b1.real, temp_b1.imag, temp_b1.imag, temp_b1.real);

	            v0.vector = _mm_mul_ps(mul0, mul1);
                v1.vector = _mm_mul_ps(mul2, mul3);
*/
				//gC[j][ii].real += (temp_a.real * temp_b.real) + (-temp_a.imag * temp_b.imag);
				//gC[j][ii].imag += (temp_a.real * temp_b.imag) + (temp_a.imag * temp_b.real);
			
				/*gC[j][ii].real += v0.access[0] + v0.access[2];
				gC[j][ii].imag += v0.access[1] + v0.access[3];

                gC[j][ii].real += v1.access[0] + v1.access[2];
				gC[j][ii].imag += v1.access[1] + v1.access[3];
           */ }
       }

		ii += NUM_THREADS;
	}
	return NULL;
}

/* the fast version of matmul written by the team */
void team_matmul(struct complex ** A, struct complex ** B, struct complex ** C, int a_rows, int a_cols, int b_cols) {
	gA = A;
	gB = B;
	gC = C;
	ga_rows = a_rows;
	ga_cols = a_cols;
	gb_cols = b_cols;
	g_size = a_rows * b_cols;
	long i;
	pthread_t t[NUM_THREADS];
	for (i = 0; i < NUM_THREADS; i++) {
		pthread_create(&t[i], NULL, go, (void *)i);
	}
	//go((void *)NUM_THREADS);
	for (i = 0; i < NUM_THREADS; i++) {
		pthread_join(t[i], NULL);
	}
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

