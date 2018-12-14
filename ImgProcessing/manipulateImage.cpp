#include "CImg.h"
#include <cstdlib>
#include <iostream>
#include <vector>
#include <stdlib.h>
#include <string>
#include <ctime>

using namespace std;
using namespace cimg_library;


void invert(CImg<double> * input) {
    CImg<double> output(input->width(), input->height(), 1, input->spectrum());

    for (int x = 0; x < input->width(); x++) {
	for (int y = 0; y < input->height(); y++) {
	    for (int z = 0 ; z < input->spectrum(); z++) {
		    output(input->width()-1-x,y,0,z) = (*input)(x,y,z);
	    }
	}
    }

    *input = output;
}

double * getMatrix(int kernelSize, int x, int y, int z, CImg<double> * input) {
    // Dynamically allocate the matrix
    double * matrix = (double*) malloc(kernelSize*kernelSize*sizeof(double));
    int index = 0;
    int margin = kernelSize/2;

    for (int inX = x-margin ; inX <= x + margin ; inX++) {
	for (int inY = y-margin; inY <= y + margin ; inY++) {
	    // Check that we always grab values from within bounds of input image's arrays
	    int tempX, tempY;
	    if (inX<0) tempX = 0;
	    else if (inX>=input->width()) tempX = input->width() - 1;
	    else tempX = inX;
	    if (inY<0) tempY = 0;
	    else if (inY>=input->height()) tempY = input->height() - 1;
	    else tempY = inY;

	    matrix[index] = (*input)(tempX, tempY, z);

	    index++;
	}
    }

    return matrix;
}

void printMatrix(double * arr, int kernelSize){
    for (int x = 0; x < kernelSize*kernelSize ; x++) printf("%lf   ", arr[x]);
    printf("\n");
}

int compare(const void * a, const void * b) {
    return (int) (*(double *)a-*(double *)b);
}

CImg<double> meanFilter(int kernelSize, CImg<double> * input) {
    CImg<double> output(input->width(), input->height(), 1, input->spectrum());

    for (int x = 0; x < input->width(); x++) {
	for (int y = 0; y < input->height(); y++) {
	    for (int z = 0 ; z < input->spectrum(); z++) {
		    double * matrix = NULL;
		    matrix = getMatrix(kernelSize, x, y, z, input);
		    double sum = 0;
		    for (int a = 0; a < kernelSize*kernelSize ; a++) sum += matrix[a];

		    double mean = sum/ ((double) kernelSize*kernelSize);
		    output(x,y,0,z) = mean;
		    free(matrix);
		    // printf("Median: %lf\n", median);

	    }
	}
    }

    input->empty();
    return output;
}

CImg<double> medianFilter(int kernelSize, CImg<double> * input) {
    CImg<double> output(input->width(), input->height(), 1, input->spectrum());

    for (int x = 0; x < input->width(); x++) {
	for (int y = 0; y < input->height(); y++) {
	    for (int z = 0 ; z < input->spectrum(); z++) {
		    double * matrix = NULL;
		    matrix = getMatrix(kernelSize, x, y, z, input);
		    // printf("Matrix (%d,%d): ", x,y);
		    // printMatrix(matrix, kernelSize);
		    qsort(matrix, kernelSize*kernelSize, sizeof(double), compare);

		    // printf("Sorted(%d,%d): ", x,y);
		    // printMatrix(matrix, kernelSize);

		    double median = matrix[(kernelSize*kernelSize)/2];
		    free(matrix);
		    output(x,y,0,z) = median;
		    // printf("Median: %lf\n", median);

	    }
	}
    }

    input->empty();
    return output;
}

int main (int argc, char **argv) {
    if (argc != 3) {
	cout << "Incorrect number of arguments.  Usage: " << endl;
	cout << "cimg-demo input_file output_file" << endl;
	return -1;
    }



    char *input_file = argv[1];
    char *output_file = argv[2];

    cout << "Input : " << input_file << endl;
    cout << "Output : " << output_file << endl;

    CImg<double> input(input_file);


    invert(&input);
    input = medianFilter(9, &input);
    input = meanFilter(9, &input);

    input.save(output_file);

    return 0;
}


