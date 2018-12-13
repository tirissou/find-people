#include "CImg.h"
#include <cstdlib>
#include <iostream>
#include <vector>
#include <stdlib.h>
#include <string>
#include <ctime>

using namespace std;
using namespace cimg_library;


CImg<double> invert(CImg<double> input) {
    CImg<double> output(input.width(), input.height(), 1, input.spectrum());

    for (int x = 0; x < input.width(); x++) {
	for (int y = 0; y < input.height(); y++) {
	    for (int z = 0 ; z < input.spectrum(); z++) {
		    output(input.width()-1-x,y,0,z) = input(x,y,z);
	    }
	}
    }

    input = CImg<double>::empty();
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

    if (input.spectrum() == 1) {
	cout << "Input file is already greyscale" << endl;
	return -1;
    }

    input = invert(input);

    input.save(output_file);
}


