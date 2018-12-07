#pragma once
#include <math.h>
#include <float.h>
#include <string.h>
#include "layer_t.h"

#pragma pack(push, 1)
struct fc_layer_t
{
	layer_type type = layer_type::fc;
	tensor_t<float> grads_in;
	tensor_t<float> in;
	tensor_t<float> out;
	std::vector<float> input;
	tensor_t<float> weights;
	std::vector<gradient_t> gradients;

	fc_layer_t( tdsize in_size, int out_size )
		:
		in( in_size.x, in_size.y, in_size.z ),
		out( out_size, 1, 1 ),
		grads_in( in_size.x, in_size.y, in_size.z ),
		weights( in_size.x*in_size.y*in_size.z, out_size, 1 )
	{
		input = std::vector<float>( out_size );
		gradients = std::vector<gradient_t>( out_size );


		int maxval = in_size.x * in_size.y * in_size.z;

		for ( int i = 0; i < out_size; i++ )
			for ( int h = 0; h < in_size.x*in_size.y*in_size.z; h++ )
				weights( h, i, 0 ) = 2.19722f / maxval * rand() / float( RAND_MAX );
		// 2.19722f = f^-1(0.9) => x where [1 / (1 + exp(-x) ) = 0.9]
	}
	float activator_function(float x)
	{
		//leaky relu
		if( x > 0.01){
			return x;
		}
		else{
			return 0;
		}
	}
	// Leaky Relu's derivaitve
	float activator_derivative(float x)
	{
		if (x >=0 ){
			return 1;
		}
		else{
			return 0.01;

		}
	}
	// self explanatory
	void activate(tensor_t<float>& input)
	{
		this->input =input;
		activate();
	}
	int map(point_t d)
	{
		return d.z * (input.size.x * input.size.y) +
			d.y * (input.size.x) +
				d.x;
	}
		// self explanatory

	void activate()
	{
		for ( int n = 0; n < output.size.x; n++ )
		{
			float inputv = 0;

			for ( int i = 0; i < input.size.x; i++ )
				for ( int j = 0; j < input.size.y; j++ )
					for ( int z = 0; z < input.size.z; z++ )
					{
						int m = map( { i, j, z } );
						inputv += in( i, j, z ) * weights( m, n, 0 );
					}

			input[n] = inputv;

			out( n, 0, 0 ) = activator_function( inputv );
		}
	}
	//fixes weights, much easier than CNN layer.
	void fix_weights()
	{
		for ( int n = 0; n < output.size.x; n++ )
		{
			gradient_t& grad = gradients[n];
			for ( int i = 0; i < input.size.x; i++ )
				for ( int j = 0; j < input.size.y; j++ )
					for ( int z = 0; z < input.size.z; z++ )
					{
						int m = map( { i, j, z } );
						float& w = weights( m, n, 0 );
						w = update_weight( w, grad, in( i, j, z ) );
					}

			update_gradient( grad );
		}
	}

	void calc_grads(tensor_t<float>& grad_next_layer)
	{
			memset( grads_in.data, 0, grads_in.size.x *grads_in.size.y*grads_in.size.z * sizeof( float ) );
			for ( int n = 0; n < output.size.x; n++ )
		{
			gradient_t& grad = gradients[n];
			grad.gradients = grad_next_layer( n, 0, 0 ) * activator_derivative( input[n] );

			for ( int i = 0; i < input.size.x; i++ )
				for ( int j = 0; j < input.size.y; j++ )
					for ( int z = 0; z < input.size.z; z++ )
					{
						int m = map( { i, j, z } );
						grads_in( i, j, z ) += grad.gradients * weights( m, n, 0 );
					}
		}
	}
};
#pragma pack(pop)






















