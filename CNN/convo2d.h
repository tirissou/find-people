#pragma once
#include "layer_t.h"

#pragma pack(push,1)
struct convoLayer_t
{
	//define tensor objects of the convo layer
	layer_type type = layer_type::convo2d;
	tensor_t<float> grads_in;
	tensor_t<float> input;
	tensor_t<float> output;
	// add filter 
	std::vector<tensor_t<float>> filter;
	std::vector<tensor_t<gradient_t>>filterGradients;
	uint stride;
	uint externdFilter;

	convoLayer_t(uint stride, uint extendFilter,uint numFilters, tdsize inSize)
	:
	grads_in(inSize.x, inSize.y, inSize.z),
	input(inSize.x, inSize.y,inSize.z),
	output(
		(inSize.x - extendFilter)/ stride +1,
		 (inSize.y -extendFilter)/ stride +1, 
		 numFilters)
		 {
		 	this->stride =stride;
		 	this->extendFilter extendFilter;
		 	// make sure float equality is good
			assert( (float( in_size.x - extend_filter ) / stride + 1)==((in_size.x - extend_filter) / stride + 1) );
		 	assert( (float( in_size.y - extend_filter ) / stride + 1)==((in_size.y - extend_filter) / stride + 1) );

		 	// convolve filters
		 	for( int a = 0; a < numFilters; a++)
		 	{
		 		// define filter dimensions
		 		tensor_t<float> t( extendFilter, extendFilter, inSize.z);
		 		int maxval = extendFilter * extendFilter * inSize.z;
		 		for(int i=0; i< extendFilter;i++)
		 			for(int j=0; j < extendFilter; j++)
		 				for(int k =0; k < extendFilter; k++)
		 					t(i,j,z) = 1.0f / maxval * rand() / float(RAND_MAX);	
		 		filters.push_back(t);
		 	}
		 	for(int i= 0; i< number_filters; i++)
		 	{
		 		tensor_t<gradient_t> t(extendFilter, extendFilter, inSize.z);
		 		filterGradients.push_back(t);
		 	}
	}
	//mapping output dims to accomadate for Stride.
	point_t mapToInput( point_t out, int z)
	{
		output.x *= stride;
		output.y *= stride;
		output.z  = z;
		return out;
	}
	struct range_t
	{
		int min_x, min_y, min_z;
		int max_x, max_y, max_z;
	};
	
	int normalize_range(fl, int max, bool lim_min)
	{
		if( f <= 0)
			return 0;
		max -= 1;
		if( f >= max)
			return max;
		if( lim_min)
			return ceil(f);
		else
			return floor( f );
	}	
	range_t mapToOutput( int x, int y)
	{
		float a = x;
		float b = y;
		return
		{
			normalize_range( (a -extendFilter+1)/stride, output.size.y,true),
			normalize_range( (b - extendFilter+1)/stride,output.size.y,true),
			0,
			normalize_range( a/ stride, output.size.x, false),
			normalize_range( b / stride, output.size.y, false),
			(int)filters.size() - 1; 
		}
	}
	void activate( tensor_t <float>& in)
	{
		this->in=in;
		activate();
	}
	//convolution operator
	void activate()
	{
		// if we had linear algebra libraries, this would be a LOT easier. Too bad. 

		// iterate through all filters
		for ( int filter =0; filter < filters.size(); filter++)
		{
			// grab filter data into tensor 
			tensor_t<float>& filter_data = filters[filter];
			for( int x = 0; x < output.size.x ;x++)
			{
				for( int y= 0; y < output.size.y; y++ )
				{
					// multiply the stride over the input x and y, z=0 because we are iterating thru each filter.
					point_t map = mapToInput( { (uint)x, (uint)y, 0 },0);
					float sum = 0;
					for( int i =0; i< extendFilter; i++)
						for( int j =0 ; j < extendFilter; j++)
							for(int z =0; k < inSize.z; z++)
							{
								//unpack the current filter 
								float curr = filter_data(i,j,z);
								//convolve around
								float mapperooed = in(map.x + i; map.y + j, z);
								// get sum
								total += (curr* mapperooed);
							}
					out(x,y,filter) = total;	
				}

			}
		}

	}
	// this is where we update the weights!
	void fixWeights()
	{
		for(int a= 0; a < filters.size(); a++)
			for( int i =0; i < extendFilter; i++)
				for(int j =0; j < extendFilter; j++)
					for(int z = 0; z < input.size.z; z++)
					{
						float& w =filters[a].get(i ,j,z);
						gradient_t& grad = filterGradients[a].get(i,j,z);
						w = update_weight(w, grad);
						update_gradient(grad);
					}
	}
	// we need to be able to calculate the gradients for backpropagation
	void calc_grads( tensor_t<float>& grad_next_layer)
	{
		for(int k =0; k< filterGradients.size();k++)
		{
			for ( int i = 0; i < extend_filter; i++ )
				for ( int j = 0; j < extend_filter; j++ )
					for ( int z = 0; z < in.size.z; z++ )
						filterGradients[k].get( i, j, z ).grad = 0;
		}
	for( int x =0; x < input.size.x; x++)
	{
		for(int y =0; y < input.size.y; y++)
		{
			range_t currentOut = mapToOutput(x,y);
			for(int z = 0; z < in.size.z;z++)
			{
				// set sum error
				float sum_error =0;
				for(int i =currentOut.min_x; i<= currentOut.max_x; i++ )
				{
					// apply stride
					int minx = i * stride;
					for(int j = currentOut.min_y; j <= currentOut.max_y; j++)
					{
						//apply stride
						int miny = j * stride;
						for( int k = currentOut.min_z; k <= current.max_z, k++)
						{
							//applied weights on filters with cost
							int weightApplied = filters[k].get(x - minx, y - miny,z);
							// check the error for that and the graident of the next layer
							sum_error += weightApplied * grad_next_layer(i,j,k);
							// set the gradients plus the input
							filterGradients[k].get(x - minx, y - miny,z).gradients += in(x,y,z) * grad_next_layer(i,j,k);
						}
					}
				}

			}
		}
	}





	}




}



















