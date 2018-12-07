#pragma once
#include "layer_t.h"
#pragma pack(push,1)

struct dropout_layer_t
{
	layer_type type = layer_type::dropout_layer;
	tensor_t<float> grads_in;
	tensor_t<float> input;
	tensor_t<float> output;
	tensor_t<bool>hitmap;
	float p_val;
	dropout_layer_t( tdsize inSize, float p_val)
	:
	input(inSize.x, inSize.y, inSize.z),
	ouput(inSize.x,inSize.y,inSize.z),
	hitmap(inSize.x,inSize.y,inSize.z),
	grads_in(inSize.x,inSize.y,inSize.z),
	p_val(p_val)
	{
	}

	void activate( tensor_t<float>& input)
	{
		this->in=in;
		activate();
	}
	void activate()
	{
		for( int i =0; i < input.size.x * input.size.y * input.size.z; i++)
		{
			bool active = (rand() % RAND_MAX)/float(RAND_MAX) <=p_val;
			hitmap.data[i] = active;
			output.data[i] = active ? input.data[i] : 0.0f;
		}
	}
	void fix_weights()
	{

	}
	void calc_grads(tensor_t<float>& grad_next_layer)
	{
		for( int i = 0; i < input.size.x * input.size.y * input.size.z; i++)
			grads_in.data[i] = hitmap.data[i] ? grad_next_layer.data[i] : 0.0f;
	}
};
#pragma pack(pop)