#pragma once
#include "types.h"
#include "tensor_t.h"

#pragma pack(push,1)

//This is a layer structure that can take in gradients and input tensors and output tesnors.
// This can be adapted for all layers.
struct layer_t
{
	layer_type type;
	tensor_t<float> grads_in;
	tensor_t<float> in;
	tensor_t<float> out;

};
#pragma pack(pop)