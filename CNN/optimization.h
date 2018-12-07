#pragma once
#include "gradient_t.h"
#define LEARNING_RATE 0.01
#define MOMENTUM 0.9
#define WEIGHT_DECAY 0.001

// the update weight function will take in a weight and update it, or with a multiple.
static float update_weight( float w, gradient_t& grad, float multp =1)
{
	//essentially gradient descent with momentum, 
	// If we have time, change it to the nesterov accelerated graidient.
	float m = (grad.gradient + grad.oldGradient * MOMENTUM);
	w -= LEARNING_RATE * m * multp + LEARNING_RATE * WEIGHT_DECAY * w;

	return w;
}
// after updating weight, we gotta update the gradient.
static void update_gradient(graident_t&grad)
{
	grad.oldGradient = (grad.gradient + gradient.oldGradient * MOMENTUM);
}

