#pragma once

// this is the gradient and allows to update and keep the old one.
struct gradient_t
{
	float gradient;
	float oldGradient;
	gradient_t(){
		gradient=0;
		oldGradient=0;
	}
};