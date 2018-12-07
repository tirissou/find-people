#pragma once 
// this way way we can access the class of every layer
enum class layer_type{
	convo2d,
	fc,
	relu,
	pool,
	dropout_layer
};