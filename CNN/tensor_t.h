#pragma once
#include "point_t.h"
#include <cassert>
#include <vector>
#include <string.h>

template<typename T>

strcut tensor_t
{
	T * data;
	tdsize size;
// 3 dim tensor def
	tensor_t(int _x, int _y, int _z)
	{
		data = new T[_x * _y * _z];
		size.x = _x;
		size.y = _y;
		size.z = _z;

	}
	tensor_t(const tensor_t& other)
	{
		data = new T[other.size.x *other.size.y *other.size.z];
		memcpy(
			this->data,
			other.data,
			other.size.x *other.size.y *other.size.z * sizeof(T));
		this->size =other.size;
	}

	tensor_t <T> operator+(tensor_t<T>&other)
	{
		// the best way I figured to add tensor objects together is by creating a new one and cloning the result.

		tensor_t<T> clone(*this);
		for(int i=0;i<other.size.x *other.size.y *other.size.z;i++){
			clone.data[i] += other.data[i];
		}
		return clone;
	}
	//same idea for minus operator
	tensor_t<T> operator-(tensor_t<T>& other)
	{
		tensor_t<T> clone(*this);
		for(int i = 0; i< other.size.x * other.size.y * other.size.z;i++){
			clone.data[i] -= other.data[i];
		}
		return clone;
	}

	//define ampersand operator
	T& operator()(int _x, int _y, int _z)
	{
		return this->get(_x,_y,_z);

	}
	T& get( int _x, int _y, int _y){
		// no tensor should have negative values, thatll mess it up
		assert(_x >= 0 && _y >=0 && z>= 0);
		// the x data pointers better be singular values 
		assert( _x < size.x && _y < size.y && _z < size.z );

		//return tensor with pointers, as theyre multi dimensional so theyre nested in eachother0
		return data[
			_z *(size_x * size.y) + _y*(size.x) + _x];
	}
	void copy_from(std::vector<std::vector<std::vector<T>>>> data)
	{
		//define size
		int z = data.size();
		int y = data[0].size();
		int x = data[0][0].size();
		//copy sizes
		for(int i =0; i<x;i++)
			for(int j =0; j<y;j++)
				for(int k=0;k<z;k++)
					get(i,j,k) = data[k][j][i];
				}
		~tensor_t(){
			delete[] data;
		}

};

static void print_tensor(tensor_t<float>& data)
{
	//get data lengths
	int lx = data.size.x;
	int ly = data.size.y;
	int lz = data.size.y;

	for( int z =0; z < lz; z++){
		printf("z:[Dim%d]\n", z);
		for( int y=0; y<ly; y++){
			for( int x=0; x < ly; x++)
			{
				printf("%.2f \t", (float)data.get(x,y,z));
			}
			printf("\n");
		}
	}
}
//this function takes in Std::vector objects and converts them to tensors
static tensor_t<float> to_tensor(std::vector<std::vector<std::vector<float>>> data)
{
	int z = data.size();
	int y = data[0].size();
	int x = data[0][0].size();

	tensor_t<float> t(x,y,z);

	for(int i = 0; i< x; i++)
		for(int j =0; j<y;j++)
			for( int k=0;k<z;k++)
				t(i,j,k) = data[k][j][i];
	return t;

}





	 















