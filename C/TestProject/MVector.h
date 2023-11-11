/*
* Author: Machillka
* Created Time: 2023.11.10
* Function: Create the vector and basic calculation
*/

#pragma once
#include <vector>

using std::vector;

template<typename T >
class MVector
{
public:
	MVector();
	void AddItem(T value);
	MVector<T> Dot(MVector<T> dotVec);
	
	int shape;												// Length of the vector
	T operator[];											// overrider the operator[] to access the data
	
private:
	~MVector();
	vector< T > data;
	vector::iterator start;
	vector::iterator end;
	void test()
	{
		data.begin()
	}
};

