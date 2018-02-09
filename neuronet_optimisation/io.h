#pragma once;

#include <stdio.h>
#include <math.h>
#include <vector>
#include <string>



using namespace std;

void readTrajectory(string filename, vector<vector<float>>& trajectory, int tlen);

void writeTrajectory(string filename, vector<vector<float>>& trajectory, float timestep);

void writeNNStructure(string filename, int nneurons, int nvisneurons, float* values, float* weights); 
