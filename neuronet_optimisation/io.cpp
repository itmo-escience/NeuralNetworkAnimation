#pragma once;
#include <sstream>
#include <stdio.h>
#include "io.h"

#define COL_LEN 16;

void readTrajectory(string filename, vector<vector<float>>& trajectory, int tlen)
{
	FILE* f = fopen(filename.c_str(), "r");

	int num_of_cols = 0;

	fscanf(f, "%d\n", &num_of_cols);
	num_of_cols++;

	int max_string_len = num_of_cols * COL_LEN;
	
	char* buffer = new char[max_string_len];

	vector<vector<float>> read_trajectory;

	while(!feof(f))
	{
		fgets(buffer, max_string_len, f);

		string buf;
		stringstream ss(buffer);

		vector<string> tokens;
		vector<float> timestep;
		
		while (ss >> buf)
			timestep.push_back(stof(buf.c_str(), 0));
		read_trajectory.push_back(timestep);

		if(read_trajectory.size()>=tlen)
			break;
	}

	fclose(f);
	
	for(int i=1; i<read_trajectory[0].size(); i++)
	{
		vector<float> single_t;
		single_t.resize(read_trajectory.size(), 0.0f);
		
		for(int j=0; j<single_t.size(); j++)
			single_t[j] = read_trajectory[j][i];
		trajectory.push_back(single_t);
	}

	delete [] buffer;
}

void writeTrajectory(string filename, vector<vector<float>>& trajectory, float timestep)
{
	float time=0;
	FILE* f = fopen(filename.c_str(), "w");

	for(int i=0; i<trajectory[0].size(); i++)
	{
		string buf = "";
		stringstream ss;
		ss << std::to_string(time) << "\t";
		for(int j=0; j<trajectory.size(); j++)
			ss << std::to_string(trajectory[j][i]) << "\t";
		ss << "\n";
		buf = ss.str();
		fputs(buf.c_str(), f);
		time+=timestep;
	}

	fclose(f);
}

void writeNNStructure(string filename, int nneurons, int nvisneurons, float* values, float* weights)
{	
	FILE* f = fopen(filename.c_str(), "w");
	fprintf(f, "Initial values:\n");
	
	string buf = "";
	stringstream ss;
	
	for(int j=0; j<nneurons; j++)
	{
		if(j<nvisneurons)
			ss << "*";	

		ss << std::to_string(values[j]) << " ";	
	}

	ss << "\n";
	buf = ss.str();
	fputs(buf.c_str(), f);

	fprintf(f, "* - visible neuron.\n\n");

	fprintf(f, "Weights:\n");

	for(int i=0; i<nneurons; i++)
	{
		for(int j=0; j<nneurons; j++)
		{
			fprintf(f, "%f ", weights[i*nneurons + j]);
		}
		fprintf(f, "\n");
	}

	fclose(f);
}