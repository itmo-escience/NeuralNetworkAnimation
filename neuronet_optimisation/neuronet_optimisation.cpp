// neuronet_optimisation.cpp : Defines the entry point for the console application.
//
#include <iostream>

#include "neuron.h"
#include "io.h"

using namespace std;


#define SMD_STEP 0.01
#define MAX_ITER 1000
#define PI 3.141592

#define NUM_VIS_DOF  2
#define NUM_OPT_ITER 500

void rand_koshi_distr(int n, const float* in_arr, float* out_arr, float gamma)
{
	float x, prob;
	float bond = gamma*6;
	for(int i=0; i<n; i++)
	{
		do
		{
			x = uniformRand(-bond+in_arr[i],bond*2);
			prob = uniformRand(0,1);
		}
		while(prob>gamma/(pow(x-in_arr[i],2)+gamma*gamma)/PI);
			out_arr[i] = x;
	}
};

float my_measure(float* fit, float* sample, int len, int factor = 1)
{
	float temp=(fit[0]-sample[0]);
	float count=0;
	for(int i=1; i<len/factor; ++i)
	{
		count+=abs(temp-(fit[i*factor]-sample[i*factor]));
		temp=(fit[i*factor]-sample[i*factor]);
	};

	return count;
};

struct OptHistory
{
	void calc_mask()
	{
		mask.resize(nnum*nnum,0);
		for(int j=0; j<nnum; j++)
				mask[j]=1;

		int size = architecture.size();
		if(!len.empty())
		for(int i=0; i<len.back(); i++)
		{
			if(architecture[size-i-1])
			for(int j=0; j<nnum; j++)
				mask[((size-i-1)%nnum)*nnum + j]=1;
		}
	}
	vector <float>	value;
	vector <bool>	architecture;
	vector <int>	len;
	vector <bool>	mask;
	int nnum;
};

void sin_generator(vector<float>& trajectory, int len, float mult=1.0f, float shift=0.0f)
{
	float g = 10.0;
	float L = 1.0 ;

	float omega = sqrt(g/L);

	trajectory.resize(len, 0);
	for(int i=0; i<len; i++)
		trajectory[i] = cos(i*T_STEP*omega);

	//for(int i=0; i<len; i++)
		//trajectory[i] = sin(2*i*T_STEP) + cos(2*i*T_STEP*sqrt(3));
}

int main(int argc, char argv[]) // Простая подборка весов
{
	FILE* output;	
	float* fit	  = new float[3*T_LEN];	
	
	float final_error = 0;

	vector<vector<float>> ref_trajectory;
	vector<vector<float>> fit_trajectory;


	readTrajectory("trajectory.txt", ref_trajectory, T_LEN);

	ref_trajectory.resize(NUM_VIS_DOF);
	//sin_generator(ref_trajectory[0], T_LEN, 1.0, 0.0);
	//sin_generator(ref_trajectory[1], T_LEN, sqrt(3.0), 1.0);


	//

	int fit_N = 10; // Число нейронов
	float*		value  = new float[fit_N];
	float* init_value  = new float[fit_N];
	float* weight = new float[fit_N*fit_N];

	rand_init(fit_N		 ,value ,-0.1, 0.2);		// Случайная инициализация весов
	rand_init(fit_N*fit_N,weight,-0.1, 0.2); // Случайная инициализация весов

	NeuroNet* fit_net	 = new NeuroNet(fit_N, T_LEN, NUM_VIS_DOF, value, weight); // создали нейронную сеть

	fit_net->setTLength(T_LEN);
		
	final_error = fit_net->gradRelease(ref_trajectory, T_LEN, NUM_OPT_ITER);

	fit_net->clear();
	fit_net->calculate(T_STEP);
	fit_net->getTrajectory(fit_trajectory);
	fit_net->getWeight(weight);
	fit_net->getNeurons(value);
	fit_net->getInitValues(init_value);

	writeTrajectory ("out_trajectory.txt" , fit_trajectory, T_STEP);
	writeTrajectory ("ref_trajectory.txt" , ref_trajectory, T_STEP);
	writeNNStructure("out_NNstructure_init.txt",fit_N, fit_trajectory.size(), init_value, weight);	
	writeNNStructure("out_NNstructure_final.txt",fit_N, fit_trajectory.size(), value, weight);	

	delete [] fit;
}

/*
int main(int argc, char argv[]) // Наращивание сети
{
	FILE* output;

	float* sample = new float[3*T_LEN];
	float* fit = new float[3*T_LEN];

	for(int i=0; i<3*T_LEN; i++)
		sample[i] = sin(2*i*T_STEP);

	vector<vector<float>> trajectory;

	readTrajectory("trajectory.txt", trajectory);
		
	//int fit_N = 2;
	int fit_N = 8; // Число нейронов
	int max_fit_N = 5;
	int max_num_W = 5;

	float* value  = new float[max_fit_N];
	float* weight = new float[max_fit_N*max_fit_N];
	
	rand_init(fit_N,value,-0.1,0.2);
	rand_init(fit_N*fit_N,weight,-0.1,0.2);

	NeuroNet* fit_net	 = new NeuroNet(fit_N, T_LEN, value,weight);

	float* error = new float[10*10];
	memset(error,0,max_fit_N*max_fit_N*sizeof(float));
	vector <bool> architecture;
	vector <bool> temp_arch;
	architecture.resize(fit_N*fit_N,0);
	temp_arch.resize(fit_N*fit_N,0);
	OptHistory opt_history;
	opt_history.nnum=fit_N;	
	float min_err=0;
	int index=0;

	for(int i=0; i<max_num_W; i++)
	{		
		opt_history.calc_mask();
		for(int j=0; j<fit_N*fit_N; j++)
		{
			error[j]=0;
			if(architecture[j]!=1&&opt_history.mask[j])
			{
			architecture[j]=1;
			error[j]=fit_net->gradRelease(architecture,sample,T_LEN, 10000);
			printf("(%i%i) %f\n",j/fit_N,j%fit_N, error[j]);
			rand_init(fit_N,value,-0.01,0.02);
			rand_init(fit_N*fit_N,weight,-0.01*T_STEP,0.02*T_STEP);
			fit_net->setValue(value);
			fit_net->setWeight(weight);
			architecture[j]=0;
			}
		}
		printf("\n");
		printf("//////////////\n");
		index = min_index(error, fit_N*fit_N,&min_err);
		printf("%f\n", min_err);
		architecture[index]=1;
		min_err=log(1+min_err)*(i+1);
		for(int h=0; h<architecture.size(); h++)
		{
			printf("%i ",(int)architecture[h]);
			if((h+1)%fit_N==0)
				printf("\n");
		}		
		printf("//////////////\n");
		printf("\n");

		opt_history.value.push_back(min_err);
		opt_history.len.push_back(architecture.size());
		opt_history.architecture.insert(opt_history.architecture.end(),
								architecture.begin(),
								architecture.end());
		memset(error,0,max_fit_N*max_fit_N*sizeof(float));
		
		if((index+1)%fit_N==0)
		{	
			temp_arch = architecture;

			architecture.erase(architecture.begin(),architecture.end());
			architecture.resize((fit_N+1)*(fit_N+1),0);
			int b = 0;
						
			for(int a=0; b<fit_N*fit_N; a++)
			{
				if((a+1)%(fit_N+1)==0)
					architecture[a]=0;
				else
				{
					architecture[a]=temp_arch[b];
					b++;
				}
			}
			
			fit_N++;		
			fit_net->addNeuron();
			opt_history.nnum++;
		}
	}
	float	final_err;
	index = min_index(&(opt_history.value[0]),max_num_W,&final_err);
	int count=0;
	for(int ind=0; ind<index; ind++)
		count+=opt_history.len[ind];

	vector <bool>	final_arch;
	for(int ind=0; ind<opt_history.len[index]; ind++)
		final_arch.push_back(opt_history.architecture[count+ind]);

	rand_init(fit_N,value,-0.1,0.2);
	rand_init(fit_N*fit_N,weight,-0.1*T_STEP,0.2*T_STEP);
	fit_net = new NeuroNet(sqrt((float)final_arch.size()),T_LEN,value,weight);

	final_err = fit_net->gradRelease(final_arch,sample,T_LEN, 100000);
	for(int f=0; f<final_arch.size(); f++)
	{
		printf("%i ",(int)final_arch[f]);
			if((f+1)%(int)sqrt((float)final_arch.size())==0)
				printf("\n");
	}

	fit_N = (int)sqrt((float)final_arch.size());

	output = fopen("output.txt","w");

	fit_net->clear();
	fit_net->getNeurons(value);
	fit_net->getWeight(weight);

	fprintf(output, "NEURONET\t %i:\n", fit_N);
	fprintf(output, "\n");
	
	for(int i=0; i<fit_N; i++)
		fprintf(output, "\t %.2f", value[i]);
	fprintf(output, "\n");

	fprintf(output,"\n");
	
	for(int i=0; i<fit_N*fit_N; i++)
	{
		if(i%fit_N==0)
			fprintf(output,"\n");
		fprintf(output,"\t%.2f",weight[i]);
		
	}
	fprintf(output,"\n");
	
	fit_net->calculate(3*T_LEN, T_STEP);
	fit_net->getTrajectory(fit);

	fprintf(output,"\nFITTING: ID# 0\n\n");
	{
		fprintf(output,"sample \t fit \n");
		for(int i=0; i<T_LEN; i++)
			fprintf(output,"%f \t %f \n",sample[i],fit[i]);

	fprintf(output,"_______________________________________________\n");
	fprintf(output,"sample \t fit \n");
		for(int i=T_LEN; i<3*T_LEN; i++)
			fprintf(output,"%f \t %f \n",sample[i],fit[i]);

			fprintf(output,"\nerr=%f\n",final_err);
	}

	fclose(output);

	return 0;
}

*/