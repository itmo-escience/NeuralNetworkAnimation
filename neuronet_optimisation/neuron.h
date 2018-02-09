#include <math.h>
#include <limits>

#include <vector>

#pragma once

using namespace std;

#define T_LEN	  5000
#define DV		1.0e-3
#define T_STEP	  0.01

inline double randomreal();

inline float uniformRand(float l, float r);

inline float rsmd(float* fit, const float* sample, int len);
inline void step_calc(float* fit_step, float p_smd, float n_smd);
void rand_init(int n, float* arr, float l, float r);
inline void norm(float len, float* grad, int n);
int min_index(float* err, int n, float* min_err=0);
int max_index(float* err, int n);

float const inf_p =  1/0.3e-55;
float const inf_n = -1/0.3e-55;

class IdGenerator
{
public:
    IdGenerator(int len = 100);
    int getId();
    void returnId(int Id);
private:
    vector <int> refundId;
             int lastId;
};

class NeuroNet
{
public:
	NeuroNet(int nneurons, int nIter, int n_vis_dof,
				  float* init_value, 
				  float* init_weight);
	~NeuroNet();

void	setTLength(int TL);

void	addNeuron(float init_val=0, float* init_weight=0);
void	delNeuron();
void	calculate(float dt);
void	clear();
void	alterWeight(const float* weight);
void	alterValue(const float* value);

void	alterWeight(int id, float weight);
void	alterValue(int id, float value);

void	setWeight(const float* weight);
void	setValue(const float* value);

float	gradRelease(vector <bool> arch, const float* sample, int s_len, int max_iter);
float	gradRelease(vector<vector<float>>& sample, int s_len, int max_iter);

void	getWeight(float* weight);
void	getNeurons(float* value_arr);
void	getInitValues(float* value_arr);
void	getTrajectory(vector<vector<float>>& out_trajectory);

private:
IdGenerator*	id_gen;
		 int	cur_nnum;
		 int	cur_wnum;

		 int	iter_count;
vector <vector<float>>	trajectory;

vector <float>	init_value;
vector <float>	value;
vector <float>	value_predictor;
vector <float>	weight;
vector <float>	grad;

int trajectory_length;
};