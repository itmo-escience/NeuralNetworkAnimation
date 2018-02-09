#include "neuron.h"
#include <math.h>

using namespace std;

inline double randomreal()
{
    int i1 = rand();
    int i2 = rand();
    while(i1==RAND_MAX)
        i1 =rand();
    while(i2==RAND_MAX)
        i2 =rand();
    double mx = RAND_MAX;
    return (i1+i2/mx)/mx;
}


int min_index(float* err, int n, float* min_err)
{
	float min=100;
	int min_ind =0 ;
	for(int i=0; i<n; i++)
	{
		if(abs(err[i])<min&&err[i]!=0)
		{
			min = abs(err[i]);
			min_ind = i;
		}
	};
	if(min_err!=0)
		*min_err = min;
	return min_ind;
};

int max_index(float* err, int n)
{
	float max=0;
	int max_ind =0 ;
	for(int i=0; i<n; i++)
	{
		if(abs(err[i])>max)
		{
			max = abs(err[i]);
			max_ind = i;
		}
	};
	
	return max_ind;
};

inline void norm(float len, float* grad, int n)
{
	float l = 0;
	for(int i=0; i<n; i++)
		l += grad[i]*grad[i];
	l = sqrt(l);

	for(int i=0; i<n; i++)
		grad[i] = grad[i]/l*len;
};

inline bool isfinite(float value)
{
  return value !=  std::numeric_limits<float>::infinity() &&
         value != -std::numeric_limits<float>::infinity();
}

inline float uniformRand(float l, float r)
{
	{
		float a=randomreal();
		float x=a*r+l;		
		return x;
	}
};

inline float rsmd(float* fit, const float* sample, int len)
{
	float smd=0;
	for(int i=0; i<len; ++i)
		smd+=pow((fit[i]-sample[i]),2);

	smd = sqrt(smd)/len;
	return smd;
};

inline void step_calc(float* fit_step, float p_smd, float n_smd)
{
	if(abs(p_smd-n_smd)<p_smd*0.01&&*fit_step<0.1)
		*fit_step = *fit_step*1.1;
	else
		*fit_step = *fit_step/1.2;
}


void rand_init(int n, float* arr, float l, float r)
{
	for(int i=0; i<n; i++)
		arr[i] = uniformRand(l,r);
}

IdGenerator::IdGenerator(int len)
{lastId=0; refundId.reserve(len);}

void IdGenerator::returnId(int id)
{refundId.push_back(id);}

int IdGenerator::getId()
{
	if(refundId.size()==0)
	{	lastId++;
		return lastId;}
	else
	{	int r=refundId.back();
		refundId.pop_back();;
		return r;}
};

NeuroNet::NeuroNet(int nneurons, int nIter, int n_vis_dof, float *i_value, float *i_weight):
	cur_wnum(nneurons*nneurons), cur_nnum(nneurons), iter_count(0)
{
	id_gen = new IdGenerator();
	weight.resize(cur_wnum,0);
	value.resize(cur_nnum,0);
	value_predictor.resize(cur_nnum,0);
	init_value.resize(cur_nnum,0);
	grad.resize(cur_nnum+cur_wnum,0);
	trajectory.resize(n_vis_dof);
	for(int i=0; i<n_vis_dof; i++)
		trajectory[i].reserve(nIter);  // Âèäèìû âñåãäà ïåðâûå N íåéðîíîâ

	memcpy(&(value[0]), i_value, cur_nnum*sizeof(float));
	init_value = value;
	memcpy(&(weight[0]), i_weight, cur_wnum*sizeof(float));

	trajectory_length = (int)1.0/T_STEP;
	setTLength(trajectory_length);
};

NeuroNet::~NeuroNet()
{
	trajectory.clear();
	value.clear();
	value_predictor.clear();
	weight.clear();
	grad.clear();
	delete id_gen;
};

void NeuroNet::setTLength(int TL)
{
	trajectory_length = TL;

	for(int i=0; i<trajectory.size(); i++)
		trajectory[i].resize(TL, 0);
};

void NeuroNet::alterWeight(const float *a_weight)
{
	for(int i=0; i<cur_wnum; i++)
		weight[i]+=a_weight[i];
};

void NeuroNet::alterValue(const float *a_value)
{
	for(int i=0; i<cur_nnum; i++)
		value[i]+=a_value[i];
	init_value = value;
};

void NeuroNet::alterWeight(int id, float w)
{
	weight[id]+=w;
	calculate(T_STEP);
	weight[id]-=w;
	value = init_value;
};

void NeuroNet::alterValue(int id, float v)
{
	value[id]+= v;
	calculate(T_STEP);
	value = init_value;
};

void NeuroNet::setWeight(const float* n_weight)
{memcpy(&(weight[0]), n_weight, cur_wnum*sizeof(float));};

void NeuroNet::setValue(const float* n_value)
{
	memcpy(&(value[0]), n_value, cur_nnum*sizeof(float));
	memcpy(&(init_value[0]), n_value, cur_nnum*sizeof(float));
};

void NeuroNet::clear()
{
	memcpy(&value[0],&init_value[0], cur_nnum*sizeof(float));
	iter_count=0;
};

/*
void NeuroNet::calculate(float dt)
{
//	if(trajectory.size()<=niter)
	//	trajectory.resize(niter,0);
	
	this->clear();

	for(int I=0; I<trajectory_length; I++)
	{
		for(int i=0; i<trajectory.size(); i++)
			trajectory[i][I]=value[i]; // ¬идимы всегда первые N нейронов

		memcpy(&value_predictor[0], &value[0], sizeof(float)*cur_nnum);
		for(int i=0; i<cur_wnum; i++)
			value_predictor[i/cur_nnum] +=weight[i]*value[i%cur_nnum]*dt;

		for(int i=0; i<cur_wnum; i++)
			value[i/cur_nnum] +=weight[i]*(value[i%cur_nnum]+value_predictor[i%cur_nnum])*dt*0.5;
	}
	iter_count+=trajectory_length;
};*/


void NeuroNet::calculate(float dt)
{
//	if(trajectory.size()<=niter)
	//	trajectory.resize(niter,0);
	
	this->clear();

	for(int I=0; I<trajectory_length; I++)
	{
		for(int i=0; i<trajectory.size(); i++)
			trajectory[i][I]=value[i];

		memcpy(&value_predictor[0], &value[0], sizeof(float)*cur_nnum);

		float signal = 0;
		float signal_1 = 0;

		for(int i=0; i<cur_wnum; i++)
		{
			signal+=weight[i]*value[i%cur_nnum];
			if(i%cur_nnum == (cur_nnum-1))
			{
				value_predictor[i/cur_nnum] += atan(signal)*dt;
				signal = 0;
			}
		}

		for(int i=0; i<cur_wnum; i++)
		{
		signal+=weight[i]*value_predictor[i%cur_nnum];		
		signal_1+=weight[i]*value[i%cur_nnum];		
			if(i%cur_nnum == (cur_nnum-1))
			{
				value[i/cur_nnum] += atan((signal_1 + signal)*0.5f)*dt;				
				signal = 0;
				signal_1 = 0;
			}
		}
	}
	iter_count+=trajectory_length;
};

void NeuroNet::getWeight(float *weight_arr)
{
	for(int i=0; i<cur_wnum; i++)
			weight_arr[i]=weight[i];
};

void NeuroNet::getNeurons(float *value_arr)
{
	for(int i=0; i<cur_nnum; i++)
			value_arr[i]=value[i];
};

void NeuroNet::getInitValues(float *value_arr)
{
	for(int i=0; i<cur_nnum; i++)
		value_arr[i]=init_value[i];
};

void NeuroNet::getTrajectory(vector<vector<float>>& out_trajectory)
{
	out_trajectory = trajectory;
}

void NeuroNet::addNeuron(float init_val, float* init_weight)
{
	cur_nnum+=1;
	cur_wnum = cur_nnum*cur_nnum;
	init_value.push_back(init_val);
	value.push_back(init_val);
	weight.resize(cur_wnum,0);
	grad.resize(cur_nnum+cur_wnum);
}

void NeuroNet::delNeuron()
{
	cur_nnum-=1;
	cur_wnum = cur_nnum*cur_nnum;
	init_value.pop_back();
	value.pop_back();
	weight.resize(cur_wnum,0);
	grad.resize(cur_nnum+cur_wnum);
};


float NeuroNet::gradRelease(vector<vector<float>>& sample, int s_len, int max_iter)
{
	float prob;
	int i=0;
	
	float err, n_err,step_const;
	float fit_step = 1000*DV;

	this->setTLength(s_len);
	
	this->calculate(T_STEP);
	this->clear();
	n_err = 0;
	for(int t=0; t<trajectory.size(); t++)
			n_err += rsmd(&(trajectory[t][0]),&sample[t][0],s_len);
	

	int m_i=0;

	grad.resize(cur_nnum+cur_wnum,0);
	err = n_err;	
	for(; m_i<max_iter; m_i++)
	{
		rand_init(cur_nnum+cur_wnum,&(grad[0]),-2*fit_step*(n_err),4*fit_step*(n_err)); // äîáàâèòü øóì äëÿ ëó÷øåé ñõîäèìîñòè
		// ëåâàÿ ãðàíèöà, äèàïàçîí (â 2 ðàçà áîëüøå)
				
		for(int d=0; d<trajectory.size(); d++)
		{
			init_value[d] = sample[d][0];
			value[d]	  = sample[d][0];
		}

		err = n_err;
		n_err = 0;

		i=trajectory.size();
		for(; i<cur_nnum; i++)
		{
			this->alterValue(i,DV);
			float ee = 0;
			for(int t=0; t<trajectory.size(); t++)			
				ee += pow(rsmd(&(trajectory[t][0]),&sample[t][0],s_len),2.0);				
			
			grad[i] += (err - ee)/DV;
		}

		for(; i<cur_wnum+cur_nnum; i++)
		{
			this->alterWeight(i-cur_nnum,DV);
			float ee = 0;
			for(int t=0; t<trajectory.size(); t++)			
				ee += pow(rsmd(&(trajectory[t][0]),&sample[t][0],s_len),2.0);				
			
			grad[i] += (err - ee)/DV;
		};

		norm(fit_step,&(grad[0]),grad.size());

		this->alterValue(&(grad[0]));
		this->alterWeight(&(grad[cur_nnum]));
		this->calculate(T_STEP);
		this->clear();

		for(int i=0; i<trajectory.size(); i++)
			n_err += pow(rsmd(&(trajectory[i][0]),&sample[i][0],s_len),2);

		printf("Error on %i iteration: %2.6f\n", m_i, n_err);

		if(n_err<1e-6)
			return n_err;
			
		if(false)
		{			
			prob = uniformRand(0,err);
			if((n_err-err)>prob)
			{
				for(int k=trajectory.size(); k<cur_nnum+cur_wnum; k++)
				grad[k]=-grad[k];
			this->alterValue(&(grad[0]));
			this->alterWeight(&(grad[cur_nnum]));
			this->calculate(T_STEP);
			this->clear();
			for(int t=0; t<trajectory.size(); t++)
				n_err += rsmd(&(trajectory[t][0]),&sample[t][0],s_len);
			fit_step = fit_step/2;
			}
		}

		if(n_err==inf_p||n_err==inf_n||_isnan(n_err))
			return 10e30;

		step_calc(&fit_step,err,n_err);
	}
	return n_err;
};

/*float NeuroNet::gradRelease(vector <bool> arch, const float* sample, 
							int s_len, int max_iter)
{
	float prob;
	int i=0;
	for(; i<cur_wnum; i++)
		weight[i]=arch[i]*weight[i];

	float err, n_err,step_const;
	float fit_step = 100*DV;

	if(trajectory.size()<=s_len)
		trajectory.resize(s_len,0);
	
	this->calculate(s_len);
	this->clear();
	n_err = rsmd(&(trajectory[0]),sample,s_len,3);

	int m_i=0;

	grad.resize(cur_nnum+cur_wnum,0);
	err = n_err;

	for(; m_i<max_iter; m_i++)
	{
		rand_init(cur_nnum+cur_wnum,&(grad[0]),-fit_step*(n_err),2*fit_step*(n_err));
				
		err = n_err;
		i=0;
		for(; i<cur_nnum; i++)
		{
			this->alterValue(i,DV);
			grad[i] += (err-rsmd(&(trajectory[0]),sample,s_len,3))/DV;
		}

		for(; i<cur_wnum+cur_nnum; i++)
		{
			if(arch[i-cur_nnum])
			{
				this->alterWeight(i-cur_nnum,DV);
				grad[i] += (err-rsmd(&(trajectory[0]),sample,s_len,3))/DV;
			}
			else
				grad[i]=0;
		};

		norm(fit_step,&(grad[0]),grad.size());

		this->alterValue(&(grad[0]));
		this->alterWeight(&(grad[cur_nnum]));
		this->calculate(T_LEN);
		this->clear();

		n_err = rsmd(&(trajectory[0]),sample,s_len,3);

		if(n_err>err)
		{			
			prob = uniformRand(0,err);
			if((n_err-err)>prob)
			{
			for(int k=0; k<cur_nnum+cur_wnum; k++)
				grad[k]=-grad[k];
			this->alterValue(&(grad[0]));
			this->alterWeight(&(grad[cur_nnum]));
			this->calculate(T_LEN);
			this->clear();
			n_err = rsmd(&(trajectory[0]),sample,s_len,3);
			fit_step = fit_step/2;
			}
		}

		if(n_err==inf_p||n_err==inf_n||_isnan(n_err))
			return 10e30;

		step_calc(&fit_step,err,n_err);
	}
	return n_err;
};*/