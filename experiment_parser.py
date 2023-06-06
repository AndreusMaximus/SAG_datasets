import matplotlib.pyplot as plt

import matplotlib.ticker as ticker

import numpy

import seaborn as sns


experiment1 = [10,20,30,40,50,60,70]
res_experiment1_por = []
res_experiment1_nml = []
res_ex1_total = []
scatter_X = []
scatter_Y = []
scatter_l = []
def read_experiment1(ex_location):
	for ex in experiment1:
		unsorted_por = []
		print(ex_location)
		with open(ex_location+"sag-por/"+str(ex)+"/results.csv") as por_ex1:
			results = por_ex1.read().splitlines();
			if results[0].find("file") != -1:
				del results[0]
			for result in results:
				unsorted_por.append(result.replace(" ","").split(","))
				
		
		unsorted_nml = []
		with open(ex_location+"sag/"+str(ex)+"/results.csv") as nml_ex1:
			results = nml_ex1.read().splitlines();
			if results[0].find("file") != -1:
				del results[0]
			for result in results:
				unsorted_nml.append(result.replace(" ","").split(","))
				
		res_experiment1_por.append(sorted(unsorted_por, key=lambda x: x[0]))
		res_experiment1_nml.append(sorted(unsorted_nml, key=lambda x: x[0]))
	
def percentage_swap(value):
	if value < 100:
		return 0 - (100-value)
	else:
		return 0 + value - 100			
			
def parse_experiment(nml, por, experiment):
	ex = 0
	runtimes_nml = []
	runtimes_por = []
	for ex in range(len(experiment)):
		nml_sched = 0
		nml_unsched = 0
		nml_states = [0,0,0,0]
		nml_cpu = [0,0,0,0]
		nml_mem = [0,0,0,0]
		nml_timeout = 0
		
		runtime_nml = [[],[]]
		runtime_por = [[],[]]
			
		por_sched = 0
		por_unsched = 0
		por_states = [0,0,0,0]
		por_cpu = [0,0,0,0]
		por_mem = [0,0,0,0]
		por_succ = [0,0]
		por_fail = [0,0]
		por_jpp = [0,0]
		por_timeout = 0
		t_unsched = 0
		t_sched = 0
		
		nml_ex_width = [0,0,0]
		por_ex_width = [0,0,0]
		t_jobs = 0
		m_jobset = 0;
		for i in range(len(nml[ex])):
			#check if the result is proper
			t_jobs += int(nml[ex][i][2]);
			m_jobset = max(m_jobset, int(nml[ex][i][2]))
			if nml[ex][i][1] == por[ex][i][1]:
				#correct result, both 1 or 0
				nml_states[0] += int(nml[ex][i][3])
				por_states[0] += int(por[ex][i][3])
				
				nml_cpu[0] += float(nml[ex][i][6])
				por_cpu[0] += float(por[ex][i][6])
				
				nml_mem[0] += float(nml[ex][i][7])
				por_mem[0] += float(por[ex][i][7])
				
				por_succ[0] += int(por[ex][i][10])
				por_fail[0] += int(por[ex][i][11])
				por_jpp[0] += int(por[ex][i][12])
				
				nml_ex_width[0] += int(nml[ex][i][5])
				por_ex_width[0] += int(por[ex][i][5])
				if int(nml[ex][i][1]) == 0:
					t_unsched += 1
					nml_unsched += 1
					por_unsched += 1
					nml_states[3] += int(nml[ex][i][3])
					por_states[3] += int(por[ex][i][3])
					
					nml_cpu[3] += float(nml[ex][i][6])
					por_cpu[3] += float(por[ex][i][6])
					
					nml_mem[3] += float(nml[ex][i][7])
					por_mem[3] += float(por[ex][i][7])
					
					nml_ex_width[1] += int(nml[ex][i][5])
					por_ex_width[1] += int(por[ex][i][5])
					
					runtime_nml[1].append(float(nml[ex][i][6]))
					runtime_por[1].append(float(por[ex][i][6]))
					
					scatter_Y.append(float(nml[ex][i][6]))
					scatter_X.append(float(nml[ex][i][2]))
					scatter_l.append("sag unschedulable")
					
					
					scatter_Y.append(float(por[ex][i][6]))
					scatter_X.append(float(por[ex][i][2]))
					scatter_l.append("sag-por unschedulable")
				else:
					t_sched += 1
					nml_sched += 1
					por_sched += 1
					nml_states[2] += int(nml[ex][i][3])
					por_states[2] += int(por[ex][i][3])
					
					nml_cpu[2] += float(nml[ex][i][6])
					por_cpu[2] += float(por[ex][i][6])
					
					nml_mem[2] += float(nml[ex][i][7])
					por_mem[2] += float(por[ex][i][7])
					
					nml_ex_width[2] += int(nml[ex][i][5])
					por_ex_width[2] += int(por[ex][i][5])
					
					runtime_nml[0].append(float(nml[ex][i][6]))
					runtime_por[0].append(float(por[ex][i][6]))
					
					scatter_Y.append(float(nml[ex][i][6]))
					scatter_X.append(float(nml[ex][i][2]))
					scatter_l.append("sag schedulable")
					
					
					scatter_Y.append(float(por[ex][i][6]))
					scatter_X.append(float(por[ex][i][2]))
					scatter_l.append("sag-por schedulable")
				
			elif nml[ex][i][1] > por[ex][i][1]:
				#normal says schedulable mine says no
				nml_states[1] += int(nml[ex][i][3])
				por_states[1] += int(por[ex][i][3])
				
				nml_cpu[1] += float(nml[ex][i][6])
				por_cpu[1] += float(por[ex][i][6])
				
				nml_mem[1] += float(nml[ex][i][7])
				por_mem[1] += float(por[ex][i][7])
				
				por_succ[1] += int(por[ex][i][10])
				por_fail[1] += int(por[ex][i][11])
				por_jpp[1] += int(por[ex][i][12])
				
				nml_sched += 1
				por_unsched += 1
				
				
				scatter_Y.append(float(nml[ex][i][6]))
				scatter_X.append(float(nml[ex][i][2]))
				scatter_l.append("sag schedulable")
				
				
				scatter_Y.append(float(por[ex][i][6]))
				scatter_X.append(float(por[ex][i][2]))
				scatter_l.append("sag-por unschedulable")
			else:
				print("FAILURE", experiment[ex])
				print(nml[ex][i])
				print(por[ex][i])
				nml_unsched += 1
				por_sched += 1
				
		experiment_result = []
		experiment_result.append(experiment[ex])
		experiment_result.append(percentage_swap((por_states[0]/max(nml_states[0],1))*100))
		experiment_result.append(percentage_swap((por_states[1]/max(nml_states[1],1))*100))
		
		experiment_result.append(percentage_swap((por_cpu[0]/max(nml_cpu[0],1))*100))
		experiment_result.append(percentage_swap((por_cpu[1]/max(nml_cpu[1],1))*100))
		
		experiment_result.append(percentage_swap((por_mem[0]/max(nml_mem[0],1))*100))
		experiment_result.append(percentage_swap((por_mem[1]/max(nml_mem[1],1))*100))
		
		experiment_result.append([(nml_sched/len(nml[ex]))*100, (por_sched/len(nml[ex]))*100])
		experiment_result.append([por_succ[0], por_fail[0]])
		experiment_result.append(por_jpp[0]/max(1,por_succ[0]))
		
		
		experiment_result.append([percentage_swap((por_states[2]/max(1,nml_states[2]))*100), percentage_swap((por_states[3]/max(1,nml_states[3]))*100)])
		experiment_result.append([percentage_swap((por_cpu[2]/max(1,nml_cpu[2]))*100), percentage_swap((por_cpu[3]/max(1,nml_cpu[3]))*100)])
		experiment_result.append([percentage_swap((por_mem[2]/max(1,nml_mem[2]))*100), percentage_swap((por_mem[3]/max(1,nml_mem[3]))*100)])
		
		experiment_result.append([percentage_swap((nml_states[2]/max(1,t_sched))*100), percentage_swap((por_states[2]/max(1,t_sched))*100)])
		experiment_result.append([percentage_swap((nml_states[3]/max(1,t_unsched))*100), percentage_swap((por_states[3]/max(1,t_unsched))*100)])
		
		experiment_result.append([nml_ex_width[0]/max(1,(t_sched+t_unsched)), nml_ex_width[1]/max(1,t_unsched), nml_ex_width[2]/max(1,t_sched), por_ex_width[0]/max(1,(t_sched+t_unsched)), por_ex_width[1]/max(1,t_unsched), por_ex_width[2]/max(1,t_sched)])
		
		experiment_result.append([nml_states[0], por_states[0]])
		res_ex1_total.append(experiment_result)
		
		runtimes_nml.append(runtime_nml)
		runtimes_por.append(runtime_por)
	print("utilization,","mean O total,","std dev O total,","mean POR total,","std dev POR total,","mean O sched,","std dev O sched,","mean O unsched,","std dev O unsched,","mean POR sched,","std dev POR sched,","mean POR unsched,","std dev POR unsched")
	for i in range(len(experiment)):
		s = str(experiment[i]) + ","
		
		if len(runtimes_nml[i][0]) + len(runtimes_nml[i][1]) > 0:
			s += str(numpy.mean(runtimes_nml[i][0]+runtimes_nml[i][1])) + ", "
			s += str(numpy.std(runtimes_nml[i][0]+runtimes_nml[i][1])) + ", "
		else:
			s += "x, "
			s += "x, "
		
		if len(runtimes_por[i][0]) + len(runtimes_por[i][1]) > 0:
			s += str(numpy.mean(runtimes_por[i][0]+runtimes_por[i][1])) + ", "
			s += str(numpy.std(runtimes_por[i][0]+runtimes_por[i][1])) + ", "
		else:
			s += "x, "
			s += "x, "
		
		if len(runtimes_nml[i][0]) > 0:
			s += str(numpy.mean(runtimes_nml[i][0])) + ", "
			s += str(numpy.std(runtimes_nml[i][0])) + ", "
		else:
			s += "x, "
			s += "x, "
			
		if len(runtimes_nml[i][1]) > 0:
			s += str(numpy.mean(runtimes_nml[i][1])) + ", "
			s += str(numpy.std(runtimes_nml[i][1])) + ", "
		else:
			s += "x, "
			s += "x, "
		
		if len(runtimes_por[i][0]) > 0:
			s += str(numpy.mean(runtimes_por[i][0])) + ", "
			s += str(numpy.std(runtimes_por[i][0])) + ", "
		else:
			s += "x, "
			s += "x, "
			
		if len(runtimes_por[i][1]) > 0:
			s += str(numpy.mean(runtimes_por[i][1])) + ", "
			s += str(numpy.std(runtimes_por[i][1])) + ""
		else:
			s += "x, "
			s += "x"
			
		print(s)
		
		

def format_y_tick(value, pos):
    if value > 0:
        return f"+{value}"
    else:
        return str(value)	
		
def plot_result(name, result, x_label, y_values, y_label, y_ticks, title,legend=False,labels=[],prefix=False, ex_location=""):
	# Extract the first and second values into separate lists
	x_values = [t[0] for t in result]
	plt.figure(figsize=(16, 12))
	# Plot the second values against the first values
	plt.plot(x_values, y_values)
	# Set labels for the x and y axes
	if legend == True:
		plt.legend(labels)
	if len(y_ticks) != 1:
		plt.yticks = y_ticks
	
	if prefix == True:
		# Apply the custom formatter to the y-axis ticks
		formatter = ticker.FuncFormatter(format_y_tick)
		plt.gca().yaxis.set_major_formatter(formatter)
	
	plt.title(title)
	plt.xlabel(x_label)
	plt.ylabel(y_label)

	# Display the plot
	#plt.show()
	plt.savefig(ex_location+name+".jpg");
	plt.close();
	
def plot_experiment(experiment, ex_location):
	plot_result("reduced states total",experiment,"utilization",[x[1] for x in experiment],"percentage", range(0,101,10),"difference in # states w.r.t. original MC SAG",prefix=True, ex_location=ex_location)
	plot_result("runtime reduction total",experiment,"utilization",[x[3] for x in experiment],"percentage", range(0,101,10),"runtime difference w.r.t. original MC SAG",prefix=True, ex_location=ex_location)
	#plot_result("",experiment,"utilization",[x[5] for x in experiment],"percentage", range(0,101,10),"memory difference w.r.t. original MC SAG",prefix=True, ex_location=ex_location)
	plot_result("schedulable",experiment,"utilization",[x[7] for x in experiment],"percentage", range(0,101,10),"schedulable result", True, ["original", "POR"],prefix=False, ex_location=ex_location)
	plot_result("nr reductions",experiment,"utilization",[x[8] for x in experiment],"# reductions", range(0,101,10),"Total number of reductions", True, ["successfull", "failed"], ex_location=ex_location)
	plot_result("avg job per por",experiment,"utilization",[x[9] for x in experiment],"jobs", range(0,101,10),"average # jobs per reduction", ex_location=ex_location)
	
	#plot_result("remaining states correct/wrong",experiment,"utilization",[[x[1], x[2]] for x in experiment],"percentage", range(0,101,10),"remaining states w.r.t. original MC SAG", True, ["correct", "wrong"], ex_location=ex_location)
	#plot_result("",experiment,"utilization",[[x[3], x[4]] for x in experiment],"percentage", range(0,101,10),"runtime difference w.r.t. original MC SAG", True, ["correct", "wrong"], ex_location=ex_location)
	#plot_result("",experiment,"utilization",[[x[5], x[6]] for x in experiment],"percentage", range(0,101,10),"memory difference w.r.t. original MC SAG", True, ["correct", "wrong"], ex_location=ex_location)
	
	
	plot_result("state reduction diff",experiment,"utilization",[x[10] for x in experiment],"percentage", range(0,101,10),"state reduction ", True, ["schedulable", "unschedulable"],prefix=True, ex_location=ex_location)
	plot_result("runtime diff",experiment,"utilization",[x[11] for x in experiment],"percentage", range(0,101,10),"runtime difference ", True, ["schedulable", "unschedulable"],prefix=True, ex_location=ex_location)
	#plot_result("",experiment,"utilization",[x[12] for x in experiment],"percentage", range(0,101,10),"memory useage ", True, ["schedulable", "unschedulable"], ex_location=ex_location)
	
	plot_result("avg states for sched",experiment,"utilization",[x[13] for x in experiment],"Number of states", [0],"Average # states for correct schedulable sets", True, ["original", "MC_POR"], ex_location=ex_location)
	
	plot_result("avg states for unsched",experiment,"utilization",[x[14] for x in experiment],"Number of states", [0],"Average # states for correct unschedulable sets", True, ["original", "MC_POR"], ex_location=ex_location)
	
	plot_result("avg exploration widht",experiment,"utilization",[x[15] for x in experiment],"average exploration with", [0],"Average # states for correct unschedulable sets", True, ["O avg", "O unsched", "O sched" , "POR avg", "POR unsched","POR sched"], ex_location=ex_location)
	
	plot_result("total states",experiment,"utilization",[x[16] for x in experiment],"Number of states", [0],"Total number of states", True, ["original", "POR"], ex_location=ex_location)
	
def scatter_plot(ex_location):

	# Create scatter plot with different colors for each type
	sns.scatterplot(x=scatter_X, y=scatter_Y, hue=scatter_l)

	# Set labels for the x and y axes
	plt.xlabel('# of jobs')
	plt.ylabel('time')

	# Get the current legend
	legend = plt.legend()


	# Display the scatter plot
	plt.show()
	

def main(args):
	ex_location = "results-utilization-Auto-comb"+"/"
	read_experiment1(ex_location)
	parse_experiment(res_experiment1_nml, res_experiment1_por, experiment1)
	plot_experiment(res_ex1_total, ex_location)
	scatter_plot(ex_location)
	
	print([1,2,3,]+[4,5,6])
	return 0

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
