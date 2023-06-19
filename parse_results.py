import matplotlib.pyplot as plt

import matplotlib.ticker as ticker

import numpy

import seaborn as sns

def correctness_check(set_index, test_index, comp_index, ex):
	if int(full_raw[comp_index][ex][set_index][1]) < int(full_raw[test_index][ex][set_index][1]):
		print(full_raw[comp_index][ex][set_index][0])
	

experiment = []
test_names = []
comp_names = []
scatter = []	
parsed_results = []
full_raw = []

def set_config(ex_path):
	global experiment
	global test_names
	global scatter
	global parsed_results
	global full_raw
	experiment = []
	test_names = []
	scatter = []
	parsed_results = []
	full_raw = []
	with open(ex_path+"/config.txt") as c:
		lines = c.read().splitlines()
		for ex in lines[0].replace(" ","").split(","):
			experiment.append(int(ex))
			scatter.append([])
		for t in lines[1:]:
			test_names.append(t.split("|")[0])
			comp_names.append(t.split("|")[1])
		

def parse_result(ex_path,test_name):
	tmp_result = []
	tmp_result.append(test_name);
	for ex in range(len(experiment)):
		tmp_test = []
		with open(f"{ex_path}/{test_name}/{str(experiment[ex])}/results.csv") as res_ex:
			results = res_ex.read().splitlines();
			#if the headers are still attached
			if results[0].find("file") != -1:
				del results[0]	
			for result in results:
				tmp_test.append(result.replace(" ","").split(","))
		tmp_result.append(sorted(tmp_test, key=lambda x: x[0]))
	full_raw.append(tmp_result)
		
def parse_raw(test_name,comp_name):
	raw = 0
	for r in range(len(full_raw)):
		if full_raw[r][0] == test_name:
			raw = r
	comp = -1
	if comp_name != "":
		for c in range(len(full_raw)):
			if full_raw[c][0] == comp_name:
				comp = c
		
	tmp_results = []
	tmp_results.append(test_name)
	std_dev = [test_name + " stdev"]
	mean = [test_name + " mean"]
	
	states = []
	width = []
	runtimes = []
	mem = []
	reductions = []
	sched = []
	for e in range(len(experiment)):
		ex = e+1
		tmp_result = []
		#read the full raws
		t_sched = 0
		
		t_states = [0,0]
		t_width = [0,0]
		t_time = [0,0]
		t_mem = [0,0]
		
		t_s_red = 0
		t_f_red = 0
		t_jpred = 0
		
		times = [[],[]]
		for r in range(len(full_raw[raw][ex])):
			#check the runtime first
			t_sched += int(full_raw[raw][ex][r][1])
			t_states[int(full_raw[raw][ex][r][1])] += int(full_raw[raw][ex][r][3])
			t_width[int(full_raw[raw][ex][r][1])] += float(full_raw[raw][ex][r][5])
			t_time[int(full_raw[raw][ex][r][1])] += float(full_raw[raw][ex][r][6])
			t_mem[int(full_raw[raw][ex][r][1])] += float(full_raw[raw][ex][r][7])
			
			times[int(full_raw[raw][ex][r][1])].append(float(full_raw[raw][ex][r][6]))
			if test_name.find("por") != -1:
				t_s_red += int(full_raw[raw][ex][r][10])
				t_f_red += int(full_raw[raw][ex][r][11])
				t_jpred += int(full_raw[raw][ex][r][12])
			
			#fill scatterplot for current util
			if int(full_raw[raw][ex][r][1]) == 0:
				scatter[e].append([int(full_raw[raw][ex][r][2]),float(full_raw[raw][ex][r][6]),f"{test_name}-unsched"])
			if int(full_raw[raw][ex][r][1]) == 1:
				scatter[e].append([int(full_raw[raw][ex][r][2]),float(full_raw[raw][ex][r][6]),f"{test_name}-sched"])
				if comp != -1:
					correctness_check(r,raw, comp, ex)
		
		#calc the standard deviation and mean
		t_std = numpy.std(times[0] + times[1])
		sched_std = numpy.std(times[0]) if len(times[0]) > 0 else "x"
		unsched_std = numpy.std(times[1]) if len(times[1]) > 0 else "x"
		std_dev.append([t_std, sched_std, unsched_std])
		
		
		t_mean = numpy.mean(times[0] + times[1])
		sched_mean = numpy.mean(times[0]) if len(times[0]) > 0 else "x"
		unsched_mean = numpy.mean(times[1]) if len(times[1]) > 0 else "x"
		mean.append([t_mean, sched_mean, unsched_mean])
		
		states.append([t_states[0]/max(1, len(full_raw[raw][ex])- t_sched) , t_states[1]/max(1,t_sched)])
		width.append([t_width[0]/max(1, len(full_raw[raw][ex])- t_sched ), t_width[1]/max(1,t_sched)])
		runtimes.append([t_time[0]/max(1, len(full_raw[raw][ex])- t_sched ), t_time[1]/max(1,t_sched)])
		mem.append([t_mem[0]/max(1, len(full_raw[raw][ex])- t_sched ), t_mem[1]/max(1,t_sched)])
		reductions.append([t_s_red/len(full_raw[raw][ex]), t_f_red/len(full_raw[raw][ex]), int(t_jpred/max(1,t_s_red))])
		sched.append(t_sched/len(full_raw[raw][ex]))
	parsed_results.append([test_name, states,width,runtimes,mem,reductions,sched,mean,std_dev])
			
def plot_scatters(ex_location):
	for ex in range(len(experiment)):
		plt.figure(figsize=(16, 12))
		# Create scatter plot with different colors for each type
		X = []
		Y = []
		Z = []
		for x,y,z in scatter[ex]:
			X.append(x)
			Y.append(y)
			Z.append(z)
		sns.scatterplot(x=X, y=Y, hue=Z)
		# Set labels for the x and y axes
		plt.xlabel('# of jobs')
		plt.ylabel('time')
		
		plt.title(f"runtime for for different jobsizes with U={experiment[ex]}");

		# Get the current legend
		legend = plt.legend()


		# Display the scatter plot
		#plt.show()
		plt.savefig(f"{ex_location}/scatter_u{experiment[ex]}.jpg");
		plt.close();
	
#state plots
def plot_avg_states(ex_location):
	labels = []
	for label in range(len(parsed_results)):
		labels.append(parsed_results[label][0])
		
	x_values = []
	for states in range(len(experiment)):
		x = []
		for l in range(len(parsed_results)):
			x.append(parsed_results[l][1][states][0]+parsed_results[l][1][states][1])
		x_values.append(x)
		
	plt.figure(figsize=(16, 12))
	plt.plot(experiment,x_values)
	plt.legend(labels)
	plt.title("Average # states")
	plt.xlabel("utilization")
	plt.ylabel("# states")

	plt.savefig(f"{ex_location}/avg_states.jpg");
	plt.close();

def plot_avg_states_unsched(ex_location):
	labels = []
	for label in range(len(parsed_results)):
		labels.append(parsed_results[label][0])
		
	x_values = []
	for states in range(len(experiment)):
		x = []
		for l in range(len(parsed_results)):
			x.append(parsed_results[l][1][states][0])
		x_values.append(x)
		
	plt.figure(figsize=(16, 12))
	plt.plot(experiment,x_values)
	plt.legend(labels)
	plt.title("Average # states for unschedulable sets")
	plt.xlabel("utilization")
	plt.ylabel("# states")

	plt.savefig(f"{ex_location}/avg_states_unsched.jpg");
	plt.close();
	
def plot_avg_states_sched(ex_location):
	labels = []
	for label in range(len(parsed_results)):
		labels.append(parsed_results[label][0])
		
	x_values = []
	for states in range(len(experiment)):
		x = []
		for l in range(len(parsed_results)):
			x.append(parsed_results[l][1][states][1])
		x_values.append(x)
		
	plt.figure(figsize=(16, 12))
	plt.plot(experiment,x_values)
	plt.legend(labels)
	plt.title("Average # states for schedulable sets")
	plt.xlabel("utilization")
	plt.ylabel("# states")

	plt.savefig(f"{ex_location}/avg_states_sched.jpg");
	plt.close();

#width plots
def plot_avg_width(ex_location):
	labels = []
	for label in range(len(parsed_results)):
		labels.append(parsed_results[label][0])
		
	x_values = []
	for states in range(len(experiment)):
		x = []
		for l in range(len(parsed_results)):
			x.append(parsed_results[l][2][states][0]+parsed_results[l][2][states][1])
		x_values.append(x)
		
	plt.figure(figsize=(16, 12))
	plt.plot(experiment,x_values)
	plt.legend(labels)
	plt.title("Average maximum exploration width")
	plt.xlabel("utilization")
	plt.ylabel("# states")

	plt.savefig(f"{ex_location}/avg_width.jpg");
	plt.close();

def plot_avg_width_unsched(ex_location):
	labels = []
	for label in range(len(parsed_results)):
		labels.append(parsed_results[label][0])
		
	x_values = []
	for states in range(len(experiment)):
		x = []
		for l in range(len(parsed_results)):
			x.append(parsed_results[l][2][states][0])
		x_values.append(x)
		
	plt.figure(figsize=(16, 12))
	plt.plot(experiment,x_values)
	plt.legend(labels)
	plt.title("Average maximum exploration width unsched")
	plt.xlabel("utilization")
	plt.ylabel("# states")

	plt.savefig(f"{ex_location}/avg_width_unsched.jpg");
	plt.close();
	
def plot_avg_width_sched(ex_location):
	labels = []
	for label in range(len(parsed_results)):
		labels.append(parsed_results[label][0])
		
	x_values = []
	for states in range(len(experiment)):
		x = []
		for l in range(len(parsed_results)):
			x.append(parsed_results[l][2][states][1])
		x_values.append(x)
		
	plt.figure(figsize=(16, 12))
	plt.plot(experiment,x_values)
	plt.legend(labels)
	plt.title("Average maximum exploration width sched")
	plt.xlabel("utilization")
	plt.ylabel("# states")

	plt.savefig(f"{ex_location}/avg_width_sched.jpg");
	plt.close();

#runtime plots
def plot_avg_runtime(ex_location):
	labels = []
	for label in range(len(parsed_results)):
		labels.append(parsed_results[label][0])
		
	x_values = []
	for states in range(len(experiment)):
		x = []
		for l in range(len(parsed_results)):
			x.append(parsed_results[l][3][states][0]+parsed_results[l][3][states][1])
		x_values.append(x)
		
	plt.figure(figsize=(16, 12))
	plt.plot(experiment,x_values)
	plt.legend(labels)
	plt.title("Average runtime")
	plt.xlabel("utilization")
	plt.ylabel("time (s)")

	plt.savefig(f"{ex_location}/avg_runtime.jpg");
	plt.close();

def plot_avg_runtime_unsched(ex_location):
	labels = []
	for label in range(len(parsed_results)):
		labels.append(parsed_results[label][0])
		
	x_values = []
	for states in range(len(experiment)):
		x = []
		for l in range(len(parsed_results)):
			x.append(parsed_results[l][3][states][0])
		x_values.append(x)
		
	plt.figure(figsize=(16, 12))
	plt.plot(experiment,x_values)
	plt.legend(labels)
	plt.title("Average runtime for unschedulable sets")
	plt.xlabel("utilization")
	plt.ylabel("time (s)")

	plt.savefig(f"{ex_location}/avg_runtime_unsched.jpg");
	plt.close();
	
def plot_avg_runtime_sched(ex_location):
	labels = []
	for label in range(len(parsed_results)):
		labels.append(parsed_results[label][0])
		
	x_values = []
	for states in range(len(experiment)):
		x = []
		for l in range(len(parsed_results)):
			x.append(parsed_results[l][3][states][1])
		x_values.append(x)
		
	plt.figure(figsize=(16, 12))
	plt.plot(experiment,x_values)
	plt.legend(labels)
	plt.title("Average runtime for schedulable sets")
	plt.xlabel("utilization")
	plt.ylabel("time (s)")

	plt.savefig(f"{ex_location}/avg_runtime_sched.jpg");
	plt.close();

#memory plots
def plot_avg_mem(ex_location):
	labels = []
	for label in range(len(parsed_results)):
		labels.append(parsed_results[label][0])
		
	x_values = []
	for states in range(len(experiment)):
		x = []
		for l in range(len(parsed_results)):
			x.append(parsed_results[l][4][states][0]+parsed_results[l][4][states][1])
		x_values.append(x)
		
	plt.figure(figsize=(16, 12))
	plt.plot(experiment,x_values)
	plt.legend(labels)
	plt.title("Average memory useage")
	plt.xlabel("utilization")
	plt.ylabel("memory (MB)")

	plt.savefig(f"{ex_location}/avg_mem.jpg");
	plt.close();

def plot_avg_mem_unsched(ex_location):
	labels = []
	for label in range(len(parsed_results)):
		labels.append(parsed_results[label][0])
		
	x_values = []
	for states in range(len(experiment)):
		x = []
		for l in range(len(parsed_results)):
			x.append(parsed_results[l][4][states][0])
		x_values.append(x)
		
	plt.figure(figsize=(16, 12))
	plt.plot(experiment,x_values)
	plt.legend(labels)
	plt.title("Average memory useage unsched")
	plt.xlabel("utilization")
	plt.ylabel("memory MB")

	plt.savefig(f"{ex_location}/avg_mem_unsched.jpg");
	plt.close();
	
def plot_avg_mem_sched(ex_location):
	labels = []
	for label in range(len(parsed_results)):
		labels.append(parsed_results[label][0])
		
	x_values = []
	for states in range(len(experiment)):
		x = []
		for l in range(len(parsed_results)):
			x.append(parsed_results[l][4][states][1])
		x_values.append(x)
		
	plt.figure(figsize=(16, 12))
	plt.plot(experiment,x_values)
	plt.legend(labels)
	plt.title("Average memory useage sched")
	plt.xlabel("utilization")
	plt.ylabel("memory (MB)")

	plt.savefig(f"{ex_location}/avg_mem_sched.jpg");
	plt.close();

#reduction plots
def plot_avg_s_red(ex_location):
	labels = []
	for label in range(len(parsed_results)):
		labels.append(parsed_results[label][0])
		
	x_values = []
	for states in range(len(experiment)):
		x = []
		for l in range(len(parsed_results)):
			x.append(parsed_results[l][5][states][0])
		x_values.append(x)
		
	plt.figure(figsize=(16, 12))
	plt.plot(experiment,x_values)
	plt.legend(labels)
	plt.title("Succesfull reductions on average")
	plt.xlabel("utilization")
	plt.ylabel("# reductions")

	plt.savefig(f"{ex_location}/avg_s_red.jpg");
	plt.close();

def plot_avg_f_red(ex_location):
	labels = []
	for label in range(len(parsed_results)):
		labels.append(parsed_results[label][0])
		
	x_values = []
	for states in range(len(experiment)):
		x = []
		for l in range(len(parsed_results)):
			x.append(parsed_results[l][5][states][1])
		x_values.append(x)
		
	plt.figure(figsize=(16, 12))
	plt.plot(experiment,x_values)
	plt.legend(labels)
	plt.title("Failed reductions on average")
	plt.xlabel("utilization")
	plt.ylabel("# reductions")

	plt.savefig(f"{ex_location}/avg_f_red.jpg");
	plt.close();
	
def plot_avg_job_per_s_por(ex_location):
	labels = []
	for label in range(len(parsed_results)):
		labels.append(parsed_results[label][0])
		
	x_values = []
	for states in range(len(experiment)):
		x = []
		for l in range(len(parsed_results)):
			x.append(parsed_results[l][5][states][2])
		x_values.append(x)
		
	plt.figure(figsize=(16, 12))
	plt.plot(experiment,x_values)
	plt.legend(labels)
	plt.title("Average # jobs per succesful por")
	plt.xlabel("utilization")
	plt.ylabel("# jobs")

	plt.savefig(f"{ex_location}/avg_jobs_per_por.jpg");
	plt.close();

#schedulability plot
def plot_schedulability(ex_location):
	labels = []
	for label in range(len(parsed_results)):
		labels.append(parsed_results[label][0])
		
	x_values = []
	for states in range(len(experiment)):
		x = []
		for l in range(len(parsed_results)):
			x.append(parsed_results[l][6][states])
		x_values.append(x)
		
	plt.figure(figsize=(16, 12))
	plt.plot(experiment,x_values)
	plt.legend(labels)
	plt.title("schedulability ratio")
	plt.xlabel("utilization")
	plt.ylabel("ratio")

	plt.savefig(f"{ex_location}/sched.jpg");
	plt.close();

#std deviation and mean
def write_std_mean(ex_location):
	with open(f"{ex_location}/std_mean.csv", "w") as sm:
		for m in range(len(parsed_results)):
			total = parsed_results[m][7][0] + ", total"
			schedulable = parsed_results[m][7][0] + ", schedulable"
			unschedulable = parsed_results[m][7][0] + ", unschedulable"
			for ex in range(len(experiment)):
				total += f",{parsed_results[m][7][ex+1][0]}"
				schedulable += f",{parsed_results[m][7][ex+1][0]}"
				unschedulable += f",{parsed_results[m][7][ex+1][0]}"
			sm.write(total+"\n")
			sm.write(schedulable+"\n")
			sm.write(unschedulable+"\n")
			
			
			total = parsed_results[m][8][0] + ", total"
			schedulable = parsed_results[m][8][0] + ", schedulable"
			unschedulable = parsed_results[m][8][0] + ", unschedulable"
			for ex in range(len(experiment)):
				total += f",{parsed_results[m][8][ex+1][0]}"
				schedulable += f",{parsed_results[m][8][ex+1][0]}"
				unschedulable += f",{parsed_results[m][8][ex+1][0]}"
			sm.write(total+"\n")
			sm.write(schedulable+"\n")
			sm.write(unschedulable+"\n")
		
def parse_experiment(path):
	set_config(path)
	print(test_names)
	print(comp_names)
	for t in test_names:
		parse_result(path, t)
	for name in range(len(test_names)):
		parse_raw(test_names[name],comp_names[name])
	
	plot_scatters(path)
	
	plot_avg_states(path)
	plot_avg_states_unsched(path)
	plot_avg_states_sched(path)
	
	plot_avg_width(path)
	plot_avg_width_unsched(path)
	plot_avg_width_sched(path)
	
	plot_avg_mem(path)
	plot_avg_mem_unsched(path)
	plot_avg_mem_sched(path)
	
	plot_avg_runtime(path)
	plot_avg_runtime_unsched(path)
	plot_avg_runtime_sched(path)
	
	
	plot_avg_s_red(path)
	plot_avg_f_red(path)
	plot_avg_job_per_s_por(path)
	
	plot_schedulability(path)
	
	write_std_mean(path)
		
	
def main(args):
	with open("config.txt") as c_main:
		for test_run in c_main.read().splitlines():
			print(f"Generating images for: {test_run}")
			parse_experiment(test_run)
	
	return 0

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
