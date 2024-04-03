import random

def objective_function(x):
	return x*x

def hill_climb(starting_point,max_iterations=1000,step_size=0.01):
	current_solution = starting_point
	current_value = objective_function(current_solution)
	
	for _ in range(max_iterations):
		new_solution=current_solution + random.choice([-1,1]) * step_size
		new_value= objective_function(new_solution)
		
		if new_value>current_value: 
			current_solution=new_solution
			current_value=new_value
	return current_solution, current_value
	
starting_point=float(input("Enter the starting point:"))
best_solution,best_value=hill_climb(starting_point)
print("Best Sol:",best_solution)
print("Best Val:",best_value)
