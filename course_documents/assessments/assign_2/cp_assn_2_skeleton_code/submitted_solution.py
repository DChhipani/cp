import random
import networkx as nx
import time
import numpy as np
import pulp
import minizinc as mz
from datetime import timedelta

# ========================= Parameters ===================== #
FF_BUDGET = 1           # Firefighter budget each turn
NUM_TESTS = 5           # Number of tests to run for benchmarking
pulpTimes = []          # To store ILP solver times
mznTimes = []           # To store CP solver times
# ========================================================== #

def run_ilp(instance_graph, start_node=1, timeout=1000):
    """
    Runs the ILP implementation for the infectious firefighter problem.
    
    Args:
        instance_graph (networkx.Graph): The graph representing the network.
        start_node (int): The node where the fire starts.
        timeout (int): Maximum time in milliseconds allowed for the solver to run.

    Returns:
        dict: Dictionary containing the number of nodes saved and any other useful info.
    """
    num_nodes = len(instance_graph.nodes)
    adjacency_matrix = nx.to_numpy_array(instance_graph).astype(int).tolist()
    
    # Initialize variables for ILP formulation
    prob = pulp.LpProblem("Firefighter_ILP", pulp.LpMinimize)
    b = pulp.LpVariable.dicts("b", (range(num_nodes), range(num_nodes)), 0, 1, pulp.LpBinary)
    d = pulp.LpVariable.dicts("d", (range(num_nodes), range(num_nodes)), 0, 1, pulp.LpBinary)

    # Objective: minimize the number of burnt nodes at the end of the process
    prob += pulp.lpSum(b[x][num_nodes - 1] for x in range(num_nodes))

    # Initial conditions and constraints
    for x in range(num_nodes):
        prob += d[x][0] == 0
        prob += b[x][0] == (1 if x == start_node else 0)

    for t in range(1, num_nodes):
        prob += pulp.lpSum(d[x][t] - d[x][t - 1] for x in range(num_nodes)) <= FF_BUDGET  # Budget constraint
        for x in range(num_nodes):
            prob += b[x][t] >= b[x][t - 1]
            prob += d[x][t] >= d[x][t - 1]
            prob += b[x][t] + d[x][t] <= 1  # Constraint: cannot protect and burn the same node
            # Fire spread constraint
            neighbors = [i for i, connected in enumerate(adjacency_matrix[x]) if connected]
            prob += pulp.lpSum(b[y][t - 1] for y in neighbors) >= b[x][t]

    # Solve the problem with a timeout
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=timeout / 1000)
    start_time = time.time()
    prob.solve(solver)
    ilp_duration = time.time() - start_time

    # Return number of saved nodes
    saved_nodes = num_nodes - sum(pulp.value(b[x][num_nodes - 1]) for x in range(num_nodes))
    
    # Return result dictionary
    return {'num_saved': saved_nodes, 'duration': ilp_duration}

def run_cp(instance_graph, start_node=1, timeout=1000):
    """
    Runs the CP implementation for the infectious firefighter problem using MiniZinc.
    
    Args:
        instance_graph (networkx.Graph): The graph representing the network.
        start_node (int): The node where the fire starts.
        timeout (int): Maximum time in milliseconds allowed for the solver to run.

    Returns:
        dict: Dictionary containing the number of nodes saved and any other useful info.
    """
    num_nodes = len(instance_graph.nodes)
    adjacency_matrix = nx.to_numpy_array(instance_graph).astype(int).tolist()

    # Prepare the MiniZinc model and solver
    model = mz.Model()
    model.add_file("infectious_firefighter.mzn")
    solver = mz.Solver.lookup("gecode")

    # Set the instance values
    instance = mz.Instance(solver, model)
    instance["n"] = num_nodes
    instance["T"] = num_nodes  # Set time horizon as number of nodes
    instance["f"] = [start_node + 1]  # Convert to 1-based index for MiniZinc
    instance["G"] = adjacency_matrix
    instance["budget"] = FF_BUDGET

    # Solve with timeout
    start_time = time.time()
    result = instance.solve(timeout=timedelta(milliseconds=timeout))
    cp_duration = time.time() - start_time

    # Return number of saved nodes from the result
    saved_nodes = int(str(result)) if result.solution else 0

    # Return result dictionary
    return {'num_saved': saved_nodes, 'duration': cp_duration}

def run_comparison_tests(num_tests=NUM_TESTS):
    """
    Run performance comparison between ILP and CP solvers.
    """
    ilp_times, cp_times = [], []
    ilp_saved, cp_saved = [], []

    for test_id in range(num_tests):
        # Create a random graph for testing
        num_nodes = 10
        graph = nx.fast_gnp_random_graph(num_nodes, 0.2)
        start_node = random.choice(list(graph.nodes))

        # Run ILP Solver
        ilp_result = run_ilp(graph, start_node)
        ilp_saved.append(ilp_result['num_saved'])
        ilp_times.append(ilp_result['duration'])

        # Run CP Solver
        cp_result = run_cp(graph, start_node)
        cp_saved.append(cp_result['num_saved'])
        cp_times.append(cp_result['duration'])

        print(f"Test {test_id + 1}: ILP saved = {ilp_result['num_saved']} in {ilp_result['duration']:.2f}s, "
              f"CP saved = {cp_result['num_saved']} in {cp_result['duration']:.2f}s")

    # Plot the comparison results
    plt.figure(figsize=(12, 6))

    # Average nodes saved
    plt.subplot(1, 2, 1)
    plt.bar(["ILP", "CP"], [np.mean(ilp_saved), np.mean(cp_saved)], color=['blue', 'green'])
    plt.ylabel("Average Nodes Saved")
    plt.title("Nodes Saved Comparison")

    # Average durations
    plt.subplot(1, 2, 2)
    plt.bar(["ILP", "CP"], [np.mean(ilp_times), np.mean(cp_times)], color=['blue', 'green'])
    plt.ylabel("Average Duration (s)")
    plt.title("Solver Duration Comparison")

    plt.tight_layout()
    plt.show()

# Run the comparison tests
run_comparison_tests()
