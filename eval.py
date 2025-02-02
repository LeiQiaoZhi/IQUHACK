import warnings
warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")
import sys
import os
from datetime import datetime
import concurrent.futures
from qiskit_aer import AerSimulator
import matplotlib
matplotlib.use("Agg")  # Use a non-GUI backend before importing pyplot
import matplotlib.pyplot as plt
from Functions import *
import multiprocessing
from tqdm.contrib.concurrent import process_map
from ours import *

shots = 500_000
num_steps = [100, 30, 50, 100, 50, 50, 50]
learning_rate = 0.1
# num_steps = [2 for _ in range(7)]


def process_graph(i, graph, log_dir):
    ansatz = build_ansatz(graph)
    ansatz.draw(output="mpl", fold=1, filename=f"{log_dir}/{i+1}_ansatz.png")
    ham = build_maxcut_hamiltonian(graph)
    plt.close("all")

    qit_evolver = QITEvolver(ham, ansatz)
    qit_evolver.evolve(num_steps=num_steps[i], lr=learning_rate, verbose=False, i=i+1)

    qit_evolver.plot_convergence()
    plt.savefig(f"{log_dir}/{i+1}_converge.png")
    plt.close("all")

    backend = AerSimulator()
    optimized_state = ansatz.assign_parameters(qit_evolver.param_vals[-1])
    optimized_state.measure_all()
    counts = backend.run(optimized_state, shots=shots).result().get_counts()

    brute_force_results = brute_force(graph, verbose=False)

    base_score = final_score(graph, brute_force_results, counts, shots, ansatz, "base", verbose=False)
    balanced_score = final_score(graph, brute_force_results, counts, shots, ansatz, "balanced", verbose=False)
    connected_score = final_score(graph, brute_force_results, counts, shots, ansatz, "connected", verbose=False)

    log_file_path = f"{log_dir}/_output.log"
    with open(log_file_path, "a") as log_file:
        sys.stdout = log_file
        sys.stderr = log_file  # Capture errors
        print(f"{i+1} Base score: {base_score}")
        print(f"{i+1} Balanced score: {balanced_score}")
        print(f"{i+1} Connected score: {connected_score}\n")
    plt.close("all")

    return (i, base_score, balanced_score, connected_score)

def main():
    graphs = get_graphs()
    # graphs = [graphs[0], graphs[3], graphs[2]]

    # set up logs
    timestamp = datetime.now().strftime("%H-%M-%S")
    log_dir = f"logs/{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = f"{log_dir}/_output.log"
    with open(log_file_path, "a") as log_file:
        sys.stdout = log_file
        sys.stderr = log_file  # Capture errors
        print(f"Num steps: {num_steps}")
        print(f"Learning rate: {learning_rate}")
        print(f"Shots: {shots}\n")

    # Run in parallel
    # multiprocessing.set_start_method("spawn", force=True)  # Fixes multiprocessing on macOS
    # results = process_map(process_graph, range(len(graphs)), graphs, max_workers=multiprocessing.cpu_count())

    # lock = multiprocessing.Manager().Lock()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(process_graph, range(len(graphs)), graphs, [log_dir]*len(graphs)))

    # # Sort and print results
    for result in results:
        print(f"Graph {result[0]+1} - Base: {result[1]}, Balanced: {result[2]}, Connected: {result[3]}")


if __name__ == "__main__":
    main()