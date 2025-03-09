from matplotlib import pyplot as plt
import pennylane as qml
from pennylane import numpy as np
from scipy.optimize import minimize
from codes.rl_qaoa import *
from codes.data_process import add_constraint
import random
import json
import os


# Define QAOA depth
def data_to_QUBO(matrix, hamming_weight, l):
    return -np.diag([1] * len(matrix)) + matrix / hamming_weight * l

def QpasqalOptimized(matrix, number):
    res = copy.deepcopy(matrix)
    N = res.shape[0]
    for i in range(N):
        res[i][i] = 0
    for i in range(N):
        for j in range(i + 1, N):
            res[i][j] = res[i][j] + (matrix[i][i] + matrix[j][j]) / ((number - 1) * 2)
            res[j][i] = res[j][i] + (matrix[i][i] + matrix[j][j]) / ((number - 1) * 2)

    return res



def test_qaa(num_episode , num_epoch ,beta , lr ,matrix_idx ,model_name ,matrix_size,hamming_weight ,save_dir):
    depth = 1
    size =matrix_size
    seed = 50
    penalty = 1
    with open(f"./data/matrices{size}by{size}.json", "r") as f:
        matrices_data = json.load(f)

    # Generate a QUBO matrix that is challenging for classical QAOA optimization

    Q = data_to_QUBO(np.array(matrices_data[matrix_idx]), hamming_weight, 20)
    Q_cal = zero_lower_triangle(
    Q + add_constraint([1] * size, hamming_weight) * penalty
)

    n = Q.shape[0]
    n_c = 2

    init_params = np.reshape(np.array([0, 0.0] * (n - n_c)), -1)

    # RL-QAOA setup
    rl_qaa = RL_QAA(
    Q_cal,
    n,
    np.array([[beta] * int((n**2)) for i in range(n - n_c)]),
    learning_rate_init=lr,
    )
    final_config = rl_qaa.rqaoa_execute()
    rl_qaa.n_c = n_c
    print(
    f"classical_result : {float(final_config[2])},best : {rl_qaa.node_assignments}"
)
    # Execute RQAOA
    rl_qaa.RL_QAOA(
        episodes=num_episode,
        epochs=num_epoch,
        log_interval=25,
        correct_ans=float(final_config[2]),
    )

    data = {
        "cal_list": matrix_idx,
        "QAOA_list": [
            list(rl_qaa.avg_values),float(final_config[2]),int(rl_qaa.tree.node_num),
        ],
    }

    with open(f"{save_dir}/{model_name}_data_{matrix_idx}.json", "w") as json_file:
        json.dump(data, json_file, indent=4)

    data = rl_qaa.avg_values
    optimal_value = float(final_config[2])
    plt.figure(figsize=(10, 5))
    plt.plot(data, marker="o", linestyle="-", color="b", label="Optimization Progress")
    plt.axhline(y=optimal_value, color="r", linestyle="--", label="Optimal Value")
    plt.xlabel("Iterations")
    plt.ylabel("Value")
    plt.title("Optimization Process")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}/{model_name}_Optimization_Process_list_{matrix_idx}.png")

    plt.figure(figsize=(10, 5))
    plt.plot(data, marker="o", linestyle="-", color="b", label="Optimization Progress")
    plt.axhline(y=optimal_value, color="r", linestyle="--", label="Optimal Value")

    plt.ylim(optimal_value, optimal_value + 0.1)
    plt.xlabel("Iterations")
    plt.ylabel("Value")
    plt.title("Zoomed View: Convergence to Optimal Value")
    plt.legend()

    plt.grid(True)
    plt.title(
    f"Zoomed View: Convergence to Optimal Value - cal_list={matrix_idx}"
)
    plt.savefig(f"{save_dir}/{model_name}_zoomed_cal_list_{matrix_idx}.png")
    plt.close()

if __name__ =="__main__":
    test_qaa(num_episode=1,num_epoch=1,beta=25.0,matrix_idx=1,model_name="RL_QAA")
