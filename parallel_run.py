import ast
import itertools
import subprocess
import tempfile
import os

def read_parameters(file_path):
    parameters = {}
    with open(file_path, 'r') as file:
        for line in file:
            if not line.strip():  
                continue
            key, value = line.strip().split('=')
            try:
                parsed = ast.literal_eval(value) 
            except Exception:
                parsed = value
            parameters[key.strip()] = parsed
    return parameters

def expand_parameter_grid(params):
    keys = []
    values = []
    for k, v in params.items():
        if isinstance(v, list):
            keys.append(k)
            values.append(v)
        else:
            keys.append(k)
            values.append([v])  

    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))

def run_simulation(param_dict):
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".txt") as tmp:
        for k, v in param_dict.items():
            tmp.write(f"{k}={v}\n")
        tmp_path = tmp.name

    cmd = ["python3", "main.py", tmp_path]
    print("Launching:", " ".join(cmd))
    subprocess.Popen(cmd)  

if __name__ == "__main__":
    params = read_parameters("parameters.txt")
    for combo in expand_parameter_grid(params):
        run_simulation(combo)