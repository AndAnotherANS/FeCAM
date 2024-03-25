import itertools
import subprocess
import json

alpha_vals = [1000, 0.001, 0.1, 1, 10, 100]

config_path = './exps/FeCAM_cifar100.json'

print(alpha_vals)

for alpha in alpha_vals:
    with open(config_path, 'r') as file:
        data = json.load(file)
    data['optimized_cov_alpha'] = alpha
    with open(config_path, 'w') as file:
        json.dump(data, file, indent=4)
    process = subprocess.Popen(['python', 'main.py', '--config', config_path])
    process.wait()
    
