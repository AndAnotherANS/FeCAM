import itertools
import subprocess
import json

alpha_vals = [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]

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
    
