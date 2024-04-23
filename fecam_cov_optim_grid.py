import itertools
import subprocess
import json

alpha_vals = [100000]
lr = [0.00001]
batch_size = [16, 32]

config_path = './exps/FeCAM_cifar100.json'

print(alpha_vals)

run_n = 1
for alpha, learning_rate, batch in itertools.product(alpha_vals, lr, batch_size):
    with open(config_path, 'r') as file:
        data = json.load(file)
    data['optimized_cov_alpha'] = alpha
    data['cov_optim_lr'] = learning_rate
    data['cov_optim_batch_size'] = batch
    with open(config_path, 'w') as file:
        json.dump(data, file, indent=4)
    process = subprocess.Popen(['python', 'main.py', '--config', config_path, "--run_n", str(run_n)])
    process.wait()

    run_n += 1
    
