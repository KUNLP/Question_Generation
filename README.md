# Question_Generation

# Dependencies
* python 3.7
* PyTorch 1.6.0
* Transformers 4.3.3
* AttrDict

# Model Architecture

# Data 
* KLUE Machine Reading Comprehension [Click](https://klue-benchmark.com/tasks/72/data/download)

# Train & Test
python3.7 run_qg --train_file [train file] --test_file [test_file] --from_init_weight --do_train
python3.7 run_qg --test_file [test_file] --do_evaluate
