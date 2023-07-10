# seq2seq_no_trainer 

This repository provides an example of how to run a sequence-to-sequence (seq2seq) model using PEGASUS-large. The model operates on a dataset that is stored in the "archive" folder as CSV files.

## Installation

To use this code, you will need to install the following dependencies:

- Accelerate
- Transformer

The versions of the required packages can be found in the provided Dockerfile. 

## Usage
To run the seq2seq model on your dataset, follow these steps:

1. Clone this repository: 

```shell
git clone https://github.com/xtie97/seq2seq_no_trainer.git

2. Naviage to the repository's root directory: 
```shell
cd seq2seq_no_trainer

3. Ensure that your dataset is stored in the "archive" folder as a CSV file.

4. run the following command:
```shell
python finetune_PEGASUS_no_trainer.py --model_name_or_path google/pegasus-large --output_dir pegasus_ex1 --learning_rate 4e-4 --per_device_train_batch_size 4 --gradient_accumulation_steps 8 
