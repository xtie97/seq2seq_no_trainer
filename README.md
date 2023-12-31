# Seq2seq model finetuning without using trainer function

This repository provides an example of how to run a sequence-to-sequence (seq2seq) model using PEGASUS-large. The model operates on a dataset that is stored in the `archive` folder as CSV files.

## Installation

To use this code, you will need to install the following dependencies:

- Accelerate
- Transformer

The versions of the required python libraries can be found in the provided `Dockerfile`. Build the Docker image if needed.

## Usage
To finetune the seq2seq model on your dataset, follow these steps:

1. Clone this repository: 

```shell
git clone https://github.com/xtie97/seq2seq_no_trainer.git
```

2. Naviage to the root directory:
```shell
cd seq2seq_no_trainer
```

3. Ensure that your dataset is stored in the "archive" folder as a CSV file.

4. Run the following command:
```shell
python finetune_seq2seq_no_trainer.py --model_name_or_path google/pegasus-large --output_dir pegasus_ex1 --learning_rate 4e-4 --per_device_train_batch_size 4 --gradient_accumulation_steps 8
```

## Inference
To evaluate the finetuned seq2seq model on your testing set, run the following command:
```shell
python predict_seq2seq.py
```

Make sure you have the necessary dependencies installed before running the command.

Adjust any parameters in the  `model.generate()` function to suit your specific inference requirements. 
