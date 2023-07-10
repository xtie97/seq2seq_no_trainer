import os 
os.environ['CURL_CA_BUNDLE'] = ''
os.system("pip --trusted-host pypi.org --trusted-host files.pythonhosted.org install bert-score fastai==2.7.11") 
os.system("pip --trusted-host pypi.org --trusted-host files.pythonhosted.org install scipy uvicorn gunicorn==19.9.0 fastapi uvloop httptools")
os.system("pip --trusted-host pypi.org --trusted-host files.pythonhosted.org install ohmeow-blurr==1.0.5")
os.system("pip --trusted-host pypi.org --trusted-host files.pythonhosted.org install huggingface-hub --upgrade")
os.system("pip --trusted-host pypi.org --trusted-host files.pythonhosted.org install -U transformers")
os.system("pip --trusted-host pypi.org --trusted-host files.pythonhosted.org install requests==2.27.1")

import pandas as pd
from tqdm import tqdm 
import torch
from fastai.text.all import *
from transformers import *
from blurr.text.data.all import *
from blurr.text.modeling.all import *
import datasets
import time 

def predict_text(reports, input_col, length_penalty, model_template, savename, finetune_nepoch=1):

    #Import the pretrained model
    tokenizer = AutoTokenizer.from_pretrained(model_template.split('/')[0])
    model = AutoModelForSeq2SeqLM.from_pretrained(model_template).eval().to('cuda')
    #Create mini-batch and define parameters
    encoder_max_length = 1024
    decoder_max_length = 512
   
    # map data correctly
    def generate_summary(batch):
        # Tokenizer will automatically set [BOS] <text> [EOS]
        inputs = tokenizer(batch[input_col], padding="max_length", truncation=True, max_length=encoder_max_length, return_tensors="pt")
        input_ids = inputs.input_ids.to("cuda")
        attention_mask = inputs.attention_mask.to("cuda")
        outputs = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=decoder_max_length, 
                                                                           num_beam_groups=1,
                                                                           num_beams=4, 
                                                                           do_sample=False,
                                                                           diversity_penalty=0.0, # 1.0 
                                                                           num_return_sequences=1, 
                                                                           length_penalty=length_penalty,
                                                                           no_repeat_ngram_size=3,
                                                                           early_stopping=True)

        # all special tokens including will be removed
        output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        batch["pred"] = output_str
        return batch

    test_data = datasets.Dataset.from_pandas(reports)
    results = test_data.map(generate_summary, batched=True, batch_size=8)
    pred_str = results["pred"]
    reports['AI_impression'] = pred_str
     
    os.makedirs('./test_length_penalty_{}/'.format(length_penalty), exist_ok=True)
    reports.to_excel('./test_length_penalty_{}/test_{}_epoch{:02d}_all.xlsx'.format(length_penalty, 
                                                                                    savename, 
                                                                                    finetune_nepoch), 
                                                                                    index=False)

def filter_test_data(df):
    # select the recent 5 years data
    exam_date = df['Exam Date Time']
    exam_date_new = []
    for ii in tqdm(range(len(exam_date))):
        if '2018' in exam_date[ii] or '2019' in exam_date[ii] or '2020' in exam_date[ii] \
        or '2021' in exam_date[ii] or '2022' in exam_date[ii] or '2023' in exam_date[ii]:
            exam_date_new.append(exam_date[ii])
        else:
            exam_date_new.append('Remove')
    df['Exam Date Time'] = exam_date_new
    # drop the data with no exam date
    #df = df[df['Exam Date Time'] != 'Remove'].reset_index(drop=True)
    #df = df[df['Study Description'] == 'PET CT WHOLE BODY'].reset_index(drop=True)
    #df = df.sample(n=1000, random_state=716).reset_index(drop=True)
    return df 


if __name__ == '__main__':
    # Testing    
    df = pd.read_excel('./archive/test.xlsx')
    df = filter_test_data(df)
    df['impressions'] = df['impressions'].apply(lambda x: x.replace('\n',' '))
    df['findings_info'] = df['findings_info'].apply(lambda x: x.replace('\n',' '))
    print('Start testing')
    finetune_nepoch = 6
    predict_text(df, 'findings_info', length_penalty=2.0, model_template=f'pegasus_ex1/epoch_{finetune_nepoch}', savename='pegasus_ex1', finetune_nepoch=finetune_nepoch)