import os
import torch
import json
import pandas as pd
import sys
from model_factory import *
from dataset_factory import *
from torch.cuda.amp import autocast

def convert_question(prediction, vocab):
        """
        Converts predicted question indices to word tokens
        prediction: N x Q
        """
        word_idxs = prediction.cpu().numpy()
        captions = []
        for i in range(prediction.shape[0]):
            words = [vocab.idx2word[idx].lower() for idx in word_idxs[i]]
            try:
                end_idx = words.index('<end>') + 1 # cut off after predicting end
            except ValueError as e:
                end_idx = None
            
            words = words[:end_idx]
            captions.append(words)
        
        to_return = []
        for i in range(len(captions)):
            clean_list = ['<pad>', '<start>', '<end>', '<unk>', ' ', ';', ',', '.', '\'', '-', '(', ')', '[', ']', '@', '$', \
                '%', '!', '?', '/', '+', '^', '&', '*']
            cleaned_caption = [word for word in captions[i] if word not in clean_list]
            to_return.append(cleaned_caption)

        return to_return

if __name__ == '__main__':
    args = sys.argv[1:]
    config_name = args[0]
    
    with open (config_name +'.json', 'r') as f:
        config = json.load(f)

    model_dir = os.path.join('experiment_data', config['experiment_name'])
    vocab, train_loader, val_loader, test_loader = get_datasets(config)
    model = get_model(config, vocab)
    model.load_state_dict(torch.load(os.path.join(model_dir, 'best_model.pth')))
    model.to("cuda")
    model.eval()

    df_dic = {'passage': [], 'answer': [], 'pred_question': [], 'true_question': []}

    for i, (passages, answers, questions) in enumerate(test_loader):
        with torch.no_grad():

            passages = passages.cuda().long()
            answers = answers.cuda().long()
            questions = questions.cuda().long()
            
            with autocast():
                predicted_questions = model.predict(passages, answers)

            # list of lists
            passages = convert_question(passages, vocab)
            answers = convert_question(answers, vocab)
            questions = convert_question(questions, vocab)
            predicted_questions = convert_question(predicted_questions, vocab)
            
            # list of strings
            passages = [' '.join(elem) for elem in passages]
            answers = [' '.join(elem) for elem in answers]
            questions = [' '.join(elem) for elem in questions]
            predicted_questions = [' '.join(elem) for elem in predicted_questions]

            df_dic['passage'] += passages
            df_dic['answer'] += answers
            df_dic['pred_question'] += predicted_questions
            df_dic['true_question'] += questions

    out_path = os.path.join(model_dir, 'questions.csv')
    pd.DataFrame(df_dic).to_csv(out_path, index=False)
