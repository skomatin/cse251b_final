################################################################################
# CSE 253: Programming Assignment 4
# Code snippet by Ajit Kumar, Savyasachi
# Fall 2020
################################################################################

import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime

from constants import *
from dataset_factory import get_datasets
from model_factory import get_model
from file_utils import *

import warnings
import json

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torchvision.transforms as transforms

from datasets import load_metric

warnings.filterwarnings("ignore", category=UserWarning)
torch.autograd.set_detect_anomaly(True)
# Class to encapsulate a neural experiment.
# The boilerplate code to setup the experiment, log stats, checkpoints and plotting have been provided to you.
# You only need to implement the main training logic of your experiment and implement train, val and test methods.
# You are free to modify or restructure the code as per your convenience.
class Experiment(object):
    def __init__(self, name):
        config_data = read_file_in_dir(sys.path[0], name + '.json')
        if config_data is None:
            raise Exception("Configuration file doesn't exist: ", name)

        self.__config_data = config_data
        self.__name = config_data['experiment_name']
        self.__experiment_dir = os.path.join(ROOT_STATS_DIR, self.__name)

        # Load Datasets
        self.__vocab, self.__train_loader, self.__val_loader, self.__test_loader = get_datasets(config_data)

        # Setup Experiment
        self.__generation_config = config_data['generation']
        self.__epochs = config_data['experiment']['num_epochs']
        self.__current_epoch = 0
        self.__training_losses = []
        self.__val_losses = []
        self.__best_model = None  # Save your best model in this field and use this in test method.

        # Init Model
        self.__model = get_model(config_data, self.__vocab)

        for p in self.__model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        self.__criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
        self.__optimizer = torch.optim.Adam(self.__model.parameters(), lr=self.__config_data['experiment']['learning_rate'])

        self.__init_model()

        # Load Experiment Data if available
        self.__load_experiment()

    # Loads the experiment data if exists to resume training from last saved checkpoint.
    def __load_experiment(self):
        os.makedirs(ROOT_STATS_DIR, exist_ok=True)

        if os.path.exists(self.__experiment_dir):
            self.__training_losses = read_file_in_dir(self.__experiment_dir, 'training_losses.txt')
            self.__val_losses = read_file_in_dir(self.__experiment_dir, 'val_losses.txt')
            self.__current_epoch = len(self.__training_losses)

            state_dict = torch.load(os.path.join(self.__experiment_dir, 'latest_model.pt'))
            self.__model.load_state_dict(state_dict['model'])
            self.__optimizer.load_state_dict(state_dict['optimizer'])

        else:
            os.makedirs(self.__experiment_dir)

    def __init_model(self):
        if torch.cuda.is_available():
            self.__model = self.__model.cuda().float()
            self.__criterion = self.__criterion.cuda()

    # Main method to run your experiment. Should be self-explanatory.
    def run(self):
        start_epoch = self.__current_epoch
        for epoch in range(start_epoch, self.__epochs):  # loop over the dataset multiple times
            start_time = datetime.now()
            self.__current_epoch = epoch
            train_loss = self.__train()
            val_loss = self.__val()
            self.__record_stats(train_loss, val_loss)
            self.__log_epoch_stats(start_time)
            self.__save_model()

    def convert_question(self, prediction):
        """
        Converts predicted question indices to word tokens
        prediction: N x Q
        """
        word_idxs = prediction.cpu().numpy()
        captions = []
        for i in range(prediction.shape[0]):
            words = [self.__vocab.idx2word[idx].lower() for idx in word_idxs[i]]
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

    def __train(self):
        self.__model.train()
        training_loss = 0

        for i, (passages, answers, questions) in enumerate(self.__train_loader):

            if torch.cuda.is_available:
                passages = passages.cuda().long()
                answers = answers.cuda().long()
                questions = questions.cuda().long()

            out_seq = self.__model(passages, answers, questions) # N x Q x vocab_size

            self.__optimizer.zero_grad()
            if self.__config_data['model']['model_type'] == 'v_transformer':
                # Since the length is max_len - 1... Need to fix TODO
                loss = self.__criterion(out_seq[0].permute(0, 2, 1), out_seq[1])
            else:
                loss = self.__criterion(out_seq.permute(0, 2, 1), questions)
            loss.backward()
            self.__optimizer.step()

            batch_loss = loss.sum().item() / questions.shape[1]
            training_loss += batch_loss

            if i % 100 == 0:
                print("Batch {} Loss: {}".format(i, batch_loss))

        training_loss /= len(self.__train_loader)

        return training_loss

    def __val(self):
        self.__model.eval()
        val_loss = 0

        with torch.no_grad():
            for i, (passages, answers, questions) in enumerate(self.__val_loader):
                
                if torch.cuda.is_available:
                    passages = passages.cuda().long()
                    answers = answers.cuda().long()
                    questions = questions.cuda().long()
                
                out_seq = self.__model(passages, answers, questions)
                # loss = self.__criterion(out_seq.permute(0, 2, 1), questions)
                if self.__config_data['model']['model_type'] == 'v_transformer':
                    # Since the length is max_len - 1... Need to fix TODO
                    loss = self.__criterion(out_seq[0].permute(0, 2, 1), out_seq[1])
                else:
                    loss = self.__criterion(out_seq.permute(0, 2, 1), questions)
                batch_loss = loss.sum().item() / questions.shape[1]
                val_loss += batch_loss

            val_loss /= len(self.__val_loader)

            if len(self.__val_losses) == 0:
                self.__best_model = self.__model.state_dict()
                torch.save(self.__model.state_dict(), os.path.join(self.__experiment_dir, 'best_model.pth'))
            elif val_loss < min(self.__val_losses):
                self.__best_model = self.__model.state_dict()
                torch.save(self.__model.state_dict(), os.path.join(self.__experiment_dir, 'best_model.pth'))
        
        return val_loss

    def test(self):
        self.__model.eval()
        test_loss = 0
        
        meteor_score = 0
        rougeL_score = 0
        bleu1_score = 0
        bleu4_score = 0

        model = get_model(self.__config_data, self.__vocab)
        model.load_state_dict(torch.load(os.path.join(self.__experiment_dir, 'best_model.pth')))
        model.temperature = self.__config_data['generation']['temperature']
        model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        model.eval()

        meteor = load_metric("meteor")
        rouge = load_metric("rouge")
        bleu = load_metric("bleu")

        with torch.no_grad():
            for i, (passages, answers, questions) in enumerate(self.__test_loader):
                if torch.cuda.is_available:
                    passages = passages.cuda().long()
                    answers = answers.cuda().long()
                    questions =questions.cuda().long()

                out_seq = model(passages, answers, questions) # N x Q

                if self.__config_data['model']['model_type'] == 'v_transformer':
                    # Since the length is max_len - 1... Need to fix TODO
                    loss = self.__criterion(out_seq[0].permute(0, 2, 1), out_seq[1])
                else:
                    loss = self.__criterion(out_seq.permute(0, 2, 1), questions)

                batch_loss = loss.sum().item() / questions.shape[1]
                test_loss += batch_loss

                # Metric Evaluation
                predictions = model.predict(passages, answers) # N x Q
                predictions = self.convert_question(predictions) # list of lists of tokens
                true_questions = self.convert_question(questions) # list of lists of tokens
                print (predictions)

                bleu_list = [[elem] for elem in true_questions]
                bleu1_score += bleu.compute(predictions=predictions, references=bleu_list, max_order=1)['bleu']
                bleu4_score += bleu.compute(predictions=predictions, references= bleu_list, max_order=4)['bleu']

                predicted_strings = [' '.join(elem) for elem in predictions]
                true_strings = [' '.join(elem) for elem in true_questions]
                meteor_score += meteor.compute(predictions=predicted_strings, references=true_strings)['meteor']
                rougeL_score += rouge.compute(predictions=predicted_strings, references=true_strings)['rougeL'].mid.fmeasure

            test_loss /= len(self.__test_loader)
            perp = np.exp(test_loss)

            # Normalize metric scores
            bleu1_score /= len(self.__test_loader)
            bleu4_score /= len(self.__test_loader)
            meteor_score /= len(self.__test_loader)
            rougeL_score /= len(self.__test_loader)

        result_str = "Test Performance: Loss: {}, Perplexity: {}, Bleu1: {}, Bleu4: {}, Meteor: {}, Rouge-L: {}".format(
                                                                                            test_loss,
                                                                                            perp,
                                                                                            bleu1_score,
                                                                                            bleu4_score,
                                                                                            meteor_score,
                                                                                            rougeL_score)
        self.__log(result_str)

        dic = {'Test Loss': test_loss, 'Perplexity': perp, 'BLEU1': bleu1_score, 'BLEU4': bleu4_score}
        with open(os.path.join(self.__experiment_dir, 'results.json'), 'w') as f:
            json.dump(dic, f)

        return test_loss, bleu1_score, bleu4_score

    def __save_model(self):
        root_model_path = os.path.join(self.__experiment_dir, 'latest_model.pt')
        model_dict = self.__model.state_dict()
        state_dict = {'model': model_dict, 'optimizer': self.__optimizer.state_dict()}
        torch.save(state_dict, root_model_path)

    def __record_stats(self, train_loss, val_loss):
        self.__training_losses.append(train_loss)
        self.__val_losses.append(val_loss)

        self.plot_stats()

        write_to_file_in_dir(self.__experiment_dir, 'training_losses.txt', self.__training_losses)
        write_to_file_in_dir(self.__experiment_dir, 'val_losses.txt', self.__val_losses)

    def __log(self, log_str, file_name=None):
        print(log_str)
        log_to_file_in_dir(self.__experiment_dir, 'all.log', log_str)
        if file_name is not None:
            log_to_file_in_dir(self.__experiment_dir, file_name, log_str)

    def __log_epoch_stats(self, start_time):
        time_elapsed = datetime.now() - start_time
        time_to_completion = time_elapsed * (self.__epochs - self.__current_epoch - 1)
        train_loss = self.__training_losses[self.__current_epoch]
        val_loss = self.__val_losses[self.__current_epoch]
        summary_str = "Epoch: {}, Train Loss: {}, Val Loss: {}, Took {}, ETA: {}\n"
        summary_str = summary_str.format(self.__current_epoch + 1, train_loss, val_loss, str(time_elapsed),
                                         str(time_to_completion))
        self.__log(summary_str, 'epoch.log')

    def plot_stats(self):
        e = len(self.__training_losses)
        x_axis = np.arange(1, e + 1, 1)
        plt.figure()
        plt.plot(x_axis, self.__training_losses, label="Training Loss")
        plt.plot(x_axis, self.__val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.legend(loc='best')
        plt.title(self.__name + " Stats Plot")
        plt.savefig(os.path.join(self.__experiment_dir, "stat_plot.png"))
        plt.show()
