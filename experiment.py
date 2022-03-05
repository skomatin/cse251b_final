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

from constants import ROOT_STATS_DIR
from dataset_factory import get_datasets
from model_factory import get_model
from file_utils import *

import warnings
import json

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torchvision.transforms as transforms

warnings.filterwarnings("ignore", category=UserWarning)
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
        self.__vocab, self._train_loader, self._val_loader, self._test_loader = get_datasets(config_data)

        # Setup Experiment
        self.__generation_config = config_data['generation']
        self.__epochs = config_data['experiment']['num_epochs']
        self.__current_epoch = 0
        self.__training_losses = []
        self.__val_losses = []
        self.__best_model = None  # Save your best model in this field and use this in test method.

        # Init Model
        self.__model = get_model(config_data, self.__vocab)

        # TODO: Set these Criterion and Optimizers Correctly
        self.__criterion = None
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

            # print("Epoch {} Train Loss: {} Val Loss: {}".format(epoch, train_loss, val_loss))

    def plot_image_captions(self, images, caption_length, kind='test', epoch=None):

        with torch.no_grad():
            image_dir = kind + '_images'
            if epoch is not None:
                image_dir += '_epoch{}'.format(epoch)
            out_dir = os.path.join(self.__experiment_dir, image_dir)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            images = images[:10] # 10 x 3 x 256 x 256
            
            model = get_model(self.__config_data, self.__vocab)
            if kind == 'test':
                model.load_state_dict(torch.load(os.path.join(self.__experiment_dir, 'best_model.pth')))
            else:
                model.load_state_dict(self.__model.state_dict())

            model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            model.eval()

            predictions = model.predict(images, caption_length)
            captions = self.generate_caption(predictions) # list(list(str))
            
            denormalize = transforms.Compose([
                transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1/0.229, 1/0.224, 1/0.225])
            ])
            images = denormalize(images)
            images = list(images.permute(0, 2, 3, 1).cpu().numpy())
            
            x = 0
            for img, caption in zip(images, captions):
                plt.imshow(img)
                plt.xlabel(' '.join(caption))
                plt.savefig(os.path.join(out_dir, 'img{}.png'.format(x)))
                x += 1

    def generate_caption(self, prediction):
        """
        Generates words from predicted one-hot encoded captions
        prediction: N x L
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
            clean_list = ['<pad>', '<start>', '<end>', '<unk>']
            cleaned_caption = [word for word in captions[i] if word not in clean_list]
            to_return.append(cleaned_caption)

        return to_return

    def __train(self):
        self.__model.train()
        training_loss = 0

        for i, (images, captions, _) in enumerate(self.__train_loader):
            # 64 x 3 x 256 x 256
            # 64 x 21
            # hidden_state = None
            if torch.cuda.is_available:
                images = images.cuda().float()
                captions = captions.cuda().float()

            imgs = self.__model.embed_image(images) # N x 1 x embedding_size
            caps = self.__model.embed_word(captions.long()) # N x 20 x embedding_size

            inp = torch.cat([imgs, caps[:, :-1, :]], axis=1)
            
            out, hidden_state = self.__model(inp) # N x L x vocab_size

            self.__optimizer.zero_grad()
            loss = self.__criterion(out.permute(0, 2, 1), captions.long())
            loss.backward()
            self.__optimizer.step()

            batch_loss = loss.sum().item() / captions.shape[1]
            training_loss += batch_loss

            if i % 100 == 0:
                print("Batch {} Loss: {}".format(i, batch_loss))

            if i == 0:
                self.plot_image_captions(images, captions.shape[1], kind='train', epoch=self.__current_epoch)

        training_loss /= len(self.__train_loader)

        return training_loss

    # TODO: Perform one Pass on the validation set and return loss value. You may also update your best model here.
    def __val(self):
        self.__model.eval()
        val_loss = 0

        with torch.no_grad():
            for i, (images, captions, _) in enumerate(self.__val_loader):
                
                if torch.cuda.is_available:
                    images = images.cuda().float()
                    captions = captions.cuda().float()
                
                imgs = self.__model.embed_image(images) # N x 1 x embedding_size
                caps = self.__model.embed_word(captions.long()) # N x 20 x embedding_size

                inp = torch.cat([imgs, caps[:, :-1, :]], axis=1) 
                out, hidden_state = self.__model(inp) # N x L x vocab_size

                loss = self.__criterion(out.permute(0, 2, 1), captions.long())

                batch_loss = loss.sum().item() / captions.shape[1]
                val_loss += batch_loss

                if i == 0:
                    self.plot_image_captions(images, captions.shape[1], kind='val', epoch=self.__current_epoch)

            val_loss /= len(self.__val_loader)

            if len(self.__val_losses) == 0:
                self.__best_model = self.__model.state_dict()
                torch.save(self.__model.state_dict(), os.path.join(self.__experiment_dir, 'best_model.pth'))
            elif val_loss < min(self.__val_losses):
                self.__best_model = self.__model.state_dict()
                torch.save(self.__model.state_dict(), os.path.join(self.__experiment_dir, 'best_model.pth'))
        
        return val_loss

    # TODO: Implement your test function here. Generate sample captions and evaluate loss and
    #  bleu scores using the best model. Use utility functions provided to you in caption_utils.
    #  Note than you'll need image_ids and COCO object in this case to fetch all captions to generate bleu scores.
    def test(self):
        self.__model.eval()
        test_loss = 0
        
        bleu1_score = 0
        bleu4_score = 0

        model = get_model(self.__config_data, self.__vocab)
        model.load_state_dict(torch.load(os.path.join(self.__experiment_dir, 'best_model.pth')))
        # model.load_state_dict(self.__best_model)
        model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        model.eval()

        with torch.no_grad():
            print("Test Loss and Caption Generation")
            for i, (images, captions, img_ids) in enumerate(self.__test_loader):
                if torch.cuda.is_available:
                    images = images.cuda().float()
                    captions = captions.cuda().float()

                imgs = self.__model.embed_image(images) # N x 1 x embedding_size
                caps = self.__model.embed_word(captions.long()) # N x 20 x embedding_size

                inp = torch.cat([imgs, caps[:, :-1, :]], axis=1)
                
                out, hidden_state = self.__model(inp) # N x L x vocab_size
                
                loss = self.__criterion(out.permute(0, 2, 1), captions.long())

                batch_loss = loss.sum().item() / captions.shape[1]
                test_loss += batch_loss

                # Caption Evaluation
                predictions = model.predict(images, captions.shape[1])
                pred_captions = self.generate_caption(predictions)
                
                temp_bleu1 = 0
                temp_bleu4 = 0
                for k in range(len(img_ids)):
                    pred_caption = pred_captions[k] # list(str)
                    ref_captions = [elem['caption'].lower() for elem in self.__coco_test.imgToAnns[img_ids[k]]]
                    ref_captions = [nltk.tokenize.word_tokenize(elem) for elem in ref_captions]

                    temp_bleu1 += bleu1(ref_captions, pred_caption)
                    temp_bleu4 += bleu4(ref_captions, pred_caption)

                bleu1_score += temp_bleu1 / len(img_ids)
                bleu4_score += temp_bleu4 / len(img_ids)

                if i == 0:
                    self.plot_image_captions(images, captions.shape[1], kind='test')

            test_loss /= len(self.__test_loader)
            perp = np.exp(test_loss)

            # Normalize BLEU scores
            bleu1_score /= len(self.__test_loader)
            bleu4_score /= len(self.__test_loader)
        
        result_str = "Test Performance: Loss: {}, Perplexity: {}, Bleu1: {}, Bleu4: {}".format(test_loss,
                                                                                               perp,
                                                                                               bleu1_score,
                                                                                               bleu4_score)
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
