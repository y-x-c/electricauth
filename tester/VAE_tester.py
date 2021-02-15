import torch
from torch import nn
import numpy as np
import pickle
import os

from model.VAE import VAE
from data.normalization import ChallengeNormalizer
from util.utils import iterate_minibatches
from util.cache import load_model, save_model
from config.general import cache_dir

class VAETester(object):
    
    def __init__(self, X_train, y_train, X_val=None, y_val=None, gpu=None,
            lr=1e-3, batchsize_train=2, batchsize_test=200,
            epochs=50, verbose=False,
            cache_name=None, load_trained=False, no_train=False,
            n_latent_features=20,
            n_hidden=400,
            no_variational=False,
        ):
        normalizer = ChallengeNormalizer(X_train, y_train)
        self.normalizer = normalizer

        net = VAE(input_length=X_train.shape[2],
                n_sensor_channel=X_train.shape[1],
                n_latent_features=n_latent_features,
                n_hidden=n_hidden,
                no_variational=no_variational)
        self.net = net

        self.batchsize_test = batchsize_test

        self.use_gpu = not gpu is None

        if self.use_gpu:
            torch.cuda.set_device(gpu)
            net = net.cuda()

        model_state_dict = load_model(cache_name, gpu) if load_trained else None
        if model_state_dict:
            self.net.load_state_dict(model_state_dict)
        elif not no_train:
            self.train(X_train, y_train, X_val, y_val, epochs=epochs, lr=lr,
                batchsize_train=batchsize_train, batchsize_test=batchsize_test,
                verbose=verbose)
            
            if cache_name:
                save_model(cache_name, net)
                cache_path = os.path.join(cache_dir, cache_name + '.log')
                with open(cache_path, 'wb') as f:
                    pickle.dump(dict(
                        train_loss=self.train_loss_list,
                        val_loss=self.val_loss_list,
                        lr=lr,
                        batchsize_train=batchsize_train,
                        epochs=epochs
                    ), f)
        
            if X_val is not None:
                print('@last epoch, train loss {:4f}, val loss {:4f}'.format(
                    self.train_loss_list[-1], self.val_loss_list[-1]))
            else:
                print('@last epoch, train loss {:4f}'.format(
                    self.train_loss_list[-1]))
        else:
            raise RuntimeError('No trained model available and no_train=True')

    def train(self, X_train, y_train, X_val, y_val, epochs, lr, batchsize_train,
            batchsize_test, verbose):

        normalizer = self.normalizer
        X_train = normalizer.normalize(X_train, y_train)
        if X_val is not None:
            X_val = normalizer.normalize(X_val, y_val)

        net = self.net

        optimizer = torch.optim.Adam(net.parameters(), lr=lr)

        criterion = nn.MSELoss()

        self.train_loss_list = train_loss_list = []
        self.val_loss_list = val_loss_list = []

        for e in range(epochs):
            net.train()

            train_loss_epoch = 0.
            val_loss_epoch = 0.
            for batch in iterate_minibatches(inputs=X_train, targets=None,
                    batchsize=batchsize_train):
                x = batch

                inputs = torch.from_numpy(x)
                if self.use_gpu:
                    inputs = inputs.cuda()
                
                optimizer.zero_grad()   
                
                inputs = inputs.view(inputs.shape[0], -1)
                outputs, mu, logvar = net(inputs)

                loss = criterion(outputs, inputs)

                train_loss_epoch += loss.item() * x.shape[0]

                loss.backward()
                optimizer.step()
            
            train_loss_epoch /= X_train.shape[0]
            train_loss_list.append(train_loss_epoch)

            if X_val is not None:
                net.eval()

                with torch.no_grad():
                    for batch in iterate_minibatches(inputs=X_val,
                            targets=None, batchsize=batchsize_test):
                        x = batch     

                        inputs = torch.from_numpy(x)
                        if self.use_gpu:
                            inputs = inputs.cuda()
                            
                        inputs = inputs.view(inputs.shape[0], -1)
                        outputs, mu, logvar = net(inputs)

                        loss = criterion(outputs, inputs)

                        val_loss_epoch += loss.item() * x.shape[0]
                    
                    val_loss_epoch /= X_val.shape[0]
                    val_loss_list.append(val_loss_epoch)

                net.train()

            if verbose:
                print("Epoch: {}/{}...".format(e+1, epochs),
                "Train Loss: {:.4f}...".format(train_loss_epoch),
                "Val Loss: {:.4f}...".format(val_loss_epoch))

    def test(self, X_test, y_test):
        X_test = self.normalizer.normalize(X_test, y_test)

        net = self.net

        criterion = nn.MSELoss(reduction='none')

        score_list = np.array([])
        mu_list = None
        logvar_list = None
        reconstructed_list = None

        net.eval()
        with torch.no_grad():
            for batch in iterate_minibatches(X_test, targets=None,
                    batchsize=self.batchsize_test, shuffle=False):
                x = batch

                inputs = torch.from_numpy(x)
                if self.use_gpu:
                    inputs = inputs.cuda()
                    
                inputs = inputs.view(inputs.shape[0], -1)
                outputs, mu, logvar = net(inputs)

                loss = criterion(outputs, inputs)
                loss = torch.mean(loss, dim=(1))
                loss = loss.cpu().numpy()

                score_list = np.concatenate([score_list, loss])

                mu = mu.detach().cpu().numpy()
                logvar = logvar.detach().cpu().numpy()
                outputs = outputs.detach().cpu().numpy()
                mu_list = np.concatenate([mu_list, mu]) \
                    if mu_list is not None else mu
                logvar_list = np.concatenate([logvar_list, logvar]) \
                    if logvar_list is not None else logvar
                reconstructed_list = \
                    np.concatenate([reconstructed_list, outputs]) \
                    if reconstructed_list is not None else outputs
        
        reconstructed_list = reconstructed_list.reshape(X_test.shape)
        reconstructed_list = self.normalizer.denormalize(
            reconstructed_list, y_test)

        return score_list, mu_list, logvar_list, reconstructed_list
