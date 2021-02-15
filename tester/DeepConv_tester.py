import torch
from torch import nn
import numpy as np
import pickle
import os

from model.DeepConv import DeepConv
from util.utils import iterate_minibatches
from util.cache import load_model, save_model
from config.general import cache_dir

class DeepConvTester(object):

    def __init__(self, X_train, y_train, X_val=None, y_val=None, gpu=None,
            lr=3e-3, batchsize_train=100, batchsize_test=200,
            epochs=50, verbose=False,
            cache_name=None, load_trained=False, no_train=False,
            weighted_loss=False,
    ):

        self.net = DeepConv(input_length=X_train.shape[2],
            n_sensor_channel=X_train.shape[1], n_classes=len(set(y_train)))

        self.batchsize_test = batchsize_test
        
        self.use_gpu = not gpu is None

        self.weighted_loss = weighted_loss

        if self.use_gpu:
            torch.cuda.set_device(gpu)
            self.net = self.net.cuda()

        model_state_dict = load_model(cache_name, gpu) if load_trained else None
        if model_state_dict:
            self.net.load_state_dict(model_state_dict)
        elif not no_train:
            self.train(X_train, y_train, X_val, y_val, epochs=epochs, lr=lr,
                batchsize_train=batchsize_train, batchsize_test=batchsize_test,
                verbose=verbose
            )

            if cache_name:
                save_model(cache_name, self.net)
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

    def init_weights(self, m):
        if type(m) == nn.Conv1d or type(m) == nn.Linear:
            torch.nn.init.orthogonal_(m.weight)
            m.bias.data.fill_(0)
            
    def train(self, X_train, y_train, X_val, y_val, epochs, lr,
            batchsize_train, batchsize_test, verbose
    ):
        net = self.net
        net.apply(self.init_weights)

        optimizer = torch.optim.SGD(
            net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-3
        )

        if self.weighted_loss:
            weight = [len(np.where(y_train == i)[0]) for i in set(y_train)]
            weight = torch.Tensor([1./w for w in weight]).cuda()
            criterion = nn.CrossEntropyLoss(weight=weight)
        else:
            criterion = nn.CrossEntropyLoss()
    
        self.train_loss_list = train_loss_list = []
        self.val_loss_list = val_loss_list = []

        for e in range(epochs):
            net.train()

            train_loss_epoch = 0.
            val_loss_epoch = 0.
            for batch in iterate_minibatches(inputs=X_train, targets=y_train,
                    batchsize=batchsize_train):
                
                inputs, targets = batch
                inputs = torch.from_numpy(inputs)
                targets = torch.from_numpy(targets)
                if self.use_gpu:
                    inputs = inputs.cuda()
                    targets = targets.cuda()
                
                optimizer.zero_grad()
                
                outputs = net(inputs)

                loss = criterion(outputs, targets.long())

                train_loss_epoch += loss.item() * inputs.shape[0]

                loss.backward()
                optimizer.step()
            
            train_loss_epoch /= X_train.shape[0]
            train_loss_list.append(train_loss_epoch)

            if X_val is not None:
                net.eval()

                with torch.no_grad():
                    for batch in iterate_minibatches(inputs=X_val,
                            targets=y_val, batchsize=batchsize_test,
                            shuffle=False
                    ):
                        inputs, targets = batch

                        inputs = torch.from_numpy(inputs)
                        targets = torch.from_numpy(targets)
                        if self.use_gpu:
                            inputs = inputs.cuda()
                            targets = targets.cuda()
                            
                        outputs = net(inputs)

                        loss = criterion(outputs, targets.long())

                        val_loss_epoch += loss.item() * inputs.shape[0]
                    
                    val_loss_epoch /= X_val.shape[0]
                    val_loss_list.append(val_loss_epoch)

                net.train()

            if verbose:
                print("Epoch: {}/{}...".format(e+1, epochs),
                "Train Loss: {:.4f}...".format(train_loss_epoch),
                "Val Loss: {:.4f}...".format(val_loss_epoch))

    def test(self, X_test):
        net = self.net

        softmax = nn.Softmax(dim=1)

        y_pred, probs = [], []

        net.eval()
        with torch.no_grad():
            for batch in iterate_minibatches(X_test, None, batchsize=self.batchsize_test, shuffle=False):
                inputs = batch

                inputs = torch.from_numpy(inputs)
                if self.use_gpu:
                    inputs = inputs.cuda()
                    
                output = net(inputs)
                prob = softmax(output)
                top_p, top_class = prob.topk(1, dim=1)
                
                probs += top_p.tolist()
                y_pred += top_class.tolist()

        y_pred = [p[0] for p in y_pred]
        probs = [p[0] for p in probs]

        y_pred, probs = np.array(y_pred), np.array(probs)
        return y_pred, probs