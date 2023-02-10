from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
# from model import TransformerEncoder, CorruptionLayer
from model_cnn_transformer import TransformerEncoder, CorruptionLayer
# from standard_model import TransformerEncoder, CorruptionLayer
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns

class Trainer:
    def __init__(self, args, train_loader, test_loader):
        self.args = args
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = 'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'
        self.model = TransformerEncoder(sentence_len=args.sample_per_input,
                                        d_feature=args.sample_len,
                                        n_layers=args.n_layers,
                                        n_heads=args.n_attn_heads,
                                        p_drop=args.dropout,
                                        d_ff=args.ffn_hidden,
                                        )
        self.corruption = CorruptionLayer(self.device, args.corrupt_probability)
        self.model.to(self.device)
        self.corruption.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), args.lr)
        self.criterion_generation = nn.MSELoss()
        self.criterion_classification =  nn.CrossEntropyLoss()
        self.generation_weight = args.generation_weight

    def pretrain(self, epoch):
        # generation task only
        losses = 0
        n_batches, n_samples = len(self.train_loader), len(self.train_loader.dataset)
        print(self.model)

        self.model.train()
        for i, batch in enumerate(self.train_loader):
            inputs, labels = map(lambda x: x.to(self.device), batch)
            inputs_copy = inputs.clone()
            corrupted_inputs = self.corruption(inputs_copy)
            outputs_feature_corrupted, _ ,_ = self.model(corrupted_inputs)
            loss = self.criterion_generation(outputs_feature_corrupted, inputs)
            losses += loss.item()
            self.optimizer.zero_grad()

            loss.backward()
            self.optimizer.step()
        print('Pretrain Epoch: {}'.format(epoch))
        print('Generation:\tLoss: {:.4f}'.format(losses / n_batches))

    def train(self, epoch):
        losses, accs = 0, 0
        losses_generation = 0
        losses_classification = 0
        n_batches, n_samples = len(self.train_loader), len(self.train_loader.dataset)
        self.model.train()
        for i, batch in enumerate(self.train_loader):
            inputs, labels = map(lambda x: x.to(self.device), batch)

            inputs_copy = inputs.clone()
            corrupted_inputs = self.corruption(inputs_copy)
            _, outputs_classification, _= self.model(inputs)
            outputs_feature_corrupted, _, _ = self.model(corrupted_inputs)
            loss_classification = self.criterion_classification(outputs_classification, labels) * (1 - self.generation_weight)
            loss = loss_classification

            losses_classification += loss_classification
            losses += loss.item()
            acc = (outputs_classification.argmax(dim=-1) == labels).sum()
            accs += acc
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


        print('Train Epoch: {}'.format(epoch))
        print('Classification:\tLoss: {:.4f}\tAcc: {:.1f}%'.format(losses_classification / n_batches,
                                                                   accs / n_samples * 100.))
        print('Generation:\tLoss: {:.4f}'.format(losses_generation / n_batches))

        return accs / n_samples * 100., losses_classification / n_batches

    def validate(self, epoch):
        negative = None
        neutral = None
        positive = None
        Att_pin = None
        losses, accs = 0, 0
        n_batches, n_samples = len(self.test_loader), len(self.test_loader.dataset)
        losses_generation = 0
        losses_classification = 0
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(self.test_loader):
                inputs, labels = map(lambda x: x.to(self.device), batch)
                inputs_copy = inputs.clone()
                corrupted_inputs = self.corruption(inputs_copy)
                _, outputs_classification, _ = self.model(inputs)
                outputs_feature_corrupted, _ , att_pin= self.model(corrupted_inputs)

                if Att_pin == None:
                    Att_pin = att_pin
                else:
                    Att_pin = torch.vstack((Att_pin, att_pin))

                loss_classification = self.criterion_classification(outputs_classification, labels) * (1 - self.generation_weight)
                losses_classification += loss_classification
                loss =  loss_classification
                losses += loss.item()
                acc = (outputs_classification.argmax(dim=-1) == labels).sum()
                print('labels',labels)
                print('outputs_classification', outputs_classification.argmax(dim=-1))
                accs += acc

            print('Validate Epoch: {}'.format(epoch))
            print(('Classification:\tLoss: {:.4f}\tAcc: {:.1f}%').format(losses_classification / n_batches,
                                                                       accs / n_samples * 100.))

        return accs / n_samples * 100. , att_pin

    def save(self, epoch, model_prefix='model', root='.model'):
        path = Path(root) / (model_prefix + '.ep%d' % epoch)
        if not path.parent.exists():
            path.parent.mkdir()
        torch.save(self.model, path)

    def save_model(self):
        torch.save(self.model, 'mymodel.pth')