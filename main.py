import argparse
from torch.utils.data import DataLoader
import torch
from data_utils import set_seeds, load_mat, kfold_train_test_split_SEED_subject1session1,kfold_train_test_split_SEEDIV_subject1session1
from standard_trainer import Trainer
from sklearn.model_selection import KFold
from torch.utils.data import TensorDataset
import numpy as np


SUB_ACC=[]
SUB_STD=[]

def main(args):
    print(args)
    set_seeds()

    # k折交叉检验

    dataset_dir = args.dataset_dir
    features, labels , cumulative_arr = load_mat(dataset_dir)
    print('labels',labels)
    # data, label = kfold_train_test_split_SEED_subject1session1(features, labels, cumulative_arr, normalization=False) # for SEED
    data, label = kfold_train_test_split_SEEDIV_subject1session1(features, labels, cumulative_arr, normalization=False) # for SEED-IV

    ks = 10
    Acc = 0
    Acc1 = 0
    Acc_std=[]

    kf = KFold(n_splits=ks, random_state=2666, shuffle=True)
    for train_index, test_index in kf.split(data):
        best_acc = 0
        print('train_index', train_index, 'test_index', test_index)

        train_arr = data[train_index]
        train_label = label[train_index]
        test_arr = data[test_index]
        test_label = label[test_index]

        train_tensor = torch.tensor(train_arr, dtype=torch.float)
        train_label = torch.tensor(train_label, dtype=torch.int64)
        test_tensor = torch.tensor(test_arr, dtype=torch.float)
        test_label = torch.tensor(test_label, dtype=torch.int64)

        train_dataset = TensorDataset(train_tensor, train_label)
        test_dataset = TensorDataset(test_tensor, test_label)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

        # Build Trainer
        trainer = Trainer(args, train_loader, test_loader)

        # # Warm up
        for epoch in range(1, args.pretrain + 1):
            trainer.pretrain(epoch)

        # # Train
        for epoch in range(1, args.epochs + 1):
            print('######', epoch)
            acc, loss = trainer.train(epoch)
            if acc >= best_acc:
                best_acc = acc
                trainer.save_model()

            # Test
            model = torch.load('mymodel.pth')
            model.eval()
            test_acc, ppp = trainer.validate(epoch)
            Acc += test_acc.cpu().numpy()
            print(type(Acc))

        # Test
        model = torch.load('mymodel.pth')
        model.eval()
        test_acc, pin = trainer.validate(1)
        Acc_std.append(test_acc.cpu().numpy())
        Acc1 += test_acc.cpu().numpy()
        print('test_acc = {:.1f}%'.format(Acc1.mean()))

    Acc_std = np.array(Acc_std)
    print('After {}-fold cross validation, test_acc = {:.1f}%'.format(ks, Acc1 / ks))  # k-fold交叉验证
    SUB_ACC.append(Acc1 / ks)
    SUB_STD.append(Acc_std.std())
    print('acc of each subject',SUB_ACC)
    print('std of each subject', SUB_STD)

if __name__ == '__main__':
  for session in range(3):
     for sub in range(15):

        parser = argparse.ArgumentParser()

        parser.add_argument('--dataset_dir', type=str,
                    default='...\subject' + str(sub + 1) + 'session' + str(session + 1),
                    help='path to the folder that contains feature.npy, label.npy, cumulative.npy')  # SEED数据集


        parser.add_argument('--train_percentage', type=int,
                            default=0.6,
                            help='portion of training samples for each subject')

        parser.add_argument('--normalization', type=bool,
                            default=True,
                            help='do normalization for train/test dataset by the rule learned from train dataset')
        parser.add_argument('--sample_per_input', type=int,
                            default=1,
                            help='number of samples for each sample, or length of the sentence')
        parser.add_argument('--sample_len', type=int,
                            default=128,
                            help='length of each sample')
        parser.add_argument('--corrupt_probability', type=float,
                            default=0.05,
                            help='the probability that a cell is corrupted in the generation task')
        parser.add_argument('--generation_weight', type=float,
                            default=0.1,
                            help='a constant that determines the importance of generation task in the multi-task training. total cost = generation_cost * generation_weight + classification_cost * (1 - generation_weight)')
        parser.add_argument('--output_model_prefix', type=str,
                            default='model_transformer',
                            help='output model name prefix')
        # Input parameters

        parser.add_argument('--batch_size', default=32, type=int, help='batch size')
        # Train parameters
        parser.add_argument('--pretrain', default=0, type=int, help='pretrain epochs, generation task only')
        parser.add_argument('--epochs', default=30, type=int, help='the number of epochs')
        parser.add_argument('--lr', default=1e-5, type=float, help='learning rate')

        parser.add_argument('--no_cuda', action='store_true')
        # Model parameters
        parser.add_argument('--n_layers', default=3, type=int,
                            help='the number of heads in the multi-head attention network')
        parser.add_argument('--n_attn_heads', default=6, type=int, help='the number of multi-head attention heads')
        parser.add_argument('--dropout', default=0.7, type=float, help='the residual dropout value')
        # parser.add_argument('--dropout',        default=0.1,  type=float, help='the residual dropout value')
        parser.add_argument('--ffn_hidden', default=256, type=int,
                            help='the dimension of the feedforward network')

        args = parser.parse_args()

        main(args)
