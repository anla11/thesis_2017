from model import Net
from train_test import train, test

train(filename = '../dataset/train.txt', model_name = 'model/rnn_ctc', epochs = 45, checkpoint = 1, old_epochs = 0)
test('model/rnn_ctc.pth', filename = '../dataset/test.txt', idx_from = 4000, idx_to = -1, logfile = 'log/logfile_test.txt')

'''
cd /mnt/e/Workspace/pytorch/src
~/anaconda3/bin/python3.5 main.py;
'''
