import os
import time
from os.path import basename
import random
import numpy as np
import torch
from warpctc_pytorch import CTCLoss

from model import *
from read_data import read_data, read_dict, read_listimg
from decoder import ArgMaxDecoder
from minibatch_process import sort_minibatch, padding_listseq

''' Read dict'''
char_dataset, lookup, decode_lookup = read_dict()

''' Model parameters '''
width_frame = 2
height_frame = 40
nIn = width_frame * height_frame
nHidden = 140
nOut = len(char_dataset) + 1
nLayer = 1

''' Init '''
criterion, state0, net, optimizer, decoder = None, None, None, None, None
def init(model_name, old_epochs):
	global criterion
	criterion = CTCLoss()

	h0 = Variable(torch.rand(nLayer, batch_size, nHidden))
	c0 = Variable(torch.rand(nLayer, batch_size, nHidden))
	global state0
	state0 = (h0, c0)

	global net
	if old_epochs == 0:
	    print ('Create model ', model_name)
	    net = Net(nIn, nHidden, nLayer, nOut)
	else:
	    old_modelname = '%s_%02d.pth' % (model_name, old_epochs)
	    print ('Load ', old_modelname)
	    net = torch.load(old_modelname)

	global optimizer
	optimizer = torch.optim.Adadelta(net.parameters())
	# optimizer = torch.optim.SGD(net.parameters(), lr=0.0001,momentum=0.99)
	global decoder
	decoder = ArgMaxDecoder(char_dataset, decode_lookup)

''' Testing '''
def testing(net, image_list, title_list, list_index, logfile = None):
	batch_size = 1
	h0 = Variable(torch.rand(nLayer, batch_size, nHidden))
	c0 = Variable(torch.rand(nLayer, batch_size, nHidden))
	global state0
	state0 = (h0, c0)

	total_cer, total_wer = 0, 0
	net.eval()
	for start_idx in range(0, len(list_index)-batch_size+1, batch_size):
		#load image
		imgs, labels = read_listimg(lookup, char_dataset, image_list, title_list, np.array(list_index)[range(start_idx,start_idx + batch_size)].tolist())
		imgs, labels = sort_minibatch(imgs, labels)
		imgs, img_len, img_maxlen = padding_listseq(imgs, nIn)
		labels, label_len, label_maxlen = padding_listseq(labels, 1)        

		#preprocess input/target
		input_decode = Variable(torch.Tensor(imgs))
		input_decode = input_decode.view(int(img_maxlen/nIn), batch_size, nIn)
		target_decode = Variable(torch.IntTensor(labels))
		
		split_targets = [labels[i][:label_len[i]] for i in range(batch_size)]

		output_decode = net.forward(input_decode, state0)
		output_len = Variable(torch.IntTensor([int((l-1) / nIn + 1) for l in img_len]))
		decoded_output = decoder.decode(output_decode.data, output_len)
		target_strings = decoder.process_strings(decoder.convert_to_strings(split_targets))
	   
		wer, cer = 0.0, 0.0
		wers, cers = [], []
		for x in range(len(target_strings)):
			x_wer = decoder.wer(decoded_output[x], target_strings[x]) / float(len(target_strings[x].split()))
			x_cer = decoder.cer(decoded_output[x], target_strings[x]) / float(len(target_strings[x]))
			wers = wers + [x_wer]
			cers = cers + [x_cer]
			wer += x_wer 
			cer += x_cer 
			# print (wers, cers)
			if logfile is not None:
				file = open(logfile,'a') 
				file.write('%s: ' % basename(title_list[list_index[start_idx + x]])[:-4])
				file.write('WER = %.4f %% | CER = %.4f %% ' % (x_wer * 100, x_cer * 100))
				file.write('Predict = <%s> | Truth = <%s>\n' % (decoded_output[x], target_strings[x]))
		total_cer += cer
		total_wer += wer

	wer = total_wer * 100 / len(list_index) 
	cer = total_cer * 100 / len(list_index) 
	if logfile is not None:
		file.write('AVG WER = %.6f %% | AVG CER = %.6f %% ' % (wer, cer))
		file.close() 
	print ('AVG WER = %.6f %% | AVG CER = %.6f %%' % (wer, cer))

	return wer, cer

''' Training ''' 
def training(model_name, epochs, checkpoint, old_epochs, image_list, title_list, list_index_train, list_index_dev):
	max_datasize = len(list_index_train)
	for epoch in range(1,1+epochs):
	    start = time.time()
	    loss_epochs = 0.0
	    random.shuffle(list_index_train)
	    for start_idx in range(0, max_datasize-batch_size+1, batch_size):
	        #load image
	        imgs, labels = read_listimg(lookup, char_dataset, image_list, title_list, np.array(list_index_train)[range(start_idx,start_idx + batch_size)].tolist())
	        imgs, labels = sort_minibatch(imgs, labels)
	        imgs, img_len, img_maxlen = padding_listseq(imgs, nIn)
	        
	        target = []
	        for label in labels:
	            target = target + label
	        label_len = [len(labels[i]) for i in range(len(labels))]
	    
	        #preprocess input/target
	        input = Variable(torch.Tensor(imgs))
	        input = input.view(int(img_maxlen/nIn), batch_size, nIn)
	        target = Variable(torch.IntTensor(target))

	        #forward
	        output = net.forward(input, state0)

	        #cal loss 
	        target_len = Variable(torch.IntTensor(label_len))
	        output_len = Variable(torch.IntTensor([int((l-1) / nIn + 1) for l in img_len]))

	        loss = criterion(output, target, output_len, target_len)
	        loss_epochs += loss / max_datasize #/ ((max_datasize - 1) / batch_size + 1)
	        loss = loss/batch_size

	        # gradient descent
	        optimizer.zero_grad()
	        loss.backward()
	        optimizer.step()
	    if (epoch == 1) or (epoch % checkpoint == 0):
	        model_name_epoch = '%s_%02d.pth' % (model_name, old_epochs + epoch)
	        print ('Save model as %s' % model_name_epoch)
	        torch.save(net, model_name_epoch)        
	    end = time.time()
	    
	    print ('Epoch: %d' % epoch)
	    print ('   + Train (ctc-loss) ', loss_epochs.data) 
	    wer, cer = testing(net, image_list, title_list, list_index_dev)
	    print ('   + Dev (wer, cer): ', wer, cer) 
	    print ('   + Elapsed time: ', end - start)


def train(filename, model_name, epochs, checkpoint, old_epochs):
	''' Read data '''
	image_list, title_list = read_data(filename = filename, limit = -1)
	list_index_train = list(range(len(image_list)))[:4000]
	list_index_dev = list(range(len(image_list)))[4000:]

	init(model_name, old_epochs)
	training(model_name, epochs, checkpoint, old_epochs, image_list, title_list, list_index_train, list_index_dev)

def test(model_name, filename, idx_from, idx_to, logfile):
	print ('Read test data')
	image_test, title_test = read_data(filename = filename, limit = idx_to)
	list_index_test = list(range(len(image_test)))[idx_from:]

	net = torch.load(model_name)
	print ('Net: \n', net)
	global decoder
	decoder = ArgMaxDecoder(char_dataset, decode_lookup)

	wer, cer = testing(net, image_test, title_test, list_index_test, logfile = logfile)
	print (wer, cer)	
