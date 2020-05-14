from __future__ import print_function
import sys
import random
from decoder import *
from model import *
from read_data import read_data, read_dict, read_listimg

def convert(img): #convert image from 2d to list
	img = img.T.reshape((1,-1))[0].astype(float)
	img = img * 1.0/255
	return img.tolist()

def read_img(image_list):
	imgs_read = [cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2GRAY) for img in image_list]
	imgs_seq = [convert(imgs_read[i]) for i in range(len(imgs_read))]
	return imgs_seq

def read_title(title_list):
	titles = []
	i = 0
	for title in title_list:
		with open(title) as f:
			content = f.readlines()
		titles.append(content[0])
	return titles
	
def title2label(titles, lookup, char_dataset):
	labels = []
	for title in titles:
		label = [lookup[t] for t in title if t in char_dataset]
		labels.append(label)
	return labels

def read_data(filename = 'dataset/train.txt', limit = -1):
	with open(filename) as f:
		content = f.readlines()
	image_list = [item.split()[0] for item in content[:limit]]
	title_list = [item.split()[1] for item in content[:limit]]
	
	lookup = {}
	char_dataset = []
	with open('chardict.txt') as f:
		content = f.readlines()
		char_dataset = sorted([item[0] for item in content])
		label_dataset = sorted([int(item[2:]) for item in content])
		decode_lookup = dict(zip(label_dataset, char_dataset))
		lookup = dict(zip(char_dataset, label_dataset))

	imgs = read_img(image_list)
	labels = title2label(read_title(title_list), lookup, char_dataset)
	lookup['_'] = 0
	decode_lookup[0] = '_'
	return imgs, labels, char_dataset, lookup, decode_lookup    

def sort_listseq(listseq):
	seq_len = [len(listseq[j]) for j in range(len(imgs))]
	mydict = dict(zip(range(len(seq_len)), seq_len))																																																																																																																																																																																																																																									
	index = [w for w in sorted(mydict, key=mydict.get, reverse=True)]
	return np.array(listseq)[index].tolist()

def padding_listseq(listseq, align):
	tmp = len(listseq[0]) % align
	maxlen = len(listseq[0]) + align * (tmp != 0) - tmp
	seq_len = [len(seq) for seq in listseq]
	new_seq = [listseq[i] + [0] * (maxlen-seq_len[i]) for i in range(len(listseq))]            
	return new_seq, seq_len, maxlen

def padding_seq(seq, align):
	tmp = len(seq) % align
	maxlen = len(seq) + align * (tmp != 0) - tmp
	seq_len = len(seq)
	new_seq = seq + [0] * (maxlen-seq_len)
	return new_seq, seq_len, maxlen

class Net(nn.Module):
	def __init__(self, nIn, nHidden, nLayer, nOut):
		super(Net, self).__init__()
		self.rnn = nn.LSTM(input_size = nIn, hidden_size = nHidden, num_layers = nLayer)
		self.linear = nn.Linear(nHidden, nOut)
	
	def forward(self, input, state0): #state0 = (h0, c0)
		output, _ = self.rnn(input, state0)
		T, b, h = output.size()
		output = output.view(T * b, h)        
		output = self.linear(output)
		output = output.view(T, b, -1)
		return output        

def check(net, decoder, imgs_seq, labels, list_index, config):
	nIn, state0 = config
	total_cer, total_wer = 0, 0
	net.eval()
	for i in list_index:
	    input = Variable(torch.Tensor([imgs_seq[i]]))
	    input = input.view(int(input.size(1)/nIn), 1, nIn)
	    label = Variable(torch.IntTensor(labels[i]))
			
	    split_targets = [labels[i]]
			
	    output = net.forward(input, state0)

	    act_len = Variable(torch.IntTensor([output.size(0)]))
	    decoded_output = decoder.decode(output.data, act_len)
	    target_strings = decoder.process_strings(decoder.convert_to_strings(split_targets))
			
	    wer, cer = 0, 0
	    for x in range(len(target_strings)):
	        wer += decoder.wer(decoded_output[x], target_strings[x]) / float(len(target_strings[x].split()))
	        cer += decoder.cer(decoded_output[x], target_strings[x]) / float(len(target_strings[x]))
	    total_cer += cer
	    total_wer += wer

	wer = total_wer / len(list_index) 
	cer = total_cer / len(list_index)
	wer *= 100
	cer *= 100	
	return wer, cer

def test(model_name):
	nIn, nHidden, nLayer, nOut = 160, 2, 1, 117
	criterion = CTCLoss()
	decoder = ArgMaxDecoder(char_dataset, decode_lookup)

	h0 = Variable(torch.zeros(nLayer, batch_size, nHidden))
	c0 = Variable(torch.zeros(nLayer, batch_size, nHidden))
	state0 = (h0, c0)

	imgs_seq, labels, char_dataset, lookup, decode_lookup = read_data(filename = 'dataset/train.txt', limit = -1)
	list_index_dev = list(range(len(imgs_seq)))[:2]
	net = torch.load(model_name)
	check(net, imgs_seq, labels, list_index_dev, (nIn, state0))


def main(old_epochs=0):
	print ('Read data')
	imgs_seq, labels, char_dataset, lookup, decode_lookup = read_data(filename = 'dataset/train.txt', limit = -1)
	list_index_train = list(range(len(imgs_seq)))[:4000]
	list_index_dev = list(range(len(imgs_seq)))[4000:]
	SAVE_MODEL = True

	print ('Set up parameters')
	width_frame = 4
	height_frame = 40																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																											
	nIn = width_frame * height_frame
	nHidden = 140
	nOut = len(char_dataset) + 1
	nLayer = 2
	batch_size = 1

	h0 = Variable(torch.zeros(nLayer, batch_size, nHidden))
	c0 = Variable(torch.zeros(nLayer, batch_size, nHidden))
	state0 = (h0, c0)

	print ('Build model')
	net = Net(nIn, nHidden, nLayer, nOut)
	criterion = CTCLoss()
	optimizer = torch.optim.Adadelta(net.parameters())
	decoder = ArgMaxDecoder(char_dataset, decode_lookup)

	imgs_seq = [padding_seq(img, nIn)[0] for img in imgs_seq]

	checkpoint = 1
	epochs = 1
	model_name = 'rnn_ctc_3' # model: 1
	if old_epochs > 0:
		model_load = '%s_%02d.pth' % (model_name, old_epochs)
		print ('Load model from %s' % model_load)
		net = torch.load(model_load) 

	print ('Training')
	for epoch in range(1, epochs+1):  
		loss_epochs = 0.0
		for i in list_index_train:
			input = Variable(torch.Tensor([imgs_seq[i]]))
			input = input.view(int(input.size(1)/nIn), 1, nIn)
			label = Variable(torch.IntTensor(labels[i]))
			output = net.forward(input, state0)

			label_len = Variable(torch.IntTensor([label.size(0)]))
			act_len = Variable(torch.IntTensor([output.size(0)]))

			loss = criterion(output, label, act_len, label_len) 
			loss_epochs += loss / len(list_index_train)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			
		if (epoch == 1) or (epoch % checkpoint == 0):
			print ('Epoch: %d' % (old_epochs + 1))
			if SAVE_MODEL:
				model_name_epoch = '%s_%02d.pth' % (model_name, old_epochs + epoch)
				print ('	Save model as %s' % model_name_epoch)
				torch.save(net, model_name_epoch)        
			print ('   	+ Train (ctc-loss) ', loss_epochs.data)
			wer, cer = check(net, decoder, imgs_seq, labels, list_index_dev, (nIn, state0))
			print ('   	+ Dev (wer, cer): ', wer, cer)


if __name__ == '__main__':
	main()