import cv2
import numpy as np 
import random

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

def read_data(filename = '../dataset/train.txt', limit = 4000):
    with open(filename) as f:
        content = f.readlines()
    image_list = ['../' + item.split()[0] for item in content[:limit]]
    title_list = ['../' + item.split()[1] for item in content[:limit]]
    return image_list, title_list

def read_dict():
    lookup = {}
    char_dataset = []
    with open('../chardict.txt') as f:
        content = f.readlines()
        char_dataset = sorted([item[0] for item in content])
        label_dataset = sorted([int(item[2:]) for item in content])
        decode_lookup = dict(zip(label_dataset, char_dataset))
        lookup = dict(zip(char_dataset, label_dataset))
    # char_dataset = char_dataset + ['_']
    lookup['_'] = 0
    decode_lookup[0] = '_'
    return char_dataset, lookup, decode_lookup    

def read_listimg(lookup, char_dataset, image_list, title_list, list_index):
    imgs = read_img(np.array(image_list)[list_index].tolist())
    labels = title2label(read_title(np.array(title_list)[list_index].tolist()), lookup, char_dataset)
    return imgs, labels

