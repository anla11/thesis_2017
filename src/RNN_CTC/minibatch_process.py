import numpy as np
import random 

def sort_listseq(listseq):
    seq_len = [len(listseq[j]) for j in range(len(listseq))]
    mydict = dict(zip(range(len(seq_len)), seq_len))
    index = [w for w in sorted(mydict, key=mydict.get, reverse=True)]
    return index

def padding_listseq(listseq, align):
    seq_len = [len(seq) for seq in listseq]
    maxlen = max(seq_len)
    tmp = maxlen % align
    maxlen += align * (tmp != 0) - tmp
#     if (len(listseq) == 1):
#         new_seq, seq_len, maxlen = padding_seq(listseq[0], align)
#         return [new_seq], [seq_len], maxlen
    new_seq = [listseq[i] + [0] * (maxlen-seq_len[i]) for i in range(len(listseq))]            
    return new_seq, seq_len, maxlen

def padding_seq(seq, align):
#     check = isinstance(seq, list)
#     if check == False:
#         seq = seq.tolist()
    tmp = len(seq) % align
    maxlen = len(seq) + align * (tmp != 0) - tmp
    seq_len = len(seq)
    print (maxlen-seq_len)
    new_seq = seq + [0] * (maxlen-seq_len)
    return new_seq, seq_len, maxlen

def sort_minibatch(imgs, labels):
    new_idx = sort_listseq(imgs)
    imgs = np.array(imgs)[new_idx].tolist()
    labels = np.array(labels)[new_idx].tolist()
    return imgs, labels

