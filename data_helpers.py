import numpy as np
import jieba
import random
import pdb
def get_rand_vect(emb_size):
    rand_list = []
    for i in xrange(emb_size):
        rand_list.append(random.uniform(-1,1))
    return rand_list

def load_embedding(emb_file, vocab_set, emb_size=200):
    word_vect = {}
    word_id = {}
    id_word = {}
    idx = 0
    word_id['<PAD/>'] = idx
    word_vect['<PAD/>'] = get_rand_vect(emb_size=200)
    id_word[idx] = '<PAD/>'
    idx += 1
    word_id['<UNKNOWN>'] = idx
    word_vect['<UNKNOWN>'] = get_rand_vect(emb_size=200)
    id_word[idx] = '<UNKNOWN>'
    with open(emb_file) as f:
        for line in f:
            tmp_list = line.strip().split(" ")
            if len(tmp_list) < emb_size + 1:continue
            word = tmp_list[0]
            vect = tmp_list[1::]
            if word in vocab_set:
                word_vect[word] = map(float, vect)
                word_id[word] = idx
                id_word[idx] = word
                idx += 1
    return word_vect, word_id, id_word

def get_word_vocab(data_list):
    #data_list['A B c','D e f']
    #every line is a string
    #only get freq > 1 word
    word_vocab = set([])
    word_freq = defaultdict(lambda: 0)
    for line in data_list:
        for item in line.strip().split(" "):
            word_freq[item] += 1
    [word_vocab.add(item) for item,freq in word_freq.items() if freq > 1]
    return word_vocab

def pad_corpus(data_list, padding_word, max_len):
    #data_list['A B c','D e f']
    #padding_word="<PAD/>"
    pad_list = []
    real_len_list = []
    for i in range(len(data_list)):
        sent = data_list[i]
        tmp_list = sent.strip().split(' ')
        seq_len = len(tmp_list)
        diff_len = max_len - seq_len
        if diff_len > 0:
            real_len_list.append(seq_len)
            for j in range(diff_len):
                tmp_list.append(padding_word)
        else:
            tmp_list = tmp_list[:max_len]
            seq_len.append(max_len)
        assert len(tmp_list) == max_len
        pad_sent = ' '.join(tmp_list)
        pad_list.append(pad_sent)
    return pad_list,real_len_list

def encode_word(data_list, word_id):
    #data_list['A B c','D e f']
    id_list = []
    for sent in data_list:
        tmp_list = sent.strip().split(' ')
        id_vect = []
        for word in tmp_list:
            if word in word_id:
                id_vect.append(word_id[word])
            else:
                id_vect.append(word_id['<UNKNOWN>'])
        id_list.append(id_vect)
    return id_list

def get_word_vect(id_word, word_vect):
    emb_mat = []
    for idx in xrange(len(id_word)):
        emb_mat.append(word_vect[id_word[idx]])
    return mat

def get_seg_list(query):
    seg_list = jieba.cut(query)
    return seg_list

def seg_file(file_name):
    #prepare train data for segment
    fw = open(file_name+".seg","w")
    with open(file_name) as f:
        for line in f:
            tmp_list = line.strip().split('\t')
            query = tmp_list[1]
            label = tmp_list[0]
            query_seg_list = get_seg_list(query)
            query_seg = ' '.join(query_seg_list).encode('utf-8')
            fw.write(label + "\t" + query_seg+'\n')
    fw.close()
    return

def load_multi_task_data(train_data_file, task_num):
    """
    Loads multi task train data.
    Input: task1_label\ttask2_label\ttask3_label\t,,,\ttaskn_label\tquery_seg
    """
    # Load data from files
    train_examples = []
    train_label = [[] for i in range(task_num)]
    with open(train_data_file) as f:
        for line in f:
            tmp_list = line.strip().split('\t')
            if len(tmp_list) < task_num + 1:continue
            train_examples.append(tmp_list[-1])
            for i in range(task_num):
                train_label[i].append(tmp_list[i])
    
    # Generate labels
    uniq_label = [sorted(list(set(tmp_label))) for tmp_label in train_label]
    num_label = [len(tmp_label) for tmp_label in uniq_label]
    label_dict = [dict(zip(uniq_label[i], np.identity(num_label[i], int))) for i in range(task_num)]#one hot encoding for label
    train_label_one_hot = [get_label_one_hot_single_task(label, label_dict) for label,label_dict in zip(train_label, label_dict)]
    train_label = [np.array(train_label_one_hot[i]) for i in range(task_num)]
    return train_examples, train_label, uniq_label

def get_label_one_hot_single_task(label_list, label_dict):
    return [label_dict[y] for y in label_list]

def load_test_data(test_file, sorted_label, task_num):
    """
    Loads test data according to train data.
    Input: task1_label\ttask2_label\ttask3_label\t,,,\ttaskn_label\tquery_seg
    """
    # Load data from files
    test_examples = []
    raw_label = [[] for i in range(task_num)]
    with open(test_file) as f:
        for line in f:
            tmp_list = line.strip().split('\t')
            if len(tmp_list) < task_num + 1:continue        
            test_examples.append(tmp_list[-1])
            for i in range(task_num):
                raw_label[i].append(tmp_list[i])
    
    # Generate labels
    num_label = [len(sorted_label[i]) for i in range(task_num)]
    label_dict = [dict(zip(sorted_label[i], np.identity(num_label[i], int))) for i in range(task_num)]#one hot encoding for label
    test_label_one_hot = [get_label_one_hot_single_task(label, dic) for label, dic in zip(raw_label, label_dict)]
    test_label = [np.array(test_label_one_hot[i]) for i in range(task_num)]
    return test_examples, test_label

def batch_iter(data, batch_size, num_epochs, shuffle_indices, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

def multi_task_batch_iter(data_list, batch_size, num_epochs, shuffle=True):
    """
    Generates multi batch iterator for multi task.
    """
    batcher_list = []
    assert len(data_list) > 0
    single_task_data_len = len(data_list[0])
    for i in range(len(data_list)):
        assert single_task_data_len == len(data_list[i])
    shuffle_indices = np.random.permutation(np.arange(single_task_data_len))
    for data in data_list:
        batcher_list.append(
            batch_iter(data, batch_size, num_epochs, shuffle_indices, shuffle))
    total_batch_num = num_epochs * (int((single_task_data_len-1)/batch_size) + 1)
    return batcher_list, total_batch_num

if __name__=='__main__':
    seg_file('./data/cnews.test.txt')
