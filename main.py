#!/usr/bin/env python
# coding: utf-8


# In[ ]:


import argparse
import csv
import os
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--max_epoch', help='max epoch', type=int, dest='max_epoch', default=200)
parser.add_argument('--debug', help='debug or not', type=int, dest='debug', default=0)
parser.add_argument('--gpu', help='gpu index', type=int, dest='gpu', default=0)
parser.add_argument('--fold', help='cross validation fold', type=str, dest='fold', default='4')
parser.add_argument('--seed', help='random seed', type=int, dest='seed', default=8)
parser.add_argument('--outfile', help='where to save metrics', type=str, dest='out', default='chapter511_and_52_53.csv')
parser.add_argument('--branches', help='sub classification target', type=str, dest='branches', default=None)
parser.add_argument('--max_length', help='max length of input seq', type=int, dest='max_length', default=469)
args = parser.parse_args()

gpu_visible = str(int(args.gpu))  # when more than one gpu, plz be careful at gpu index.
SEED = args.seed
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_visible
import torch

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from torch.utils import data

import numpy as np

np.random.seed(SEED)
import random
random.seed(SEED)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import pudb
import pandas as pd
from visdom import Visdom
from utils import pad_and_sort_batch, save_checkpoint
from model import XfModel
from focalloss import *
# add 5.3 sampler 
from my_classes_vary_one_file import Dataset, EvenlyLengthSampler
from pytorch_tools import EarlyStopping
import json

# In[ ]:

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
if device == 'cpu':
    print('no gpu found')
    os._exit(0)

DEBUG = args.debug
max_epochs = args.max_epoch
branches = args.branches
if branches is not None:
    branches = eval(branches)
gamma = 0
fold = args.fold
max_length = args.max_length
path_to_save_result = '../' + args.out
if DEBUG:
    pudb.set_trace()

configs = {'batch_size': 40,  # lg
           'num_workers': 2}
lr = 0.00002
ALLDATA = False  # 全部或仅自发

if max_length <= 0:
    max_length = None  # for unlimited length

path_to_features = '/home/wangce/iemocap_specs/vary_128.hdf5'

if gamma == 0:
    loss_func = nn.CrossEntropyLoss()
else:
    loss_func = FocalLoss(gamma=gamma)

dic = {'gamma': gamma, 'allData': ALLDATA}
dic.update(vars(args))  # add args to dic
dic.update(configs)
# In[ ]:


labels = {}
dict_emo = {'a': 0, 'h': 1, 'n': 2, 's': 3}
trainData, testData = [], []
with open('total.txt', 'r') as f:
    total = f.read().split('\n')
for item in total:
    if ALLDATA or 'impro' in item:
        if item.startswith('0{fold}'.format(fold=fold)):  # e.g. 04F
            testData.append(item)
        else:
            trainData.append(item)
        labels[item] = dict_emo[item[4]]

training_set = Dataset(trainData, labels, path=path_to_features, max_length=max_length)
validation_set = Dataset(testData, labels, path=path_to_features, max_length=max_length)

training_generator = data.DataLoader(training_set, collate_fn=pad_and_sort_batch,
                                     sampler=EvenlyLengthSampler(training_set), **configs)
validation_generator = data.DataLoader(validation_set, collate_fn=pad_and_sort_batch,
                                       sampler=EvenlyLengthSampler(validation_set), **configs)

if DEBUG:
    pudb.set_trace()
model = XfModel(branches=branches).to(device)
optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0005)

print(sum(param.numel() for param in model.parameters()))

# In[ ]:

envPrefix = '0313BSL_'

localTimeStrs = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
envName = envPrefix + localTimeStrs
best = {'ua': 0.0}
path_prefix = '/home/wangce/2021_checkpoints/'
out_dir = Path(os.path.join(path_prefix, localTimeStrs))
out_dir.mkdir(parents=True, exist_ok=True)
to_log = {}
# In[ ]:

if not DEBUG:
    viz = Visdom(env=envName, port=8101)
    viz.text(str(dic), opts={'title': 'settings'})
    run_command = " ".join(sys.argv)
    viz.text(run_command, opts={'title': 'running command'})
    print(envName)
    train_confusion_history, train_ua_history, train_wa_history, train_loss_history = [], [], [], []
    test_confusion_history, test_ua_history, test_wa_history, test_loss_history = [], [], [], []
# In[ ]:

early_stopping = EarlyStopping(patience=25, verbose=True,
                               path=os.path.join(out_dir, 'best_test_loss.pth'))
t1 = time.time()
record = pd.DataFrame()
path_to_record = Path(out_dir, 'record.csv')
for epoch in range(1, max_epochs):
    Y_pred = torch.LongTensor().to(device)
    Y_true = torch.LongTensor().to(device)
    y_scores = torch.FloatTensor().to(device)
    filenames = []
    y_pred = torch.LongTensor().to(device)
    y_true = torch.LongTensor().to(device)

    trainBatchLoss = []
    testBatchLoss = []
    model.train()
    for local_batch, local_labels, local_lengths, local_ids in training_generator:
        # Transfer to GPU
        local_batch, local_labels, local_lengths = local_batch.to(device), local_labels.to(device), local_lengths.to(
            device)

        # Model computations
        outputs, loss = model(local_batch, local_lengths, is_train=True, labels=local_labels)
        optim.zero_grad()
        loss.backward()
        ############
        optim.step()

        trainBatchLoss.append(loss.item())
        if DEBUG:
            pudb.set_trace()

    t2 = time.time()
    print('training model an epoch takes ' + str(round(t2 - t1, 2)) + 'seconds.')
    t1 = t2

    # Validation
    model.eval()
    with torch.set_grad_enabled(False):
        for local_batch, local_labels, local_lengths, local_ids in training_generator:
            local_batch, local_labels, local_lengths = local_batch.to(device), local_labels.to(
                device), local_lengths.to(device)

            outputs, _ = model(local_batch, local_lengths)
            _, predict = torch.max(outputs, 1)
            Y_pred = torch.cat([Y_pred, predict], 0)
            Y_true = torch.cat([Y_true, local_labels], 0)
        for local_batch, local_labels, local_lengths, local_ids in validation_generator:
            # Transfer to GPU
            local_batch, local_labels, local_lengths = local_batch.to(device), local_labels.to(
                device), local_lengths.to(device)

            outputs, _ = model(local_batch, local_lengths)
            if local_batch.shape[0] == 1:
                local_ids = [local_ids]
            filenames.extend(local_ids)
            loss = loss_func(outputs, local_labels)
            testBatchLoss.append(loss.item())
            # print('testLoss: ', loss)
            _, predict = torch.max(outputs, 1)
            y_pred = torch.cat([y_pred, predict], 0)
            y_true = torch.cat([y_true, local_labels], 0)
            y_scores = torch.cat([y_scores, outputs], 0)  # 拼接分数

    t2 = time.time()
    print('test model(training and testing set) takes ' + str(round(t2 - t1, 2)) + 'seconds.')
    t1 = t2
    Y_pred, Y_true, y_pred, y_true = Y_pred.cpu(), Y_true.cpu(), y_pred.cpu(), y_true.cpu()
    Confusion = confusion_matrix(Y_true, Y_pred).astype('float16').T
    confusion = confusion_matrix(y_true, y_pred).astype('float16').T
    Wa = precision_score(Y_true, Y_pred, average='micro')
    Ua = recall_score(Y_true, Y_pred, average='macro')
    wa = precision_score(y_true, y_pred, average='micro')
    ua = recall_score(y_true, y_pred, average='macro')
    # for load model and continue training, this should not take too much storage.
    if epoch % 100 == 0 and not DEBUG:
        save_checkpoint(model=model, optimizer=optim, path=out_dir, string=epoch)
    trainLoss = np.mean(trainBatchLoss)
    testLoss = np.mean(testBatchLoss)
    print(epoch)
    print('trainUA #{:.4f},trainWA #{:.4f}'.format(Ua, Wa))
    print(Confusion)
    print(Confusion.diagonal() / Confusion.sum(0))
    print('testUA #{:.4f},testWA #{:.4f}'.format(ua, wa))
    print(confusion)
    test_emotion_accs = confusion.diagonal() / confusion.sum(0)
    print(test_emotion_accs)
    if ua > float(
            best['ua']) and testLoss < 1.3 * early_stopping.val_loss_min:  # for lessen over fitting,0.8(usual)*1.3
        best['ua'] = '{:.4f}'.format(ua)
        best['wa'] = '{:.4f}'.format(wa)
        best['ang'] = test_emotion_accs[0].item()
        best['hap'] = test_emotion_accs[1].item()
        best['neu'] = test_emotion_accs[2].item()
        best['sad'] = test_emotion_accs[3].item()
        best['testloss'] = '{:.4f}'.format(testLoss)
        best['epoch'] = epoch
        best['visdom'] = envName
        record['id'] = filenames
        record['y_true'] = y_true
        record['y_pred'] = y_pred
        record['ang'] = y_scores[:, 0].cpu()
        record['hap'] = y_scores[:, 1].cpu()
        record['neu'] = y_scores[:, 2].cpu()
        record['sad'] = y_scores[:, 3].cpu()
        if not DEBUG:
            save_checkpoint(model=model, optimizer=optim, path=out_dir, string='best')
    if not DEBUG:
        viz.line(X=np.column_stack(np.array([epoch, epoch])), Y=np.column_stack(np.array([trainLoss, testLoss])),
                 opts={'legend': ['trainLoss', "testLoss"], 'title': 'loss'}, win='loss', update='append')
        viz.line(X=np.column_stack(np.array([epoch, epoch, epoch, epoch])),
                 Y=np.column_stack(np.array([Wa, Ua, wa, ua])),
                 opts={'legend': ['trainWA', 'trianUA', 'testWA', 'testUA'], 'title': 'metrics'}, win='metrics',
                 update='append')
        viz.line(X=np.column_stack(np.array([epoch, epoch, epoch, epoch])),
                 Y=np.column_stack(np.array(test_emotion_accs)),
                 opts={'legend': ['test_ang', 'test_hap', 'test_neu', 'test_sad'], 'title': 'test_emotion_metrics'},
                 win='test_emotion_metrics', update='append')

        viz.save([envName])
        train_loss_history.append(str(trainLoss))
        train_confusion_history.append(str(Confusion))
        train_ua_history.append(str(Ua))
        train_wa_history.append(str(Wa))
        test_loss_history.append(str(testLoss))
        test_confusion_history.append(str(confusion))
        test_ua_history.append(str(ua))
        test_wa_history.append(str(wa))
    early_stopping(testLoss, model)
    if early_stopping.early_stop:
        print("Early stopping")
        break
if not DEBUG:
    viz.text(str(best))

record.sort_values('id').to_csv(path_to_record, index=False)
if not os.path.exists(path_to_save_result):
    with open(path_to_save_result, 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        to_write = list(best.keys()) + list(dic.keys())
        writer.writerow(to_write)
with open(path_to_save_result, 'a+', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter='\t')
    to_write = list(best.values()) + list(dic.values())
    writer.writerow(to_write)
try:
    from git import Repo

    repo_dir = os.path.abspath(os.path.dirname(__file__))
    repo = Repo(repo_dir)
    commit = repo.head.commit.hexsha[:7]
except:
    commit = None
    print('not a git repo')
to_log = {
    'train_loss_history': train_loss_history,
    'train_ua_history': train_ua_history,
    'train_wa_history': train_wa_history,
    'train_confusion_history': train_confusion_history,
    'test_loss_history': test_loss_history,
    'test_ua_history': test_ua_history,
    'test_wa_history': test_wa_history,
    'test_confusion_history': test_confusion_history,
    'env_name': envName,
    'run_command': run_command,
    'best': best,
    'epoch_run': epoch,
    'config': dic,
    'commit ID': commit,
    'model': str(model)
}
with open(Path(out_dir, 'log.json'), 'w') as f:
    f.write(json.dumps(to_log, indent=4, sort_keys=True))
