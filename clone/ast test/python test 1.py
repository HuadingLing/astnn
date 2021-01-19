import sys
if "D:\\GitHub\\DLpipeline\\dev" not in sys.path:
    sys.path.append("D:\\GitHub\\DLpipeline\\dev")
from dlpipeline import DLpipeline, Saver, FileNameManager, BasicExecutor, BasicReporter, Progbar, FormatDisplay2, reconstruct_from_cm, autolabel, plot_confusion_matrix
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score, auc
import matplotlib.pyplot as plt
import os


class ASTNN_Executor(BasicExecutor):
    def __init__(self, **kwargs):
        super(ASTNN_Executor, self).__init__(**kwargs)

    def train(self):
        pipeline = self.pipeline
        device = pipeline.device
        model = pipeline.model
        criterion = pipeline.criterion
        optimizer = pipeline.optimizer
        progressbar = pipeline.progressbar
        
        model.train()
        tot_loss = tot_correct = total = 0
        hist = []
        batch_size = pipeline.trainloader.batch_size
        format_display2 = FormatDisplay2(len(pipeline.trainloader.dataset))

        progressbar.bar_prepare('train', _format=format_display2._format)
        for batch_idx, (x1, x2, targets) in enumerate(pipeline.trainloader, 1):
            targets = targets.to(device)
            optimizer.zero_grad()
            
            model.batch_size = num = targets.size(0)
            model.hidden = model.init_hidden()
            outputs = model(x1, x2)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            tot_loss += batch_loss
            total += num
            with torch.no_grad():
                predicted = np.where(outputs.cpu().numpy().reshape(-1, 1) < 0.5, 1, 0)
                correct = (predicted==targets.cpu().numpy().reshape(-1, 1)).sum()
            tot_correct += correct

            progressbar(batch_idx, (tot_loss / batch_idx if num == batch_size  # 'else' is for last batch correction， calculation is kind of weird, to prevent overflow
                                    else (tot_loss - batch_loss * (1 - num / batch_size)) * (batch_size / len(pipeline.trainloader.dataset)),
                                    tot_correct / total,
                                    format_display2(tot_correct, total),
                                    batch_loss,
                                    correct / num,
                                    correct,
                                    num))
            hist.append((batch_loss, correct))

        return hist, tot_correct / total

    def validation(self):
        pipeline = self.pipeline
        device = pipeline.device
        model = pipeline.model
        criterion = pipeline.criterion
        progressbar = pipeline.progressbar
        
        model.eval()
        tot_loss = tot_correct = total = 0
        batch_size = pipeline.valloader.batch_size
        format_display2 = FormatDisplay2(len(pipeline.valloader.dataset))

        progressbar.bar_prepare('val', _format=format_display2._format)
        with torch.no_grad():
            for batch_idx, (x1, x2, targets) in enumerate(pipeline.valloader, 1):
                targets = targets.to(device)
                model.batch_size = num = targets.size(0)
                model.hidden = model.init_hidden()
                outputs = model(x1, x2)
                loss = criterion(outputs, targets)

                tot_loss += loss.item()
                total += num
                predicted = np.where(outputs.cpu().numpy().reshape(-1, 1) < 0.5, 1, 0)
                tot_correct += (predicted==targets.cpu().numpy().reshape(-1, 1)).sum()
                val_loss = tot_loss / batch_idx if num == batch_size \
                            else (tot_loss - loss.item() * (1 - num / batch_size)) * (batch_size / len(pipeline.valloader.dataset))

                progressbar(batch_idx, (val_loss,
                                        tot_correct / total,
                                        format_display2(tot_correct, total)))

        return (val_loss, tot_correct), tot_correct / total

    def test(self):
        pipeline = self.pipeline
        device = pipeline.device
        model = pipeline.model
        criterion = pipeline.criterion
        progressbar = pipeline.progressbar
        need_cm = pipeline.reporter.need_confusion_matrix
        need_output = pipeline.reporter.need_output
        need_store = need_cm or need_output
        
        
        if need_store:
            y_true = []
            if need_cm:
                y_pred = []
            if need_output:
                y_output = []
        #labels = self.pipeline.reporter.labels
        #cm = np.zeros((len(labels),len(labels)))
        
        model.eval()
        tot_loss = tot_correct = total = 0
        batch_size = pipeline.testloader.batch_size

        progressbar.bar_prepare('test')
        with torch.no_grad():
            for batch_idx, (x1, x2, targets) in enumerate(pipeline.testloader, 1):
                targets = targets.to(device)
                model.batch_size = num = targets.size(0)
                model.hidden = model.init_hidden()
                outputs = model(x1, x2)
                loss = criterion(outputs, targets)

                tot_loss += loss.item()
                total += num
                outputs = outputs.cpu().numpy().reshape(-1, 1)
                predicted = np.where(outputs < 0.5, 1, 0)
                targets = targets.cpu().numpy().reshape(-1, 1)
                tot_correct += (predicted==targets).sum()
                test_loss = tot_loss / batch_idx if num == batch_size \
                            else (tot_loss - loss.item() * (1 - num / batch_size)) * (batch_size / len(pipeline.testloader.dataset))

                progressbar(batch_idx, (test_loss,
                                        tot_correct / total,
                                        tot_correct,
                                        total))
                # precision，recall，F1 can be considered
                
                if need_store:
                    y_true.extend(list(targets))
                    if need_cm:
                        y_pred.extend(list(predicted))
                    if need_output:
                        y_output.extend(list(outputs))
                
        test_hist = {'test loss': test_loss}
        if need_cm:
            test_hist['confusion matrix'] = confusion_matrix(y_true, 
                                                             y_pred,
                                                             labels = self.pipeline.reporter.labels)
        if need_output:
            test_hist['y_true'] = y_true
            test_hist['output'] = y_output

        return test_hist, tot_correct / total
    
    
class Reporter_c2(BasicReporter):
    def __init__(self, 
                 labels = None,  # numerical value label of the classes
                 class_names = None,  # string, use to display the classes
                 need_output = True,  # whether to store the output of model during testing. useful for ROC and AUC
                 need_confusion_matrix = False, # whether to store confusion matrix during testing
                 output_to_score_fun = None,  # function to transfer the output of model to score-like value (i.e., sofmax, normalized, sum is 1)
                 batch_figsize = (14, 6),  # figsize of batch-loss-acc figure in the report
                 epoch_figsize = (10, 6), # figsize of epoch-loss-acc figure in the report
                 cm_figsize = (10, 8),  # figsize of confusion matrix figure in the report
                 cr_figsize = (18, 8),  # figsize of (sklearn's) classification_report figure in the report
                 roc_figsize = (10, 8),  # figsize of ROC curve figure in the report
                 **kwargs):
        super(Reporter_c2, self).__init__(**kwargs)
        
        self.labels = labels
        if class_names is None:
            if labels is not None:
                self.class_names = [str(label) for label in labels]
            else:
                print('Error for the labels.')
        else:
            self.class_names = class_names
        assert len(self.labels) == len(self.class_names)
        self.need_output = need_output
        self.need_confusion_matrix = need_confusion_matrix
        self.output_to_score_fun = output_to_score_fun
        self.batch_figsize = batch_figsize
        self.epoch_figsize = epoch_figsize
        self.cm_figsize = cm_figsize
        self.cr_figsize = cr_figsize
        self.roc_figsize = roc_figsize
        
        
    def check_and_report(self):
        if self.report_interval > 0 and (self.pipeline.epoch - self.pipeline.start_epoch + 1) % self.report_interval == 0:
            self.history['epoch'] = self.pipeline.epoch
            if self.show_train_report or self.pipeline.save_train_report:
                self.plot_train(hist = self.history, in_train = True)
        
    def __call__(self, **kwargs):
        self.plot_hist(self.history, **kwargs)
        pass
    
    def plot_hist(self, hist, modes = 'all', **kwargs):
        if isinstance(modes, str):
            modes = [modes]
        assert isinstance(modes, list)
        if ('train' in modes or 'all' in modes) and 'train' in hist.keys() and hist['train']:
            self.plot_train(hist, 'val' in modes or 'all' in modes, **kwargs)
        if ('test' in modes or 'all' in modes) and 'test' in hist.keys() and hist['test']:
            self.plot_test(hist)

    def plot_train(self, hist, plot_val = True, in_train = False, drop_epochs = 0):
        start_epoch = hist['start epoch']
        temp_epoch = hist['epoch']
        e = range(start_epoch, temp_epoch + 1)
        batch_size = hist['batch size']
        train_size = hist['trainset size']
        last_size = train_size % batch_size  # size of the last batch
        iters = train_size // batch_size
        train_batch_loss = [l2[0] for l1 in hist['train'] for l2 in l1]
        train_batch_correct = [l2[1] for l1 in hist['train'] for l2 in l1]
        # train_loss = [l2[0]*batch_size for l1 in hist['train'] for l2 in l1]

        if iters == len(hist['train'][0]):  # len(hist['train'][0]) indicates how many train_batch in one epoch
            # no remnant batch
            train_batch_acc = [l2[1]/batch_size for l1 in hist['train'] for l2 in l1]

            train_loss = np.array(train_batch_loss).reshape(-1, iters)
            train_acc = np.array(train_batch_acc).reshape(-1, iters)
            train_loss = np.mean(train_loss, axis=1)
            train_acc = np.mean(train_acc, axis=1)

        else:  # the last train_batch is remnant
            train_batch_acc = [(l1[i][1]/batch_size if i < iters else l1[i][1]/last_size)
                               for l1 in hist['train'] for i in range(iters+1)]

            train_loss = np.array(train_batch_loss).reshape(-1, iters+1)
            train_acc = np.array(train_batch_correct).reshape(-1, iters+1)

            train_loss = (np.sum(train_loss[:,:-1], axis=1) * batch_size + train_loss[:,-1] * last_size) / train_size
            train_acc = np.sum(train_acc, axis=1) / train_size
        
        iters = range(1, len(train_batch_loss)+1)
        if drop_epochs > 0:
            if drop_epochs > len(hist['train']):
                drop_epochs = len(hist['train'])
            drop_iters = drop_epochs*len(hist['train'][0])
            iters = iters[drop_iters:]
            train_batch_loss = train_batch_loss[drop_iters:]
            train_batch_acc = train_batch_acc[drop_iters:]
            
        fig, ax1 = plt.subplots(figsize=self.batch_figsize, facecolor='white')
        color = 'C0'
        ax1.set_xlabel('Iters')
        ax1.set_ylabel('Batch Loss', color=color)
        ax1.scatter(iters, train_batch_loss, s=8, color=color, marker='2', label='batch loss')
        # marker: ascent uses '1', descent used '2', and more like '.', '+', 'x' and '|' ,just personal preference
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'C2'
        ax2.set_ylabel('Batch Acc', color=color)  # we already handled the x-label with ax1
        ax2.scatter(iters, train_batch_acc, s=8, color=color, marker='1', label='batch acc')
        ax2.tick_params(axis='y', labelcolor=color)

        ax1.legend(loc='upper left')
        ax2.legend(loc='lower left')
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        if self.pipeline.saver.save_train_report or (not in_train and self.pipeline.saver.save_test_report):
            # not in train equal to in test, depend on saver.save_test_report at this time
            save_dir = os.path.join(self.pipeline.saver.execute_save_dir, self.pipeline.file_name_manager('report train'))
            fig.savefig(save_dir, transparent=False, dpi=80, bbox_inches="tight")
        if not in_train or self.show_train_report:
            plt.show()



        fig, ax1 = plt.subplots(figsize=self.epoch_figsize, facecolor='white')
        plt.grid(True, which='major', axis='y')  # place grid first, to prevent it cover the line follow-up (put grid in the bottom layer)
        color = 'C0'
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss', color=color)
        ax1.plot(e[drop_epochs:], train_loss[drop_epochs:], linestyle='-.', color=color, label='train loss')
        ax1.tick_params(axis='y', labelcolor=color)


        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'C2'
        ax2.set_ylabel('Acc', color=color)  # we already handled the x-label with ax1
        ax2.plot(e[drop_epochs:], train_acc[drop_epochs:], linestyle='-.', color=color, label='train acc')
        ax2.tick_params(axis='y', labelcolor=color)


        if plot_val and 'val' in hist.keys() and hist['val']:
            val_loss = [l[0] for l in hist['val']]
            val_acc = [l[1]/hist['valset size'] for l in hist['val']]
            if len(val_loss) == len(e):
                ev = e
            else:  # val incomplete, maybe some experiments haven't setup valloader
                ev = range(temp_epoch - len(val_loss) + 1, temp_epoch + 1)

            ax1.plot(e[drop_epochs:], val_loss[drop_epochs:], color='C4', label='val loss')
            ax2.plot(e[drop_epochs:], val_acc[drop_epochs:], color='C1', label='val acc')

        ax1.legend(loc='upper left')
        ax2.legend(loc='lower left')
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        if self.pipeline.saver.save_train_report or (not in_train and self.pipeline.saver.save_test_report):
            save_dir = os.path.join(self.pipeline.saver.execute_save_dir, self.pipeline.file_name_manager('report val'))
            fig.savefig(save_dir, transparent=False, dpi=80, bbox_inches="tight")
        if not in_train or self.show_train_report:
            plt.show()
        
    def plot_test(self, hist):
        test_dict = hist['test']
        y_true = None
        y_score = None
        y_pred = None
        cm = None
        
        # first see which we can get: y_true, y_score, y_pred, cm
        # use y_score can get y_pred
        # use y_true and y_pred can get cm
        # use cm can get y_true and y_pred
        if 'confusion matrix' in test_dict.keys():
            cm = test_dict['confusion matrix']
            # tn, fp, fn, tp = cm.ravel()
        if 'y_true' in test_dict.keys() and 'output' in test_dict.keys():
            y_true = test_dict['y_true']
            if self.output_to_score_fun:
                y_score = self.output_to_score_fun(test_dict['output'])
            else:
                y_score = np.array(test_dict['output'])  # make sure it's ndarray
            y_pred = np.where(y_score < 0.5, 1, 0)
            if cm is None:
                cm = confusion_matrix(y_true, y_pred, labels = self.labels)

        if cm is not None:
            assert len(cm) == len(self.labels)
            if self.pipeline.saver.save_test_report:
                save_dir = os.path.join(self.pipeline.saver.execute_save_dir, self.pipeline.file_name_manager('report cm'))
            else:
                save_dir = None
            plot_confusion_matrix(cm, labels = self.labels, figsize=self.cm_figsize, 
                                  title = 'Confusion matrix',
                                  cmap='BuGn',  # can also use cmap='GnBu', just personal preference
                                  #xticks_rotation = 30,
                                  axes_style='dark', context_mode='notebook',
                                  save_dir = save_dir, transparent=False, dpi=80, bbox_inches="tight") 
            print('Confusion matrix:')
            print(cm)  # display numerical value (text)

            if y_true is None or y_pred is None:
                y_true, y_pred = reconstruct_from_cm(cm)
            cr_dict = classification_report(y_true, y_pred, labels = self.labels, target_names = self.class_names, digits = 4, output_dict = True)
            self.plot_classification_report(cr_dict, self.cr_figsize)
            cr = classification_report(y_true, y_pred, labels = self.labels, target_names = self.class_names, digits = 4, output_dict = False)
            print('\nClassification report:\n')
            print(cr)  # display classification report (text)
            print()
        if y_true is not None and y_score is not None:
            self.plot_roc(y_true, 1-y_score)
            
            
    def plot_classification_report(self, cr, figsize=(16, 8), width=0.24):
        classes_labels = list(self.class_names)
        classes_labels.append('macro avg')
        classes_labels.append('weighted avg')
        precision = [cr[cl]['precision'] for cl in classes_labels]
        recall = [cr[cl]['recall'] for cl in classes_labels]
        f1 = [cr[cl]['f1-score'] for cl in classes_labels]

        classes_x = np.arange(len(classes_labels))  # the label locations

        fig, ax = plt.subplots(figsize=figsize, facecolor='w')
        ax.grid(True, which='major', axis='y')

        rects_precision = ax.bar(classes_x - width, precision, width, label='precision')
        rects_recall = ax.bar(classes_x, recall, width, label='recall')
        rects_f1 = ax.bar(classes_x + width, f1, width, label='f1-score')
        rects_acc = ax.bar(len(classes_x) - width, cr['accuracy'], width*1.2, color='C7', label='Acc')

        xticks = [i for i in classes_x]
        xticks.append(len(classes_x)-width)
        xticklabels = [cl+'\n('+str(cr[cl]['support']) + ')' for cl in classes_labels]
        xticklabels.append('Acc\n({:})'.format(cr['macro avg']['support']))

        ax.set_ylabel('Scores')
        ax.set_ylim([0.0, 1.02])
        ax.set_title('Scores by Multi-class')
        ax.set_xlabel('Items and support')
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.set_xlim([-3*width, len(classes_x)+width])
        ax.legend()

        autolabel(ax, rects_precision)
        autolabel(ax, rects_recall)
        autolabel(ax, rects_f1)
        autolabel(ax, rects_acc)

        fig.tight_layout()
        if self.pipeline.saver.save_test_report:
            save_dir = os.path.join(self.pipeline.saver.execute_save_dir, self.pipeline.file_name_manager('report cr'))
            fig.savefig(save_dir, transparent=False, dpi=80, bbox_inches="tight")
        plt.show()
            
    def plot_roc(self, y_true, y_score):
        assert len(y_true) == len(y_score)
        #assert len(self.class_names) == y_score.shape[1]
        macro_roc_auc = roc_auc_score(y_true, y_score, average="macro")
        weighted_roc_auc = roc_auc_score(y_true, y_score, average="weighted")
        print("ROC AUC scores:\n{:.6f} (macro),\n{:.6f} (weighted by prevalence)"
              .format(macro_roc_auc, weighted_roc_auc))
        
        p = self.class_names[1]
        n = self.class_names[0]
        
        fpr = {}
        tpr = {}
        roc_auc = {}
        
        
        fpr[p], tpr[p], _ = roc_curve(y_true, y_score, pos_label=self.labels[1])
        roc_auc[p] = auc(fpr[p], tpr[p])
        fpr[n], tpr[n], _ = roc_curve(y_true, 1-y_score, pos_label=self.labels[0])
        roc_auc[n] = auc(fpr[n], tpr[n])
        
        
        xlim1, ylim1 = [-0.02, 1.0], [0.0, 1.02]
        
        plt.figure(figsize=(10,8), facecolor='white')
        plt.grid(True, which='major', axis='both')
        
        plt.plot(fpr[p], tpr[p], lw=1.0, linestyle='--', label='class {0} (area = {1:0.4f})'.format(p, roc_auc[p]))
        plt.plot(fpr[n], tpr[n], lw=1.0, linestyle='--', label='class {0} (area = {1:0.4f})'.format(n, roc_auc[n]))
        

        plt.plot([0, 1], [0, 1], color='darkslategray', lw=1.0, linestyle='--')
        plt.xlim(xlim1)
        plt.ylim(ylim1)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves of multi-class')
        plt.legend(loc="lower right")
        if save_dir is not None:
            plt.savefig(save_dir, transparent=False, dpi=80, bbox_inches="tight")
        plt.show()
        
        
        '''
        
        
        
        
        fpr = {}
        tpr = {}
        roc_auc = {}
        y_true_one_hot = np.zeros((y_score.shape[0], len(self.labels)))
        for i, j in enumerate(y_true):
            y_true_one_hot[i][int(j)] = 1
        for i, j in enumerate(self.class_names):
            fpr[j], tpr[j], _ = roc_curve(y_true_one_hot[:,i], y_score[:,i])
            roc_auc[j] = auc(fpr[j], tpr[j])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true_one_hot.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # Compute macro-average ROC curve and ROC area
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in self.class_names]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in self.class_names:
            mean_tpr += scipy.interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= len(self.class_names)

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        xlim1, ylim1 = [-0.02, 1.0], [0.0, 1.02]
        xlim2, ylim2 = [-0.01, 0.5], [0.5, 1.01]
        if self.pipeline.saver.save_test_report:
            save_dir = os.path.join(self.pipeline.saver.execute_save_dir, self.pipeline.file_name_manager('report roc'))
            self.plot_roc_curves(fpr, tpr, roc_auc, figsize=self.roc_figsize, xlim=xlim1, ylim=ylim1, save_dir = save_dir)
            self.plot_roc_curves(fpr, tpr, roc_auc, figsize=self.roc_figsize, xlim=xlim2, ylim=ylim2, save_dir = save_dir[:-4]+' enlarge' + save_dir[-4:])  # enlarge version
        else:
            self.plot_roc_curves(fpr, tpr, roc_auc, figsize=self.roc_figsize, xlim=xlim1, ylim=ylim1)
            self.plot_roc_curves(fpr, tpr, roc_auc, figsize=self.roc_figsize, xlim=xlim2, ylim=ylim2)  # enlarge version
        
    def plot_roc_curves(self, fpr, tpr, roc_auc, plot_all_classes=True, figsize=(12, 10), xlim=[-0.02, 1.0], ylim=[0.0, 1.02], save_dir=None):
        # Plot all ROC curves
        plt.figure(figsize=figsize, facecolor='white')
        plt.grid(True, which='major', axis='both')
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average (area = {0:0.4f})'.format(roc_auc["micro"]),
                 linestyle='-', linewidth=1.5)

        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average (area = {0:0.4f})'
                       ''.format(roc_auc["macro"]),
                 linestyle='-', linewidth=1.5)
        if plot_all_classes:
            for i in self.class_names:
                plt.plot(fpr[i], tpr[i], lw=1.0, linestyle='--',
                         label='class {0} (area = {1:0.4f})'.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], color='darkslategray', lw=1.0, linestyle='--')
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves of multi-class')
        plt.legend(loc="lower right")
        if save_dir is not None:
            plt.savefig(save_dir, transparent=False, dpi=80, bbox_inches="tight")
        plt.show()
        
        
        '''
        
import pandas as pd
import torch
import time
import numpy as np
import warnings
from gensim.models.word2vec import Word2Vec
from model import BatchProgramCC
from sklearn.metrics import precision_recall_fscore_support
    

    
def get_device():
    if torch.cuda.is_available():
        if torch.cuda.get_device_name(0) == 'GeForce GT 730':
            device = 'cpu'
        else:
            device = 'cuda'
    else:
        device = 'cpu'
    return torch.device(device)

device = get_device()

save_dir = '---- java ccd model/'


from torch import nn
from torch.nn import functional as F
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, enclidean_distance, label):
        #enclidean_distance = F.pairwise_distance(output[0], output[1])
        loss_contrastive = torch.mean(label*torch.pow(enclidean_distance, 2) +
                                     (1-label) * torch.pow(torch.clamp(self.margin - enclidean_distance, min=0.0), 2))
        return loss_contrastive
    
    

basic_config = {'executor': ASTNN_Executor(),
                'progressbar': Progbar(dynamic = False),
                'reporter': Reporter_c2(labels = [0, 1], 
                                        need_confusion_matrix = True,
                                        output_to_score_fun = None, 
                                        report_interval = 2,
                                        show_train_report = True,
                                        summary_fun = None),
                'saver': Saver(save_dir = save_dir,
                               save_meta_file = True,
                               save_ckpt_model = True, 
                               save_val_model = True, 
                               save_final_model = True,
                               save_final_optim = True,
                               save_interval = 1, 
                               test_model_use = 'final', 
                               save_history = True,
                               save_train_report = True,
                               save_test_report = True),
                'file_name_manager': FileNameManager(),
                'device': device,
                'criterion': ContrastiveLoss(),
               }

pipeline = DLpipeline(**basic_config)