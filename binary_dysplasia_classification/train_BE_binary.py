#Binary dysplasia classification of pathology reports from CUIMC 

import numpy as np
import pandas as pd
import torch
import random
import math 
from transformers import AutoTokenizer, AutoModel, BertForSequenceClassification, RobertaForSequenceClassification
from transformers import BigBirdForSequenceClassification
from transformers import TrainingArguments, Trainer
import time, os, pickle, glob, shutil, sys 
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score, fbeta_score, recall_score, precision_score
from scipy.special import softmax
from collections import Counter 

start = time.time()
curr_datetime = datetime.now().strftime('%m-%d-%Y_%Hh-%Mm')
prefix=sys.argv[1] 

#Input Parameters
model_used = sys.argv[3] 
batch_size = int(sys.argv[4]) 
max_tokens = int(sys.argv[5]) 
lr_type = sys.argv[6] 
if lr_type == 'lower_lr':
    learning_rate = .000005
elif lr_type == 'default_lr':
    learning_rate = .00005
else: 
    print('Error: Learning Rate Type not recognized')

optim_method = sys.argv[7]
if optim_method == 'roc':
    optim_type = 'eval_roc_auc'
elif optim_method == 'f2_beta':
    optim_type = 'eval_f2_beta'
else:
    print('Error: optim_type not recognized')

#Input Data 
input_suffix = sys.argv[9] 
input_dir = '../../../Target_Selection/'+input_suffix+'/' #gi_binary_tvt_data
train_data = pd.read_csv(input_dir+'gi_train_rs'+str(seed)+'.csv')
val_data = pd.read_csv(input_dir+'gi_val_rs'+str(seed)+'.csv')
test_data = pd.read_csv(input_dir+'gi_test_orig_val.csv')
train_data['label'] = train_data['binary_dyplasia']
val_data['label'] = val_data['binary_dyplasia']
test_data['label'] = test_data['binary_dyplasia']

root_dir = 'model_output/' + prefix + '_output/'  
model_output_dir = root_dir + prefix +'_rs'+str(seed)+'_'+model_used+'_'+str(batch_size)+'bsize'+'_'+str(max_tokens)+'max_tokens_'+lr_type+'_'+optim_method+'_optim_'+str(sys.argv[8]) +'e_' +curr_datetime+ '/'
best_model_evaluate_dir = model_output_dir + 'val_best_model_evaluate_tmp/'
for directory in [root_dir, model_output_dir, best_model_evaluate_dir]:
    os.makedirs(directory, exist_ok=True)
output_file = prefix+'_output_rs'+str(seed)+'.txt'
meta_df = pd.DataFrame({'conditions':[prefix]})
eval_metric = optim_type 
num_classes=2 #binary 

#Set seeds
seed=int(sys.argv[2])
np.random.seed(0) 
torch.manual_seed(0)
random.seed(0)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])

def compute_metrics(eval_pred):
    raw_pred, labels = eval_pred
    score_pred = softmax(raw_pred, axis=1)[:,1]
    binary_pred = np.argmax(raw_pred, axis=1)
    accuracy = accuracy_score(labels, binary_pred)
    roc = roc_auc_score(labels, score_pred) 
    f1 = f1_score(labels, binary_pred, average = 'macro')
    f1_weighted = f1_score(labels, binary_pred, average = 'weighted')
    auprc = average_precision_score(labels, score_pred)
    recall = recall_score(labels, binary_pred)
    precision = precision_score(labels, binary_pred)
#    f2_beta = (5* precision * recall) / (4 * precision + recall)
    f2_beta = fbeta_score(labels, binary_pred, beta=2, average='macro') 
    return {"accuracy": accuracy, "roc_auc": roc, "f1": f1, "f1_weighted":f1_weighted,
    'auprc':auprc, 'recall':recall, 'precision':precision,
    'f2_beta':f2_beta} 


#Model and Tokenizer
if model_used == 'clinicalbert':
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    model = BertForSequenceClassification.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", num_labels=num_classes)
    print('ClinicalBERT')
elif model_used == 'bigbird':
    tokenizer = AutoTokenizer.from_pretrained("yikuan8/Clinical-BigBird")
    model = BigBirdForSequenceClassification.from_pretrained("yikuan8/Clinical-BigBird", num_labels=num_classes)
    print('Bigbird model number of parameters:',model.num_parameters())
else:
    print('Model not recognized')


#Input Data
X_train = list(train_data['path_text'])
y_train = list(train_data['label'])

X_val = list(val_data['path_text'])
y_val = list(val_data['label'])

X_trainval = X_train + X_val
y_trainval = y_train + y_val

X_test = list(test_data['path_text'])
y_test = list(test_data['label'])

X_train_tokenized = tokenizer(X_train, padding=True, truncation=True,max_length=max_tokens)
X_val_tokenized = tokenizer(X_val, padding=True, truncation=True,max_length=max_tokens)
X_test_tokenized=tokenizer(X_test, padding=True, truncation=True, max_length=max_tokens)

X_trainval_tokenized = tokenizer(X_trainval, padding=True, truncation=True,max_length=max_tokens)
trainval_dataset = Dataset(X_trainval_tokenized, y_trainval)
train_dataset = Dataset(X_train_tokenized, y_train)
val_dataset = Dataset(X_val_tokenized, y_val)
test_dataset = Dataset(X_test_tokenized,y_test)

#Record Train/Validation Split across Classes 
print('train:', Counter(y_train).most_common())
print('val:', Counter(y_val).most_common())
pos_prop_train = Counter(y_train).most_common()
pos_prop_val = Counter(y_val).most_common()

#steps dependent on train set size
logging_steps = math.ceil((len(y_train) / batch_size)/5) #logs 5x/epoch
save_steps = math.ceil((len(y_train) / batch_size)/10) #10x/epoch
eval_steps = math.ceil((len(y_train) / batch_size)/10) #10x/epoch
print(len(y_train),logging_steps, save_steps, eval_steps)

#Training Parameters
training_args_dict = {'save_strategy':'steps',
                      'save_steps':save_steps, 
                      'save_total_limit':2,
                      'num_train_epochs':int(sys.argv[8]), 
                      'logging_steps':logging_steps, 
                      'per_device_train_batch_size':batch_size,
                      'per_device_eval_batch_size':16, 
                      'evaluation_strategy':'steps',
                      'eval_steps':eval_steps, 
                      'load_best_model_at_end':True,
                      'metric_for_best_model':eval_metric,
                      'learning_rate':learning_rate,
                      'learning_rate_type':lr_type,
                      'num_classes':num_classes,
                      'optim_type':optim_type} 

#Fine-tuning 
#Save best model checkpoint saved (even if other models come after it); delete sub-par models during training process
training_args = TrainingArguments(model_output_dir, 
                                  report_to=None,
                                  seed=0, 
                                  save_strategy = training_args_dict['save_strategy'],
                                  save_steps = training_args_dict['save_steps'], #Change depending on the size of the dataset
                                  save_total_limit = training_args_dict['save_total_limit'], #Deletes all but last X checkpoints - sequentially/chronologically
                                  num_train_epochs = training_args_dict['num_train_epochs'], 
                                  logging_steps = training_args_dict['logging_steps'], #logs training_loss every X steps 
                                  per_device_train_batch_size = training_args_dict['per_device_train_batch_size'], #default = 8
				                          per_device_eval_batch_size = training_args_dict['per_device_eval_batch_size'], #default = 8
				                          evaluation_strategy = training_args_dict['evaluation_strategy'],
                                  eval_steps = training_args_dict['eval_steps'],
                                  load_best_model_at_end = training_args_dict['load_best_model_at_end'], #default = based on training loss, unless specified below 
                                  metric_for_best_model = training_args_dict['metric_for_best_model'],
                                  learning_rate = training_args_dict['learning_rate'])   

print(training_args_dict) #record all training parameters in output file for later reference

trainer = Trainer(model=model,
                  args=training_args,
                  train_dataset=train_dataset,
                  eval_dataset=val_dataset, 
                  compute_metrics=compute_metrics)

#Train
trainer.train()

#Track Run-Time (seconds)
end = time.time()
runtime = round((end - start)/60,3)
print('\nElapsed Time: ', runtime, 'Minutes')

#Save Performance Data
history_list = trainer.state.log_history
pickle.dump(history_list, open(model_output_dir+'state_log_history.p','wb'))

#Training - Performance 
train_info = [a for a in history_list if 'loss' in a.keys()]
train_info_dict = {'step':[a['step'] for a in train_info],
                  'training_loss':[a['loss'] for a in train_info],
                  'epoch':[a['epoch'] for a in train_info],
                  'learning_rate':[a['learning_rate'] for a in train_info]}
train_info_df=pd.DataFrame(train_info_dict)

#Evaluation - Performance
e_info = [a for a in history_list if 'eval_loss' in a.keys()]
e_info_dict = {key:[a[key] for a in e_info] for key in e_info[0].keys()}
e_info_df=pd.DataFrame(e_info_dict)
train_info_df.to_csv(model_output_dir+'train_history.csv',index=False)
e_info_df.to_csv(model_output_dir+'eval_history.csv',index=False)
history_df = e_info_df.merge(train_info_df, on=['step','epoch'],how='outer')
history_df.sort_values(by='step',inplace=True)
history_df.to_csv(model_output_dir+'full_history.csv',index=False)

#Plot Performance

#step, training_loss plot 
train_steps = [a['step'] for a in trainer.state.log_history if 'loss' in a.keys()]
train_loss = [a['loss'] for a in trainer.state.log_history if 'loss' in a.keys()]
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.plot(train_steps, train_loss)
plt.savefig(model_output_dir+'training_loss.png',bbox_inches='tight',dpi=600, facecolor='w')
plt.close() 
#plt.show()

#plot eval loss 
e_steps = [a['step'] for a in trainer.state.log_history if 'eval_loss' in a.keys()]
e_loss = [a['eval_loss'] for a in trainer.state.log_history if 'eval_loss' in a.keys()]
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Loss during Training')
plt.plot(train_steps, train_loss, label='Train')
plt.plot(e_steps, e_loss, label='Validation')
plt.legend()
plt.savefig(model_output_dir+'validation_loss.png',bbox_inches='tight',dpi=600, facecolor='w')
plt.close() 

#Identify Best_model 
model_checkpoints = glob.glob(model_output_dir+'checkpoint*')
print(model_checkpoints) #Verify correct directory selected
checkpoint_steps = [int(a.split('-')[-1]) for a in model_checkpoints]
checkpoint_data = [a for a in history_list if ((eval_metric in a.keys()) and (a['step'] in checkpoint_steps))]
eval_full_list = [a[eval_metric] for a in checkpoint_data]
best_checkpoint_steps = [a['step'] for a in checkpoint_data if a[eval_metric] == max(eval_full_list)]

#If multiple checkpoints have the same eval_metric, choose the chronologically later one (trained on more data)
best_model_checkpoint = [a for a in model_checkpoints if str(max(best_checkpoint_steps)) == a.split('-')[-1]][0]
print(best_model_checkpoint)

#Save best_model metadata 
info_for_test_evaluation = {'best_model_checkpoint':best_model_checkpoint,
                            'model_output_dir':model_output_dir}
pickle.dump(info_for_test_evaluation, open(model_output_dir+'info_for_test_evaluation.p','wb')) 

#Print Best Model metrics 
print('Best Model Performance')

if model_used == 'clinicalbert':
    best_model = BertForSequenceClassification.from_pretrained(best_model_checkpoint, num_labels=num_classes, local_files_only=True)
    print('ClinicalBERT eval')
elif model_used == 'bigbird':
    best_model = BigBirdForSequenceClassification.from_pretrained(best_model_checkpoint, num_labels=num_classes, local_files_only=True)
    print('Bigbird eval')
else:
    print('Model not recognized')

best_trainer = Trainer(model=best_model, compute_metrics=compute_metrics,
    args=TrainingArguments(output_dir = best_model_evaluate_dir)) 

y_pred, _,_  = best_trainer.predict(Dataset(X_val_tokenized)) 

#Test best model on held-out test set 
print('Held-out test set evaluation')
test_roc = round(best_trainer.evaluate(test_dataset)['eval_roc_auc'],4)
print(test_roc)

print('\nAU-ROC ', round(au_roc,4))
print('validation evaluation')
vac = round(best_trainer.evaluate(val_dataset)['eval_accuracy'],4)
vroc=round(best_trainer.evaluate(val_dataset)['eval_roc_auc'],4)
vprc=round(best_trainer.evaluate(val_dataset)['eval_auprc'],4)
val_f1 = round(best_trainer.evaluate(val_dataset)['eval_f1'],4)
val_precision = round(best_trainer.evaluate(val_dataset)['eval_precision'],4)
val_recall = round(best_trainer.evaluate(val_dataset)['eval_recall'],4)
val_f2_beta = round(best_trainer.evaluate(val_dataset)['eval_f2_beta'],4)
print(vroc, vprc)

trainval_dataset
print('trainval set evaluation')
tv_ac = round(best_trainer.evaluate(trainval_dataset)['eval_accuracy'],4)
tv_roc = round(best_trainer.evaluate(trainval_dataset)['eval_roc_auc'],4)
tv_f1 = round(best_trainer.evaluate(trainval_dataset)['eval_f1'],4)
tv_precision = round(best_trainer.evaluate(trainval_dataset)['eval_precision'],4)
tv_recall = round(best_trainer.evaluate(trainval_dataset)['eval_recall'],4)
tv_f2_beta = round(best_trainer.evaluate(trainval_dataset)['eval_f2_beta'],4)

print('held-out test set evaluation')
test_ac = round(best_trainer.evaluate(test_dataset)['eval_accuracy'],4)
test_roc = round(best_trainer.evaluate(test_dataset)['eval_roc_auc'],4)
test_prc = round(best_trainer.evaluate(test_dataset)['eval_auprc'],4)
test_f1 = round(best_trainer.evaluate(test_dataset)['eval_f1'],4)
test_precision = round(best_trainer.evaluate(test_dataset)['eval_precision'],4)
test_recall = round(best_trainer.evaluate(test_dataset)['eval_recall'],4)
test_f2_beta = round(best_trainer.evaluate(test_dataset)['eval_f2_beta'],4)
print(test_roc, test_prc, test_f1)

#Save evaluation metrics in meta_df
meta_df['runtime_min'] = [runtime]
meta_df['best_model_steps'] = [best_model_checkpoint.split('-')[-1]]
meta_df['best_model_dir'] = [best_model_checkpoint]
meta_df['pos_prop_train'] = [pos_prop_train]
meta_df['pos_prop_val'] = [pos_prop_val]
meta_df['save_steps'] = [training_args_dict['save_steps']]
meta_df['num_train_epochs'] = [training_args_dict['num_train_epochs']]
meta_df['logging_steps'] = [training_args_dict['logging_steps']]
meta_df['per_device_train_batch_size'] = [training_args_dict['per_device_train_batch_size']]
meta_df['per_device_eval_batch_size'] = [training_args_dict['per_device_eval_batch_size']]
meta_df['eval_steps'] = [training_args_dict['eval_steps']]
meta_df['learning_rate_type'] = [lr_type]
meta_df['learning_rate'] = [learning_rate]
meta_df['val_accuracy_bestmodel'] = [vac]
meta_df['val_au_roc_bestmodel'] = [vroc]
meta_df['val_au_prc_bestmodel'] = [vprc]
meta_df['test_au_roc_bestmodel'] = [test_roc]
meta_df['test_au_prc_bestmodel'] = [test_prc]
meta_df['random_seed'] = [seed]
meta_df['f1_macro_val'] = [val_f1]
meta_df['f1_macro_test'] = [test_f1]
meta_df['f1_precision_test'] = [test_precision]
meta_df['f1_recall_test'] = [test_recall]
meta_df['f1_accuracy_test'] = [test_ac]
meta_df['num_classes']=[num_classes]
meta_df['optim_type'] = [optim_type]
meta_df['val_f1'] = [val_f1]
meta_df['val_recall'] = [val_recall]
meta_df['val_precision'] =[val_precision]
meta_df['tv_ac'] = [tv_ac]
meta_df['tv_roc'] = [tv_roc]
meta_df['tv_f1'] = [tv_f1]
meta_df['tv_precision'] = [tv_precision]
meta_df['tv_recall'] = [tv_recall]
meta_df['tv_f2_beta'] = [tv_f2_beta]
meta_df['val_f2_beta'] = [val_f2_beta]
meta_df['test_f2_beta'] = [test_f2_beta]

#Save meta_df to csv
meta_df.to_csv(model_output_dir+prefix+'_rs'+str(seed)+'_meta_df.csv',index=False)
meta_df.to_csv('output/'+prefix+'_rs'+str(seed)+'bs_'+str(batch_size)+'_LR'+lr_type+'_meta_df_'+optim_type+'.csv',index=False) 

#Delete remaining checkpoint (leaving best model intact)
non_best_model_checkpoint = [a for a in model_checkpoints if str(max(best_checkpoint_steps)) != a.split('-')[-1]][0]
shutil.rmtree(non_best_model_checkpoint)
print('Non-Best-Model checkpoint deleted')
