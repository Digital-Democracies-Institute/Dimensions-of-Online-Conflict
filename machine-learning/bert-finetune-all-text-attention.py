import torch.nn as nn 
import torch 
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, f1_score
from transformers import BertForSequenceClassification, BertTokenizerFast, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import BertPreTrainedModel, BertModel
from transformers import BertConfig
from transformers import AutoModel

import random

import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="4"


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

df = pd.read_csv('train_data1.csv')


# Split the dataset into train and validation sets
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['is_there_conflict'])

# Initialize the tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# Tokenize the dataset
train_encodings = tokenizer(train_df['all_text'].apply(str).tolist(), truncation=True, padding=True, max_length=256)
val_encodings = tokenizer(val_df['all_text'].apply(str).tolist(), truncation=True, padding=True, max_length=256)

# Create torch datasets
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, additional_features, labels):
        self.encodings = encodings
        self.additional_features = additional_features
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['additional_features'] = torch.tensor(self.additional_features[idx], dtype=torch.float32)
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)



hidden_size = 768
embedding_size = 200
beta = 0.001
dropout_prob = 0.2

class AttnGating(nn.Module):
  def __init__(self):
    super(AttnGating, self).__init__()


    # number of the features
    self.linear = nn.Linear(2, embedding_size)
    self.relu = nn.ReLU(inplace=True)

    self.weight_addition_W1 = nn.Parameter(torch.Tensor(hidden_size+embedding_size, hidden_size))
    self.weight_addition_W2 = nn.Parameter(torch.Tensor(embedding_size, hidden_size))


    nn.init.uniform_(self.weight_addition_W1, -0.1, 0.1)
    nn.init.uniform_(self.weight_addition_W2, -0.1, 0.1)

    self.LayerNorm = nn.LayerNorm(hidden_size)
    self.dropout = nn.Dropout(dropout_prob)

  def forward(self, embeddings, additional_features):

     # Project additional features into vectors with comparable size
     additional_features = self.linear(additional_features)

     projected_features = additional_features.repeat(256, 1, 1) # (max_length, bs, feature_embedding_size) 
     projected_features = projected_features.permute(1, 0, 2) # (bs, max_length, feature_embedding_size)




     # Concatnate word and linguistic representations  
     features_combine = torch.cat((projected_features, embeddings), axis=2) # (bs, max_length, 768+feature_embedding_size)


     g_feature = self.relu(torch.matmul(features_combine, self.weight_addition_W1))

     # Attention gating
     H = torch.mul(g_feature, torch.matmul(projected_features, self.weight_addition_W2))
     alfa = min(beta * (torch.norm(embeddings)/torch.norm(H)), 1)
     E = torch.add(torch.mul(alfa, H), embeddings)

     # Layer normalization and dropout 
     embedding_output = self.dropout(self.LayerNorm(E)) 

     return embedding_output

class CustomBertForSequenceClassification(BertForSequenceClassification):
    def __init__(self, config, num_additional_features):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.embedding_bert = BertModel.from_pretrained("bert-base-uncased")


        #self.embedding_bert = AutoModel.from_pretrained("bert-base-uncased", output_hidden_states=True)
        self.attn_gate = AttnGating()


        self.dropout_attention = nn.Dropout(0.2)
        #self.relu = nn.ReLU(inplace=True)

        self.classifier = nn.Linear(768, 2)


        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        additional_features=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs_embeds = self.embedding_bert(input_ids, output_hidden_states=True)

        hidden_states = outputs_embeds.hidden_states

        bert_embeds = hidden_states[0]


        combined_embeds = self.attn_gate(bert_embeds, additional_features)

        outputs = self.bert(
            input_ids=None,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=combined_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output  = outputs.last_hidden_state

        x = sequence_output[:, 0, :]
        x = self.dropout_attention(x)
        x = torch.tanh(x)
        x = self.dropout_attention(x)

        logits= self.classifier(x)


        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return (loss, logits)
        else:
            return logits        
additional_features = df[['overall_toxicity','has_bidirectionality']].values  

train_additional_features, val_additional_features = train_test_split(additional_features, test_size=0.2, random_state=42, stratify=df['is_there_conflict'])
train_dataset = CustomDataset(train_encodings, train_additional_features, train_df['is_there_conflict'].values)
val_dataset = CustomDataset(val_encodings, val_additional_features, val_df['is_there_conflict'].values)

# unused parameter
num_additional_features = 2

config = BertConfig.from_pretrained('bert-base-uncased', num_labels=2)
set_seed()
model = CustomBertForSequenceClassification.from_pretrained('bert-base-uncased', config=config, num_additional_features=num_additional_features)


# Calculate class weights
class_counts = df['is_there_conflict'].value_counts().sort_index().tolist()
class_weights = [sum(class_counts) / count for count in class_counts]
weights = torch.tensor(class_weights).to('cuda')

# Set up the training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=4,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    logging_dir='./logs',
    logging_steps=10,
    save_steps=500,
    save_total_limit=3,
    evaluation_strategy="epoch",
    weight_decay=0.01,
    #learning_rate= 2.582866878848106e-05,
)

# Define the loss function with class weights
loss_fn = torch.nn.CrossEntropyLoss(weight=weights)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    #save_steps=2000, 
    #save_total_limit=3,
    #loss_fn=loss_fn,
)


print('model is training')

set_seed()
trainer.train()

print('model is done training')

# ------------------------------

print('testing')

set_seed()
test_df = pd.read_csv('test_data1.csv')
test_encodings = tokenizer(test_df['all_text'].apply(str).tolist(), truncation=True, padding=True, max_length=256)
test_additional_features = test_df[['overall_toxicity','has_bidirectionality']].values
test_labels = test_df['is_there_conflict'].tolist()

test_dataset = CustomDataset(test_encodings, test_additional_features, test_labels)

# Evaluate the model on the test set
eval_results = trainer.evaluate(test_dataset)
print("Evaluation results:", eval_results)

test_preds = trainer.predict(test_dataset)
probas = torch.softmax(torch.tensor(test_preds.predictions), dim=-1).numpy()
predicted_labels = np.argmax(probas, axis=1)

test_df['predicted_label'] = predicted_labels
test_df['probas'] = probas[:, 1]  # Assuming class 1 is the positive class
test_df.sort_values(by='probas', ascending=False).tail(200)

# test_df['probas']
#test_df['probas'].plot(kind='hist', bins=20)

from sklearn.metrics import confusion_matrix, f1_score

# Assuming you have the true labels and predicted labels in the following format:
true_labels = test_df['is_there_conflict'].tolist()
predicted_labels = test_df['predicted_label'].tolist()

# Calculate the confusion matrix
conf_mat = confusion_matrix(true_labels, predicted_labels)
print("Confusion Matrix:")
print(conf_mat)

# Calculate the F1 score
f1 = f1_score(true_labels, predicted_labels)
print("F1 Score:", f1)
