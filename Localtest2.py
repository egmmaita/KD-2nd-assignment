# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 08:18:19 2021

@author: Uni361004
"""


import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn; cudnn.benchmark = True
import os
from pathlib import Path


##UNIFIED DATA TEST##

token_dict = {".": "|fullstop|",
              ",": "|comma|",
              ":":"|semicolumn|",
              "/" : "|backlash|",
              "\'" : "|accent|",
            #   "-" : "|dash|",
            #  "–" : "|dash2|",
              "=" : "|equal|",
              "%" : "|percentage|",
            #  "“": "|quoteopen|",
              # "”" : "|quoteclosed|",
              ";": "|semicolon|",
              "!": "|exclamation|",
              "?": "|question|",
              "(": "|leftparen|",
              ")": "|rightparen|",
              "--": "|dash|",
              "\n": "|newline|",
              "•" : "|listsymbol|",
              "◦" : "|listsymbol2|"
}

data_path = Path("C:\\Users\\emanu\\Desktop\\data")
folders = sorted(os.listdir(data_path))
first_article = True
articles_length = []
words = []
for folder_name in folders:
    folder_path = data_path/folder_name
    folder_articles = sorted(os.listdir(folder_path))
    for article_name in folder_articles:
        
        article_path = folder_path/article_name
        with open(article_path) as f:
            article = f.read()
        
        for punct, token in token_dict.items():   
            article = article.replace(punct, f' {token} ')
            
        articles_length.append(len(article))
        
        words_in_article = [word for word in article.split() if len(word) > 0]  # with len(word) > 0 the remaining spaces are removed
        words += words_in_article

        if first_article:
            data = article
            first_article=False
        else:
            data += "\n\n\n\n" + article


vocabulary = list(set(words)) # set removes duplicates
vocab_to_int = {word: i for i,word in enumerate(vocabulary)}
int_to_vocab = {i: word for i,word in enumerate(vocabulary)}
print(f"The length of the vocabulary (number of unique words) is: {len(vocabulary)}")

numerical_data = [vocab_to_int[word] for word in data.split() if len(word) > 0]


print(f"The average article length is: {round(sum(articles_length)/len(articles_length))}")  
    
batch_seq_len = 16 # the model will be fed with batches with 16 words
article_length = 2800 # set a fixed value for article length, multiple of batch_seq_len
print(f"The number of mini-sequences is {round(article_length/batch_seq_len)}")    
    

# data_path = Path("C:\\Users\\emanu\\Desktop\\data")
# folders = sorted(os.listdir(data_path))
# data_raw = []
# first_pass = True
# for folder in folders:
#       folder_dir = data_path/folder
#       folder_articles = sorted(os.listdir(folder_dir))
#       for file in folder_articles:
#           file_dir = folder_dir/file
#           with open(file_dir) as f:
#               article = f.read()
#           if first_pass:
#               unified_data = article
#               first_pass = False
#           else:
#              unified_data = unified_data + "\n\n\n\n" + article
#           data_raw.append(article)

# unified_data[:5000]
# print(f"Number of articles: {len(data_raw)}")
# #------------------------------------------

    
# token_dict = {".": "|fullstop|",
#               ",": "|comma|",
#               ":":"|semicolumn|",
#               "/" : "|backlash|",
#               "\'" : "|accent|",
#            #   "-" : "|dash|",
#             #  "–" : "|dash2|",
#               "=" : "|equal|",
#               "%" : "|percentage|",
#             #  "“": "|quoteopen|",
#              # "”" : "|quoteclosed|",
#               ";": "|semicolon|",
#               "!": "|exclamation|",
#               "?": "|question|",
#               "(": "|leftparen|",
#               ")": "|rightparen|",
#               "--": "|dash|",
#               "\n": "|newline|",
#               "•" : "|listsymbol|",
#               "◦" : "|listsymbol2|"
# }

# words = []
# data = []
# articles_length = []
# for article in data_raw:
#     for punct, token in token_dict.items():
#         article = article.replace(punct, f' {token} ')
#     data.append(article)
#     articles_length.append(len(article))
#     words_in_article = [word for word in article.split() if len(word) > 0]  # with len(word) > 0 the remaining spaces are removed
#     words += words_in_article

# print(f"The total number of words is: {len(words)}")

# vocabulary = list(set(words)) # set removes duplicates
# vocab_to_int = {word: i for i,word in enumerate(vocabulary)}
# int_to_vocab = {i: word for i,word in enumerate(vocabulary)}
# print(f"The length of the vocabulary (number of unique words) is: {len(vocabulary)}")

# numerical_data = []
# for article in data:
#     article_as_numeric = [vocab_to_int[word] for word in article.split() if len(word) > 0]
#     numerical_data.append(article_as_numeric)

# print(f"The average article length is: {round(sum(articles_length)/len(articles_length))}")  
    
# batch_seq_len = 16 # the model will be fed with batches with 16 words
# article_length = 2800 # set a fixed value for article length, multiple of batch_seq_len
# print(f"The number of mini-sequences is {round(article_length/batch_seq_len)}")    
    

def get_batches(numerical_data, article_length, batch_size, batch_seq_len):

    num_articles = len(numerical_data)//article_length
    text_targets = numerical_data[1:] + [numerical_data[0]]
    article_inputs = [numerical_data[i*article_length:(i+1)*article_length] for i in range(num_articles)]
    article_targets = [text_targets[i*article_length:(i+1)*article_length] for i in range(num_articles)]
    
    # Split articles into mini-sequences of length batch_seq_len
    num_mini_sequences = article_length//batch_seq_len
    article_inputs = [[article[i*batch_seq_len:(i+1)*batch_seq_len] for i in range(num_mini_sequences)] for article in article_inputs]
    article_targets = [[article[i*batch_seq_len:(i+1)*batch_seq_len] for i in range(num_mini_sequences)] for article in article_targets]

    num_batch_groups = len(article_inputs)//batch_size
    batches = []
    
    for i in range(num_batch_groups):

        group_article_inputs = article_inputs[i*batch_size:(i+1)*batch_size]
        group_article_targets = article_targets[i*batch_size:(i+1)*batch_size]

        for j in range(num_mini_sequences):
            reset_state = (j == 0)
            batch_inputs = torch.LongTensor([group_article_inputs[k][j] for k in range(batch_size)])
            batch_targets = torch.LongTensor([group_article_targets[k][j] for k in range(batch_size)])
            batches.append((reset_state, batch_inputs, batch_targets))

    return batches    
    
    
batch_size = 8
batches = get_batches(numerical_data, article_length, batch_size, batch_seq_len)

 # Define model
class Model(nn.Module):
    

    def __init__(self, num_words, embed_size, rnn_size):

        super().__init__()
        # Store needed attributes
        self.rnn_size = rnn_size  #hidden state size, i.e. number of neurons
        self.state = None
        # Define modules
        self.embedding = nn.Embedding(num_words, embed_size)
        self.rnn = nn.LSTM(embed_size, rnn_size, batch_first=True)
        self.decoder = nn.Linear(rnn_size, num_words)
        # Flags
        self.reset_next_state = False   
        
    def reset_state(self):
        # Mark next state to be re-initialized
        self.reset_next_state = True
        
    def forward(self, x):
        # Check state reset
        if self.reset_next_state:
            # Initialize state (num_layers x batch_size x rnn_size)
            self.state = (
                x.new_zeros(1, x.size(0), self.rnn_size).float(), 
                x.new_zeros(1, x.size(0), self.rnn_size).float())
            # Clear flag
            self.reset_next_state = False
        # Embed data
        x = self.embedding(x)
        # Process RNN
        state = self.state if self.state is not None else None
        x, state = self.rnn(x, state)
        self.state = (state[0].data, state[1].data)
        # Compute outputs
        x = self.decoder(x)
        return x   
    

embed_size = 512
num_words = len(vocabulary)
rnn_size = 1024

model = Model(num_words, embed_size, rnn_size)    
    
dev = torch.device("cuda") if torch.cuda.is_available else torch.device("cpu") 
    
model = model.to(dev)   


def generate_article(model, seq_len, article_start):
    # Convert punctuaction in article start
    for punct, token in token_dict.items():
        article_start = article_start.replace(punct, f' {token} ')
        
    # Convert article start text to ints
    article_start = [vocab_to_int[word] for word in article_start.split(" ") if len(word) > 0]
    
    # Initialize output words/tokens
    article = article_start[:]
    
    # Convert article start to tensor (BxS = 1xS)
    article_start = torch.LongTensor(article_start).unsqueeze_(0)
    
    # Process article start and generate the rest of the article
    model.eval()
    model.reset_state()
    input = article_start
    for i in range(seq_len - article_start.size(1) + 1): # we include article_start as one of the generation steps
        # Copy input to device
        input = input.to(dev)
        # Pass to model
        output = model(input) # 1xSxV
        # Convert to word indexes
        words = output.max(2)[1] # 1xS
        words = words[0] # S
        # Add each word to article
        for j in range(words.size(0)):
            article.append(words[j].item())
        # Prepare next input
        input = torch.LongTensor([words[-1]]).unsqueeze(0) # 1xS = 1x1
        
    # Convert word indexes to text
    article = ' '.join([int_to_vocab[x] for x in article])
    # Convert punctuation tokens to symbols
    for punct,token in token_dict.items():
        article = article.replace(f"{token}", punct)
        
    return article
    
    
generate_article(model, 20, "Data")

# Create optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) 
       
# Initialize training history
loss_history = []
# Start training
for epoch in range(4):
    # Initialize accumulators for computing average loss/accuracy
    epoch_loss_sum = 0
    epoch_loss_cnt = 0
    # Set network mode
    model.train()
    # Process all batches
    for i,batch in enumerate(batches):
        # Parse batch
        reset_state, input, target = batch
        # Check reset state
        if reset_state:
            model.reset_state()
        # Move to device
        input = input.to(dev)
        target = target.to(dev)
        # Forward
        output = model(input)
        # Compute loss
        output = output.view(-1, num_words)
        target = target.view(-1)
        loss = F.cross_entropy(output, target)
        # Update loss sum
        epoch_loss_sum += loss.item()
        epoch_loss_cnt += 1
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Shift sequence and recompute batches
    shift_point = random.randint(1, len(numerical_data)-1)
    numerical_data = numerical_data[:shift_point] + numerical_data[shift_point:]
    batches = get_batches(numerical_data, article_length, batch_size, batch_seq_len)
    # Epoch end - compute average epoch loss
    avg_loss = epoch_loss_sum/epoch_loss_cnt
    print(f"Epoch: {epoch+1}, loss: {epoch_loss_sum/epoch_loss_cnt:.4f}")
    print("Test sample:")
    print("---------------------------------------------------------------")
    print(generate_article(model, 15, "The"))
    print("---------------------------------------------------------------")
    # Add to histories
    loss_history.append(avg_loss)
    
    
    
    
    
    
    
    