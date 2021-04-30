# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 08:18:19 2021

@author: Uni361004
"""

# Imports
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn; cudnn.benchmark = True

data_path = "C:\\Users\\emanu\\Desktop\\data"


with open("pdm.txt") as f:
    text = f.read()
    
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

for punct, token in token_dict.items():
    text = text.replace(punct, f' {token} ')
    
text_split_by_space = text.split()
words = [word for word in text_split_by_space if len(word) > 0]  # with len(word) > 0 the remaining spaces are removed
vocabulary = list(set(words)) # set removes duplicates
vocab_to_int = {word: i for i,word in enumerate(vocabulary)}
int_to_vocab = {i: word for i,word in enumerate(vocabulary)}
print(f"The length of the vocabulary (number of unique words) is: {len(vocabulary)}")

text_as_integers = [vocab_to_int[word] for word in text.split() if len(word) > 0]
print(f"The total number of words is: {len(text_as_integers)}")

num_paragraphs = len(text.split("|newline|  |newline|")) # paragraphs are delimited by \n space \n
print(f"The average paragraph length is: {round(len(text_as_integers)/num_paragraphs)}")  
    
batch_seq_len = 16 # the model will be fed with batches with 16 words
paragraph_length = 224 # set a fixed value for paragraph length, multiple of 16
print(f"The number of mini-sequences is {round(paragraph_length/batch_seq_len)}")    
    
    
def get_batches(text_as_integers, paragraph_length, batch_size, batch_seq_len):

    num_paragraphs = len(text_as_integers)//paragraph_length
    text_targets = text_as_integers[1:] + [text_as_integers[0]]
    paragraph_inputs = [text_as_integers[i*paragraph_length:(i+1)*paragraph_length] for i in range(num_paragraphs)]
    paragraph_targets = [text_targets[i*paragraph_length:(i+1)*paragraph_length] for i in range(num_paragraphs)]
    
    # Split paragraphs into mini-sequences of length batch_seq_len
    num_mini_sequences = paragraph_length//batch_seq_len
    paragraph_inputs = [[paragraph[i*batch_seq_len:(i+1)*batch_seq_len] for i in range(num_mini_sequences)] for paragraph in paragraph_inputs]
    paragraph_targets = [[paragraph[i*batch_seq_len:(i+1)*batch_seq_len] for i in range(num_mini_sequences)] for paragraph in paragraph_targets]

    num_batch_groups = len(paragraph_inputs)//batch_size
    batches = []
    
    for i in range(num_batch_groups):

        group_paragraph_inputs = paragraph_inputs[i*batch_size:(i+1)*batch_size]
        group_paragraph_targets = paragraph_targets[i*batch_size:(i+1)*batch_size]

        for j in range(num_mini_sequences):
            reset_state = (j == 0)
            batch_inputs = torch.LongTensor([group_paragraph_inputs[k][j] for k in range(batch_size)])
            batch_targets = torch.LongTensor([group_paragraph_targets[k][j] for k in range(batch_size)])
            batches.append((reset_state, batch_inputs, batch_targets))

    return batches    
    
    
batch_size = 8
batches = get_batches(text_as_integers, paragraph_length, batch_size, batch_seq_len)
    
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


def generate_paragraph(model, seq_len, paragraph_start):
    # Convert punctuaction in paragraph start
    for punct, token in token_dict.items():
        paragraph_start = paragraph_start.replace(punct, f' {token} ')
        
    # Convert paragraph start text to ints
    paragraph_start = [vocab_to_int[word] for word in paragraph_start.split(" ") if len(word) > 0]
    
    # Initialize output words/tokens
    paragraph = paragraph_start[:]
    
    # Convert paragraph start to tensor (BxS = 1xS)
    paragraph_start = torch.LongTensor(paragraph_start).unsqueeze_(0)
    
    # Process paragraph start and generate the rest of the paragraph
    model.eval()
    model.reset_state()
    input = paragraph_start
    for i in range(seq_len - paragraph_start.size(1) + 1): # we include paragraph_start as one of the generation steps
        # Copy input to device
        input = input.to(dev)
        # Pass to model
        output = model(input) # 1xSxV
        # Convert to word indexes
        words = output.max(2)[1] # 1xS
        words = words[0] # S
        # Add each word to paragraph
        for j in range(words.size(0)):
            paragraph.append(words[j].item())
        # Prepare next input
        input = torch.LongTensor([words[-1]]).unsqueeze(0) # 1xS = 1x1
        
    # Convert word indexes to text
    paragraph = ' '.join([int_to_vocab[x] for x in paragraph])
    # Convert punctuation tokens to symbols
    for punct,token in token_dict.items():
        paragraph = paragraph.replace(f"{token}", punct)
        
    return paragraph
    
    
generate_paragraph(model, 20, "Data")

# Create optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) 
       
# Initialize training history
loss_history = []
# Start training
for epoch in range(200):
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
    shift_point = random.randint(1, len(text_as_integers)-1)
    text_as_integers = text_as_integers[:shift_point] + text_as_integers[shift_point:]
    batches = get_batches(text_as_integers, paragraph_length, batch_size, batch_seq_len)
    # Epoch end - compute average epoch loss
    avg_loss = epoch_loss_sum/epoch_loss_cnt
    print(f"Epoch: {epoch+1}, loss: {epoch_loss_sum/epoch_loss_cnt:.4f}")
    #print("Test sample:")
    #print("---------------------------------------------------------------")
    #print(generate_paragraph(model, paragraph_length, "The"))
    #print("---------------------------------------------------------------")
    # Add to histories
    loss_history.append(avg_loss)
    
    
    
    
    
    
    
    