# -*- coding: utf-8 -*-
"""
Created on Mon May  3 13:57:49 2021

@author: Uni361004
"""
# %% RUN BEFORE ALL

# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"



# %% Parameters

genre1_path = "C:\\Users\\emanu\\Desktop\\2nd_ass\\content\\midi_data"
genre2_path = "C:\\Users\\emanu\\Desktop\\2nd_ass\\content\\midi_data_jazz"

BATCH_SIZE = 8
SEQUENCE_LENGTH = 100
EMBEDDING_SIZE = 256
RNN_SIZE = 1024

NUM_ITERS=1000




# %% Libraries

#import mitdeeplearning as mdl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
np.random.seed(0)
import time
import functools
#from IPython import display as ipythondisplay
from tqdm import tqdm
import os
from pathlib import Path
from matplotlib import pyplot as plt




# %% Get songs

def get_songs(genre_paths):
  songs = []
  for genre_path in genre_paths:
    fnames = sorted([x for x in os.listdir(genre_path) if x.endswith(".abc")])
    for fname in fnames:
       with open(Path(genre_path)/fname) as f:
         try:
            data = f.read()
            # Split song on different instruments (\nV:) and take only first
            text = data.split("\nV:")[0] + data.split("\nV:")[1]

            # Remove comments and useless lines
            list_selected_rows = []
            key_missing = True
            for row in text.split("\n"):
              if row[0]=="K" and key_missing:
                list_selected_rows.append(row)
                key_missing = False
              if row[0]!="K" and row[0] != "%" and row[0] != "w":  # remove eventual second key, lirics (w:...) and comments lines
                list_selected_rows.append(row)
            text = "\n".join(list_selected_rows)

            from_string = "from " + genre_path + "/"
            songs.append(text.replace(from_string, "").replace(".mid", ""))
         
         except:
            pass

  return songs


songs = get_songs([genre1_path, genre2_path])
print(f"The dataset consists of {len(songs)} songs.")




# %% Vocab and mappings

data = songs[0]  
all_other_songs = songs[1:]
for song in all_other_songs:
  data += "\n\n\n\n" + song
characters = [character for character in data if len(character)>0]
vocab = list(set(characters))
print(vocab)
char2idx = {character: idx for idx,character in enumerate(vocab)}
idx2char = {idx: character for idx,character in enumerate(vocab)}

# Extract keys
rows = ("\n".join(songs)).split("\n")
keys = sorted(list(set([x.split(" ")[0][2:] for x in rows if x[:2] == "K:"])))

# Create keys vocabulary
key2idx = {key: idx for idx,key in enumerate(keys)}
idx2key = {idx: key for idx,key in enumerate(keys)}




# %% Vectorize

def get_song_corpus_key(string):
  # Split song lines
  splitted_string = string.split("\n")

  # Get song header (title, time, key, ...)
  header = splitted_string[:6]
  key = [x.split(" ")[0][2:] for x in header if x[:2] == "K:"][0]

  # Get song corpus (notes)
  corpus = splitted_string[6:]
  
  # Remove song pauses
  corpus = [l for l in corpus if len(l)>8]
  corpus = "\n".join(corpus)

  return {"key": key, "song": corpus}



def vectorize_string(string):
  # Get song corpus and key
  song_dict = get_song_corpus_key(string)
  key = song_dict["key"]
  song = song_dict["song"]

  int_key = key2idx[key]
  vectorized_output = [char2idx[character] for character in song]

  song_dict = {"key": int_key, "song": vectorized_output}
  
  return song_dict

vec_songs = [vectorize_string(x) for x in songs]




# %% Batches

def get_batch(vec_songs, seq_length, batch_size):

  sampled_songs = np.random.choice(vec_songs, size = batch_size) 
  in_batch = []
  keys_batch = []
  tar_batch = []

  for song_dict in sampled_songs:
    key = song_dict["key"]
    song = song_dict["song"]  # -seq_length
    idx = np.random.randint(0,len(song)-SEQUENCE_LENGTH-1) # idx is a random index in range [0, len(song) - seq_length - 1]  choose safe range
    in_batch.append(song[idx: idx + seq_length])
    keys_batch.append(key)
    tar_batch.append(song[idx + 1 : idx + seq_length + 1])
  
  # in_batch = torch.LongTensor(in_batch[i] for i in range(batch_size)) ##??
  in_batch = torch.LongTensor(in_batch) ##??
  keys_batch = torch.LongTensor(keys_batch)
  keys_batch = keys_batch.unsqueeze(1)
  tar_batch = torch.LongTensor(tar_batch)


    # for i in range(num_batch_groups):
    #     # Get the scenes in this group
    #     group_scene_inputs = scene_inputs[i*batch_size:(i+1)*batch_size]
    #     group_scene_targets = scene_targets[i*batch_size:(i+1)*batch_size]
    #     # Build batches for each mini-sequence
    #     for j in range(num_mini_sequences):
    #         reset_state = (j == 0)
    #         batch_inputs = torch.LongTensor([group_scene_inputs[k][j] for k in range(batch_size)])
    #         batch_targets = torch.LongTensor([group_scene_targets[k][j] for k in range(batch_size)])
    #         batches.append((reset_state, batch_inputs, batch_targets))

  return in_batch, keys_batch, tar_batch

#test
in_batch, keys_batch, tar_batch = get_batch(vec_songs, SEQUENCE_LENGTH, BATCH_SIZE)
print(in_batch.shape)
print(keys_batch.shape)
print(tar_batch.shape)
print(in_batch[:4])
num_words = len(vocab)





# %% Model

class MusicGenerator(nn.Module):
    
    def __init__(self, vocab_size, num_keys, embed_size, rnn_size):
        super().__init__()

        self.rnn_size = rnn_size
        self.state = None
        self.embedding_words = nn.Embedding(vocab_size, embed_size)
        self.embedding_keys = nn.Embedding(num_keys, embed_size)
        self.rnn = nn.LSTM(embed_size*2, rnn_size, batch_first=True)
        self.decoder = nn.Linear(rnn_size, vocab_size)
        self.reset_next_state = False

    def reset_state(self):
        self.reset_next_state = True
    
    def forward(self, x, keys):
        # Implement forward pass (state reset, input embedding, ...) ########à
        if self.reset_next_state:
            # Initialize state (num_layers x batch_size x rnn_size) ######à
            self.state = (
                x.new_zeros(1, x.size(0), self.rnn_size).float(), 
                x.new_zeros(1, x.size(0), self.rnn_size).float()
                )
            self.reset_next_state = False
        
        x = self.embedding_words(x)
        keys = self.embedding_keys(keys)
        keys = keys.repeat(1, x.shape[1], 1)
        input = torch.cat((x,keys),dim=2)
        
        state = self.state if self.state is not None else None
        x, state = self.rnn(input, state)
        self.state = (state[0].data, state[1].data)

        x = self.decoder(x)
        return x
    
    
#tests
x = nn.Embedding(len(vocab), EMBEDDING_SIZE)(in_batch)
keyz = nn.Embedding(len(keys), EMBEDDING_SIZE)(keys_batch)
keyz = keyz.repeat(1, x.shape[1], 1)
input = torch.cat((x,keyz),dim=2)
state = (in_batch.new_zeros(1, x.size(0), RNN_SIZE).float(),
         in_batch.new_zeros(1, x.size(0), RNN_SIZE).float())
new_x, new_state = nn.LSTM(EMBEDDING_SIZE*2, RNN_SIZE, batch_first=True)(input, state)
# print(nn.Linear(RNN_SIZE, len(vocab))(new_x))
# print(nn.Linear(RNN_SIZE, len(vocab))(new_x).shape)
# print(len(vocab))
print(input.shape)


dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MusicGenerator(len(vocab), len(keys), EMBEDDING_SIZE, RNN_SIZE).to(dev)


in_batch, keys_batch, tar_batch = get_batch(vec_songs, SEQUENCE_LENGTH, BATCH_SIZE)
pred = model(in_batch.to(dev), keys_batch.to(dev))
print(pred.shape)

# Get next value prediction
val, index = pred[0, -1].max(0)
print(index, val)


def train(model, optim, criterion, dataset, num_words, dev=torch.device('cpu')):
  try:
    model.to(dev)
    loss_history = []
    model.train()

    for iter in range(NUM_ITERS):
      # reset model state 
      model.reset_state()
      in_batch, keys_batch, tar_batch = get_batch(dataset, SEQUENCE_LENGTH, BATCH_SIZE)
      in_batch = in_batch.to(dev)
      keys_batch = keys_batch.to(dev)
      tar_batch = tar_batch.to(dev)

      output = model(in_batch, keys_batch)

      output = output.view(-1, num_words)
      target = tar_batch.view(-1)
      loss = F.cross_entropy(output, target)
      loss_history.append(loss.item())

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      # Print info
      print(f"Iter: {iter+1}, loss: {loss_history[-1]:.4f}")
  
  except KeyboardInterrupt:
    print("Interrupted")
  
  finally: 
    plt.title("loss")
    plt.plot(loss_history, label="Train")
    plt.legend()
    plt.show()
    

####################################################################################
###################################################################################
###################################################################################

#     # Initialize training history
# loss_history = []
# # Start training
# for epoch in range(20):
#     # Initialize accumulators for computing average loss/accuracy
#     epoch_loss_sum = 0
#     epoch_loss_cnt = 0
#     # Set network mode
#     model.train()
#     # Process all batches
#     for i,batch in enumerate(batches):
#         # Parse batch
#         reset_state, input, target = batch
#         # Check reset state
#         if reset_state:
#             model.reset_state()
#         # Move to device
#         input = input.to(dev)
#         target = target.to(dev)
#         # Forward
#         output = model(input)
#         # Compute loss
#         output = output.view(-1, num_words)
#         target = target.view(-1)
#         loss = F.cross_entropy(output, target)
#         # Update loss sum
#         epoch_loss_sum += loss.item()
#         epoch_loss_cnt += 1
#         # Backward and optimize
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#     # Shift sequence and recompute batches
#     shift_point = random.randint(1, len(text_ints)-1)
#     text_ints = text_ints[:shift_point] + text_ints[shift_point:]
#     batches = get_batches(text_ints, scene_length, batch_size, batch_seq_len)
#     # Epoch end - compute average epoch loss
#     avg_loss = epoch_loss_sum/epoch_loss_cnt
#     print(f"Epoch: {epoch+1}, loss: {epoch_loss_sum/epoch_loss_cnt:.4f}")
#     print("Test sample:")
#     print("---------------------------------------------------------------")
#     print(generate_script(model, scene_length, "Moe_Szyslak:"))
#     print("---------------------------------------------------------------")
#     # Add to histories
#     loss_history.append(avg_loss)


# ###########################################################################
# #############################################################################
# ######################################################################
# # Initialize training history
# loss_history = []
# # Start training
# for epoch in range(20):
#     # Initialize accumulators for computing average loss/accuracy
#     epoch_loss_sum = 0
#     epoch_loss_cnt = 0
#     # Set network mode
#     model.train()
#     # Process all batches
#     for i,batch in enumerate(batches):
#         # Parse batch
#         reset_state, input, target = batch
#         # Check reset state
#         if reset_state:
#             model.reset_state()
#         # Move to device
#         input = input.to(dev)
#         target = target.to(dev)
#         # Forward
#         output = model(input)
#         # Compute loss
#         output = output.view(-1, num_words)
#         target = target.view(-1)
#         loss = F.cross_entropy(output, target)
#         # Update loss sum
#         epoch_loss_sum += loss.item()
#         epoch_loss_cnt += 1
#         # Backward and optimize
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#     # Shift sequence and recompute batches
#     shift_point = random.randint(1, len(text_ints)-1)
#     text_ints = text_ints[:shift_point] + text_ints[shift_point:]
#     batches = get_batches(text_ints, scene_length, batch_size, batch_seq_len)
#     # Epoch end - compute average epoch loss
#     avg_loss = epoch_loss_sum/epoch_loss_cnt
#     print(f"Epoch: {epoch+1}, loss: {epoch_loss_sum/epoch_loss_cnt:.4f}")
#     print("Test sample:")
#     print("---------------------------------------------------------------")
#     print(generate_script(model, scene_length, "Moe_Szyslak:"))
#     print("---------------------------------------------------------------")
#     # Add to histories
#     loss_history.append(avg_loss)



criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)




# %% Training

train(model, optimizer, criterion, vec_songs, len(vocab), dev=dev)





# %% Generation

def generate_text(model, start_string, key, generation_length=1000):
  with torch.no_grad():
      
    input_eval = [char2idx[s] for s in start_string]
    # convert to tensor and add the batch dim
    input_eval = torch.tensor(input_eval).unsqueeze(0).to(dev)


    key = [key2idx[key]]
    key = torch.tensor(key, dtype=torch.long).unsqueeze(0).to(dev)

    # Empty string to store our results
    text_generated = []

    # Here batch size == 1
    model.reset_state()

    for i in tqdm(range(generation_length)):
        
      predictions = model(input_eval, key)[:, -1:, :]
      # predictions shape : 1 x 1 x vocab_size
      
      # Get the predicted id
      predicted_id = predictions[0].max(1)[1]

      # Prepare the next input and add the batch dimension
      input_eval = predicted_id.unsqueeze(dim = 0)
      print(" ")
      print(predicted_id)
      print(idx2char)

      text_generated.append(idx2char[int(predicted_id)])

  return (start_string + ''.join(text_generated))



# {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'Gb': 6}
start_string = "FG FE|"
key="Gb"
header = "X: 1\nT: KD-Course-2021\nM: 4/4\nL: 1/8\nQ:1/4=117\nK:C % 0 sharps\n\n"

out = generate_text(model, start_string, key, generation_length=1000)
gen_song = header + out

print(gen_song)