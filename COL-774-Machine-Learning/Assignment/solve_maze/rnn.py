# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # **Part 1: Setup and Data Loading**

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-11-25T08:51:18.731892Z","iopub.execute_input":"2025-11-25T08:51:18.732429Z","iopub.status.idle":"2025-11-25T08:51:22.330903Z","shell.execute_reply.started":"2025-11-25T08:51:18.732408Z","shell.execute_reply":"2025-11-25T08:51:22.330134Z"}}
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import re
import time
import math
import random
import sys

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-11-25T08:51:22.332698Z","iopub.execute_input":"2025-11-25T08:51:22.333002Z","iopub.status.idle":"2025-11-25T08:51:22.33634Z","shell.execute_reply.started":"2025-11-25T08:51:22.332985Z","shell.execute_reply":"2025-11-25T08:51:22.335349Z"}}
# !pip install --upgrade pip
# !pip uninstall -y torch torchvision torchaudio
# !pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-11-25T08:51:22.337189Z","iopub.execute_input":"2025-11-25T08:51:22.337434Z","iopub.status.idle":"2025-11-25T08:51:22.408504Z","shell.execute_reply.started":"2025-11-25T08:51:22.337413Z","shell.execute_reply":"2025-11-25T08:51:22.407772Z"}}
# Set a random seed for reproducibility
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# Check for GPU availability and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-11-25T08:51:22.409294Z","iopub.execute_input":"2025-11-25T08:51:22.409518Z","iopub.status.idle":"2025-11-25T08:51:24.348074Z","shell.execute_reply.started":"2025-11-25T08:51:22.4095Z","shell.execute_reply":"2025-11-25T08:51:24.347435Z"}}
# --- Load Data ---
train_path = '/kaggle/input/maze-dataset-ass4/COL774-A4-Maze-Dataset/train_6x6_mazes.csv'
test_path = '/kaggle/input/maze-dataset-ass4/COL774-A4-Maze-Dataset/test_6x6_mazes.csv'

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
print("Datasets loaded successfully.")
print(f"Training samples: {len(train_df)}")
print(f"Testing samples: {len(test_df)}")

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-11-25T08:51:24.348776Z","iopub.execute_input":"2025-11-25T08:51:24.348966Z","iopub.status.idle":"2025-11-25T08:51:39.811785Z","shell.execute_reply.started":"2025-11-25T08:51:24.34895Z","shell.execute_reply":"2025-11-25T08:51:39.811162Z"}}
# The data is stored as strings; we need to evaluate them into Python lists
train_df['input_sequence'] = train_df['input_sequence'].apply(eval)
train_df['output_path'] = train_df['output_path'].apply(eval)
test_df['input_sequence'] = test_df['input_sequence'].apply(eval)
test_df['output_path'] = test_df['output_path'].apply(eval)

train_df.head()

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Part 2: Data Exploration & Visualization

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-11-25T08:51:39.812468Z","iopub.execute_input":"2025-11-25T08:51:39.812659Z","iopub.status.idle":"2025-11-25T08:51:39.816772Z","shell.execute_reply.started":"2025-11-25T08:51:39.812643Z","shell.execute_reply":"2025-11-25T08:51:39.816068Z"}}
def parse_coords(s):
    """Extracts a tuple of integer coordinates from a string like '(x, y)'."""
    nums = re.findall(r"-?\d+", s)
    return tuple(map(int, nums)) if len(nums) == 2 else None

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-11-25T08:51:39.819192Z","iopub.execute_input":"2025-11-25T08:51:39.819368Z","iopub.status.idle":"2025-11-25T08:51:39.836724Z","shell.execute_reply.started":"2025-11-25T08:51:39.819355Z","shell.execute_reply":"2025-11-25T08:51:39.83607Z"}}
def extract_between(tag, text):
    """Accepts many tag styles: <TAG_START>, <TAG START>, <TAG-START>, <TAGSTART>, etc."""
    patterns = [
        rf"<\s*{tag}\s*[_\-\s]?\s*START\s*>(.*?)<\s*{tag}\s*[_\-\s]?\s*END\s*>",
        rf"<\s*{tag}START\s*>(.*?)<\s*{tag}END\s*>",
        rf"<\s*{tag}\s*START\s*>(.*?)<\s*{tag}\s*END\s*>",
        rf"<\s*{tag.replace(' ', '_')}\s*START\s*>(.*?)<\s*{tag.replace(' ', '_')}\s*END\s*>",
    ]
    for p in patterns:
        m = re.search(p, text, re.S | re.I)
        if m:
            return m.group(1).strip()
    raise ValueError(f"Could not find section for tag '{tag}'. Tried multiple patterns.")

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-11-25T08:51:39.837366Z","iopub.execute_input":"2025-11-25T08:51:39.837548Z","iopub.status.idle":"2025-11-25T08:51:39.85649Z","shell.execute_reply.started":"2025-11-25T08:51:39.837533Z","shell.execute_reply":"2025-11-25T08:51:39.85579Z"}}

def plot_maze(tokens):
    text = " ".join(map(str,tokens))
    
    adj_section = extract_between("ADJLIST", text)
    origin_section = extract_between("ORIGIN", text)
    target_section = extract_between("TARGET", text)
    path_section = extract_between("PATH", text)

    origin = parse_coords(origin_section)
    target = parse_coords(target_section)

    # parse edges like "(r,c) <--> (r2,c2)"
    edge_matches = re.findall(r"\(\s*-?\d+\s*,\s*-?\d+\s*\)\s*<-->\s*\(\s*-?\d+\s*,\s*-?\d+\s*\)", adj_section)
    edges = []
    for em in edge_matches:
        coords = re.findall(r"\(\s*-?\d+\s*,\s*-?\d+\s*\)", em)
        a = parse_coords(coords[0])
        b = parse_coords(coords[1])
        edges.append((a, b))

    # parse path coordinates (supports parenthesized coords)
    path = [parse_coords(p) for p in re.findall(r"\(\s*-?\d+\s*,\s*-?\d+\s*\)", path_section)]
    if not path:
        # fallback: "r,c" tokens without parentheses
        nums = re.findall(r"-?\d+\s*,\s*-?\d+", path_section)
        path = [tuple(map(int, re.findall(r"-?\d+", s))) for s in nums]

    if not edges:
        raise ValueError("No edges found in adjacency list. Ensure format '(r,c) <--> (r2,c2)'.")

    # Grid size (cells indexed with (0,0) = top-left)
    all_nodes = {n for e in edges for n in e if n is not None}
    all_nodes.update([origin, target])
    all_nodes.update([p for p in path if p is not None])
    rows = 6
    cols = 6


    vertical_walls = np.ones((rows, cols + 1), dtype=bool)
    horizontal_walls = np.ones((rows + 1, cols), dtype=bool)

    for (r1, c1), (r2, c2) in edges:
        if r1 == r2:
            # same row, adjacent columns -> remove vertical wall between them
            c_between = min(c1, c2) + 1  # column index of the vertical segment between c and c+1
            vertical_walls[r1, c_between] = False
        elif c1 == c2:
            # same column, adjacent rows -> remove horizontal wall between them
            r_between = min(r1, r2) + 1  # row index of horizontal segment between r and r+1
            horizontal_walls[r_between, c1] = False
        else:
            # diagonal or invalid — ignore, but warn
            print(f"Warning: non-grid edge {(r1,c1)} <--> {(r2,c2)} ignored")

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_aspect('equal')

    # Draw a full light-gray grid (every cell border)
    for r in range(rows):
        for c in range(cols):
            x0, x1 = c, c + 1
            y_top = rows - r
            y_bot = rows - r - 1
            ax.plot([x0, x1], [y_top, y_top], color='lightgray', lw=2)   # top
            ax.plot([x0, x1], [y_bot, y_bot], color='lightgray', lw=2)   # bottom
            ax.plot([x0, x0], [y_bot, y_top], color='lightgray', lw=2)   # left
            ax.plot([x1, x1], [y_bot, y_top], color='lightgray', lw=2)   # right

    # Draw vertical walls (black) using vertical_walls[r,c]
    for r in range(rows):
        for c in range(cols + 1):
            if vertical_walls[r, c]:
                x = c
                y_top = rows - r
                y_bot = rows - r - 1
                ax.plot([x, x], [y_bot, y_top], color='black', lw=5, solid_capstyle='butt')

    # Draw horizontal walls (black)
    for r in range(rows + 1):
        for c in range(cols):
            if horizontal_walls[r, c]:
                y = rows - r
                ax.plot([c, c + 1], [y, y], color='black', lw=5, solid_capstyle='butt')

    shade_path_cells = True
    if shade_path_cells and path:
        for (r, c) in path:
            # rectangle corners in plot coords
            x0, x1 = c, c + 1
            y_top = rows - r
            y_bot = rows - r - 1
            rect = plt.Rectangle((x0, y_bot), 1, 1, facecolor=(1, 0.9, 0.9), edgecolor=None, zorder=0)
            ax.add_patch(rect)

    # Plot path line and markers (convert (r,c) top-left -> matplotlib coords)
    if path:
        path_x = [c + 0.5 for (r, c) in path]
        path_y = [rows - r - 0.5 for (r, c) in path]
        ax.plot(path_x, path_y, linestyle='--', linewidth=2, color='red', zorder=4)
        ax.scatter(path_x[0], path_y[0], c='red', s=80, marker='o', zorder=5)  # start
        ax.scatter(path_x[-1], path_y[-1], c='red', s=80, marker='x', zorder=5)  # goal
    else:
        # if no path, still mark origin/target
        ox, oy = origin[1] + 0.5, rows - origin[0] - 0.5
        tx, ty = target[1] + 0.5, rows - target[0] - 0.5
        ax.scatter(ox, oy, c='red', s=80, marker='o', zorder=5)
        ax.scatter(tx, ty, c='red', s=80, marker='x', zorder=5)

    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_xticks(np.arange(cols))
    ax.set_yticks(np.arange(rows))
    plt.yticks([]) 
    ax.set_xlabel("col")
    ax.set_ylabel("row")
    plt.tight_layout()
    plt.show()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-11-25T08:51:39.857142Z","iopub.execute_input":"2025-11-25T08:51:39.857346Z","iopub.status.idle":"2025-11-25T08:51:40.495774Z","shell.execute_reply.started":"2025-11-25T08:51:39.857331Z","shell.execute_reply":"2025-11-25T08:51:40.495017Z"}}
# --- Visualize a Sample ---
sample_idx = 0
inp_tokens = train_df['input_sequence'].iloc[sample_idx]
out_tokens = train_df['output_path'].iloc[sample_idx]

# To plot, we need a complete token sequence including the <PATH> tags
plot_tokens = inp_tokens + ['<PATH_START>'] + out_tokens + ['<PATH_END>']
plot_maze(plot_tokens)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Part 3: Vocabulary and Preprocessing

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-11-25T08:51:40.496598Z","iopub.execute_input":"2025-11-25T08:51:40.496835Z","iopub.status.idle":"2025-11-25T08:51:40.501923Z","shell.execute_reply.started":"2025-11-25T08:51:40.49682Z","shell.execute_reply":"2025-11-25T08:51:40.501319Z"}}
class Vocabulary:
    """Handles mapping between tokens and numerical indices."""
    def __init__(self):
        # Initialize with special tokens required for sequence modeling
        self.stoi = {"<pad>": 0, "<sos>": 1, "<eos>": 2}
        self.itos = {0: "<pad>", 1: "<sos>", 2: "<eos>"}

    def build_vocabulary(self, all_sequences):
        """Builds vocabulary from a list of tokenized sequences."""
        # Find all unique tokens across all sequences
        all_tokens = {token for seq in all_sequences for token in seq}
        
        # Add each unique token to the mapping
        for token in sorted(list(all_tokens)):
            if token not in self.stoi:
                idx = len(self.stoi)
                self.stoi[token] = idx
                self.itos[idx] = token

    def __len__(self):
        return len(self.stoi)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-11-25T08:51:40.502658Z","iopub.execute_input":"2025-11-25T08:51:40.502861Z","iopub.status.idle":"2025-11-25T08:51:40.807049Z","shell.execute_reply.started":"2025-11-25T08:51:40.502839Z","shell.execute_reply":"2025-11-25T08:51:40.806427Z"}}
# Combine all sequences from training data to build a comprehensive vocabulary
all_sequences = train_df['input_sequence'].tolist() + train_df['output_path'].tolist()

vocab = Vocabulary()
vocab.build_vocabulary(all_sequences)

print(f"Vocabulary size: {len(vocab)}")
print(f"Sample mapping: '<ADJLIST_START>' -> {vocab.stoi.get('<ADJLIST_START>')}")
print(f"Sample mapping: '(3,5)' -> {vocab.stoi.get('(3,5)')}")

# Define global constants for special token indices for easy access
PAD_IDX = vocab.stoi["<pad>"]
SOS_IDX = vocab.stoi["<sos>"]
EOS_IDX = vocab.stoi["<eos>"]

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Part 4: PyTorch Dataset and DataLoader

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-11-25T08:51:40.807787Z","iopub.execute_input":"2025-11-25T08:51:40.808039Z","iopub.status.idle":"2025-11-25T08:51:40.813613Z","shell.execute_reply.started":"2025-11-25T08:51:40.808015Z","shell.execute_reply":"2025-11-25T08:51:40.812962Z"}}
class MazeDataset(Dataset):
    """Custom PyTorch Dataset for maze data."""
    def __init__(self, dataframe, vocab):
        self.df = dataframe
        self.vocab = vocab

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        input_seq = self.df.iloc[idx]['input_sequence']
        output_seq = self.df.iloc[idx]['output_path']

        # Convert input tokens to their corresponding indices
        input_tensor = torch.tensor([self.vocab.stoi[token] for token in input_seq], dtype=torch.long)

        # Prepend <sos> and append <eos> to the output path, then convert to indices
        output_tokens = ['<sos>'] + output_seq + ['<eos>']
        output_tensor = torch.tensor([self.vocab.stoi[token] for token in output_tokens], dtype=torch.long)

        return input_tensor, output_tensor

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-11-25T08:51:40.814297Z","iopub.execute_input":"2025-11-25T08:51:40.814521Z","iopub.status.idle":"2025-11-25T08:51:40.831756Z","shell.execute_reply.started":"2025-11-25T08:51:40.814503Z","shell.execute_reply":"2025-11-25T08:51:40.831227Z"}}
class PadCollate:
    """A collate function to pad sequences in a batch to the same length."""
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        inputs = [item[0] for item in batch]
        outputs = [item[1] for item in batch]
        
        # pad_sequence is a PyTorch utility that pads to the length of the longest sequence
        padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=self.pad_idx)
        padded_outputs = pad_sequence(outputs, batch_first=True, padding_value=self.pad_idx)

        return padded_inputs, padded_outputs

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-11-25T08:51:40.83249Z","iopub.execute_input":"2025-11-25T08:51:40.832669Z","iopub.status.idle":"2025-11-25T08:51:40.866441Z","shell.execute_reply.started":"2025-11-25T08:51:40.832655Z","shell.execute_reply":"2025-11-25T08:51:40.865718Z"}}
# --- Creating Datasets and DataLoaders ---
full_train_dataset = MazeDataset(train_df, vocab)
test_dataset = MazeDataset(test_df, vocab)

# Split training data into training and validation sets (90% train, 10% val).
# Note: torch.utils.data.random_split is an allowed PyTorch utility.
train_size = int(0.9 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-11-25T08:51:40.867316Z","iopub.execute_input":"2025-11-25T08:51:40.867514Z","iopub.status.idle":"2025-11-25T08:51:40.871539Z","shell.execute_reply.started":"2025-11-25T08:51:40.867496Z","shell.execute_reply":"2025-11-25T08:51:40.871008Z"}}
# # --- Creating Datasets and DataLoaders ---
# full_train_dataset = MazeDataset(train_df, vocab)
# test_dataset = MazeDataset(test_df, vocab)

# # --- FOR DEBUGGING: Use a smaller subset of the data ---
# # We will use about 1% of the data to speed up the test run.
# debug_train_size = int(0.01 * len(full_train_dataset))
# debug_val_size = int(0.1 * debug_train_size) # 10% of the small training set for validation
# debug_train_size = debug_train_size - debug_val_size

# # Create a small subset for quick testing
# full_debug_dataset, _ = random_split(full_train_dataset, [debug_train_size + debug_val_size, len(full_train_dataset) - (debug_train_size + debug_val_size)])
# train_dataset, val_dataset = random_split(full_debug_dataset, [debug_train_size, debug_val_size])
# # --- END DEBUGGING SUBSET ---


# print(f"DEBUG MODE: Using smaller datasets.")
# print(f"Train dataset size: {len(train_dataset)}")
# print(f"Validation dataset size: {len(val_dataset)}")
# print(f"Test dataset size: {len(test_dataset)}") # Test set remains the same

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-11-25T08:51:40.87222Z","iopub.execute_input":"2025-11-25T08:51:40.872404Z","iopub.status.idle":"2025-11-25T08:51:40.928617Z","shell.execute_reply.started":"2025-11-25T08:51:40.872389Z","shell.execute_reply":"2025-11-25T08:51:40.927983Z"}}
BATCH_SIZE = 64
pad_collate_fn = PadCollate(pad_idx=PAD_IDX)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_collate_fn)
val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=pad_collate_fn)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=pad_collate_fn)

print("\nDataLoaders created.")
# Inspect a sample batch to verify shapes
inputs, outputs = next(iter(train_loader))
print(f"Shape of a batch of inputs: {inputs.shape}")
print(f"Shape of a batch of outputs: {outputs.shape}")

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Part 5: Model 1 - RNN with Attention

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-11-25T08:51:40.929207Z","iopub.execute_input":"2025-11-25T08:51:40.929423Z","iopub.status.idle":"2025-11-25T08:51:40.934033Z","shell.execute_reply.started":"2025-11-25T08:51:40.929398Z","shell.execute_reply":"2025-11-25T08:51:40.933442Z"}}
class Encoder(nn.Module):
    """A simple RNN encoder."""
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.RNN(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src: [batch_size, src_len]
        embedded = self.dropout(self.embedding(src))
        # embedded: [batch_size, src_len, emb_dim]
        outputs, hidden = self.rnn(embedded)
        # outputs: [batch_size, src_len, hid_dim]
        # hidden: [n_layers, batch_size, hid_dim]
        return outputs, hidden

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-11-25T08:51:40.934913Z","iopub.execute_input":"2025-11-25T08:51:40.935215Z","iopub.status.idle":"2025-11-25T08:51:40.949189Z","shell.execute_reply.started":"2025-11-25T08:51:40.935194Z","shell.execute_reply":"2025-11-25T08:51:40.948454Z"}}
class Attention(nn.Module):
    """A Bahdanau-style (additive) attention mechanism."""
    def __init__(self, hid_dim):
        super().__init__()
        self.attn = nn.Linear((hid_dim) * 2, hid_dim)
        self.v = nn.Linear(hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden: [batch_size, hid_dim] (from the last RNN layer)
        # encoder_outputs: [batch_size, src_len, hid_dim]
        src_len = encoder_outputs.shape[1]
        
        # Repeat decoder hidden state src_len times to align with encoder outputs
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        
        # Calculate energy score
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        # energy: [batch_size, src_len, hid_dim]
        
        # Get attention weights
        attention = self.v(energy).squeeze(2)
        # attention: [batch_size, src_len]
        
        return torch.softmax(attention, dim=1)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-11-25T08:51:40.949932Z","iopub.execute_input":"2025-11-25T08:51:40.950185Z","iopub.status.idle":"2025-11-25T08:51:40.966411Z","shell.execute_reply.started":"2025-11-25T08:51:40.95017Z","shell.execute_reply":"2025-11-25T08:51:40.965745Z"}}
class Decoder(nn.Module):
    """An RNN decoder that uses attention at each step."""
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.RNN(hid_dim + emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hid_dim * 2 + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs):
        # input: [batch_size]
        # hidden: [n_layers, batch_size, hid_dim]
        # encoder_outputs: [batch_size, src_len, hid_dim]
        input = input.unsqueeze(1)
        embedded = self.dropout(self.embedding(input))

        # Use the last layer's hidden state for attention calculation
        a = self.attention(hidden[-1], encoder_outputs).unsqueeze(1)
        # a: [batch_size, 1, src_len]
        
        # Compute the context vector
        context = torch.bmm(a, encoder_outputs)
        # context: [batch_size, 1, hid_dim]
        
        # Concatenate embedding and context vector for RNN input
        rnn_input = torch.cat((embedded, context), dim=2)
        
        output, hidden = self.rnn(rnn_input, hidden)
        
        # Final prediction concatenates RNN output, context, and embedding
        prediction = self.fc_out(torch.cat((output.squeeze(1), context.squeeze(1), embedded.squeeze(1)), dim=1))
        # prediction: [batch_size, output_dim]
        return prediction, hidden

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-11-25T08:51:40.96719Z","iopub.execute_input":"2025-11-25T08:51:40.967416Z","iopub.status.idle":"2025-11-25T08:51:40.984562Z","shell.execute_reply.started":"2025-11-25T08:51:40.967397Z","shell.execute_reply":"2025-11-25T08:51:40.983937Z"}}
class Seq2Seq(nn.Module):
    """The main Seq2Seq model that wraps the Encoder, Attention, and Decoder."""
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size, trg_len = trg.shape
        trg_vocab_size = self.decoder.output_dim
        
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        encoder_outputs, hidden = self.encoder(src)
        
        # First input to decoder is the <sos> token
        input = trg[:, 0]
        
        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            outputs[:,t] = output
            
            # Decide whether to use teacher forcing or the model's own prediction
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[:, t] if teacher_force else top1
            
        return outputs

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Part 6: Training the RNN Model

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-11-25T08:51:40.9852Z","iopub.execute_input":"2025-11-25T08:51:40.985433Z","iopub.status.idle":"2025-11-25T08:51:41.000382Z","shell.execute_reply.started":"2025-11-25T08:51:40.985418Z","shell.execute_reply":"2025-11-25T08:51:40.999722Z"}}
# --- Assignment Hyperparameters ---
BATCH_SIZE = 32
N_EPOCHS = 30
LEARNING_RATE = 1e-4
ENC_EMB_DIM = 128
DEC_EMB_DIM = 128
HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0.3
DEC_DROPOUT = 0.3
TEACHER_FORCING_RATIO = 0.5
CLIP = 1.0
MAX_DECODING_LEN = 200

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-11-25T08:51:41.004076Z","iopub.execute_input":"2025-11-25T08:51:41.004647Z","iopub.status.idle":"2025-11-25T08:51:41.014218Z","shell.execute_reply.started":"2025-11-25T08:51:41.004623Z","shell.execute_reply":"2025-11-25T08:51:41.013634Z"}}
# # --- DEBUG CONFIG (very small model for quick testing) ---
# BATCH_SIZE = 16
# N_EPOCHS = 2
# LEARNING_RATE = 5e-4

# ENC_EMB_DIM = 16     # small embedding dimension
# DEC_EMB_DIM = 16
# HID_DIM = 32         # small hidden size
# N_LAYERS = 1
# ENC_DROPOUT = 0.1
# DEC_DROPOUT = 0.1

# TEACHER_FORCING_RATIO = 0.5
# CLIP = 1.0
# MAX_DECODING_LEN = 40   # shorter decoding for speed

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-11-25T08:51:41.014852Z","iopub.execute_input":"2025-11-25T08:51:41.015105Z","iopub.status.idle":"2025-11-25T08:51:41.027931Z","shell.execute_reply.started":"2025-11-25T08:51:41.015069Z","shell.execute_reply":"2025-11-25T08:51:41.027131Z"}}
# Dimensions of vocabulary
INPUT_DIM = len(vocab)   # Same vocab for encoder
OUTPUT_DIM = len(vocab)  # Same vocab for decoder

print("INPUT_DIM =", INPUT_DIM)
print("OUTPUT_DIM =", OUTPUT_DIM)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-11-25T08:51:41.028674Z","iopub.execute_input":"2025-11-25T08:51:41.028963Z","iopub.status.idle":"2025-11-25T08:51:43.572827Z","shell.execute_reply.started":"2025-11-25T08:51:41.028942Z","shell.execute_reply":"2025-11-25T08:51:43.572271Z"}}
# --- Model Initialization ---
attn = Attention(HID_DIM)
enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT, attn)
model_rnn = Seq2Seq(enc, dec, device).to(device)

def init_weights(m):
    for name, param in m.named_parameters():
        if "weight" in name:
            nn.init.uniform_(param.data, -0.08, 0.08)
        else:
            nn.init.constant_(param.data, 0)

model_rnn.apply(init_weights)

optimizer = optim.Adam(model_rnn.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-11-25T08:51:43.573641Z","iopub.execute_input":"2025-11-25T08:51:43.573941Z","iopub.status.idle":"2025-11-25T08:51:43.579047Z","shell.execute_reply.started":"2025-11-25T08:51:43.573926Z","shell.execute_reply":"2025-11-25T08:51:43.578314Z"}}
def generate(model, src, max_len=MAX_DECODING_LEN):
    model.eval()
    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src)
        batch_size = src.size(0)

        input_tok = torch.full((batch_size,), SOS_IDX, dtype=torch.long, device=device)
        preds = [input_tok.unsqueeze(1)]

        for t in range(1, max_len):
            output, hidden = model.decoder(input_tok, hidden, encoder_outputs)
            top1 = output.argmax(1)
            preds.append(top1.unsqueeze(1))
            input_tok = top1

            if (top1 == EOS_IDX).all():
                break

        return torch.cat(preds, dim=1)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-11-25T08:51:43.579625Z","iopub.execute_input":"2025-11-25T08:51:43.579852Z","iopub.status.idle":"2025-11-25T08:51:43.595639Z","shell.execute_reply.started":"2025-11-25T08:51:43.57983Z","shell.execute_reply":"2025-11-25T08:51:43.595138Z"}}
def truncate_at_eos(seq):
    if EOS_IDX in seq:
        return seq[: seq.index(EOS_IDX)+1]
    return seq

def sequences_equal(pred_seq, true_seq):
    return truncate_at_eos(pred_seq) == truncate_at_eos(true_seq)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-11-25T08:51:43.596348Z","iopub.execute_input":"2025-11-25T08:51:43.596584Z","iopub.status.idle":"2025-11-25T08:51:43.610384Z","shell.execute_reply.started":"2025-11-25T08:51:43.596565Z","shell.execute_reply":"2025-11-25T08:51:43.609648Z"}}
def batch_metrics(output_logits, trg):
    preds = output_logits.argmax(dim=2)  # [B, L]
    preds = preds[:, 1:]
    trg = trg[:, 1:]

    batch_size, seq_len = trg.shape

    token_tp = 0
    token_pred_pos = 0
    token_true_pos = 0

    seq_match = 0

    for i in range(batch_size):
        pred_i = preds[i].tolist()
        true_i = trg[i].tolist()

        # token-level
        for p, t in zip(pred_i, true_i):
            if t != PAD_IDX:
                token_true_pos += 1
                if p == t:
                    token_tp += 1
            if p != PAD_IDX:
                token_pred_pos += 1

        # sequence-level
        if sequences_equal(pred_i, true_i):
            seq_match += 1

    return token_tp, token_pred_pos, token_true_pos, seq_match, batch_size

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-11-25T08:51:43.611128Z","iopub.execute_input":"2025-11-25T08:51:43.611353Z","iopub.status.idle":"2025-11-25T08:51:43.626128Z","shell.execute_reply.started":"2025-11-25T08:51:43.611338Z","shell.execute_reply":"2025-11-25T08:51:43.625594Z"}}
def compute_prf(token_tp, token_pred_pos, token_true_pos, eps=1e-12):
    precision = token_tp / (token_pred_pos + eps)
    recall = token_tp / (token_true_pos + eps)
    f1 = 2*precision*recall / (precision + recall + eps)
    return precision, recall, f1

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-11-25T08:51:43.626835Z","iopub.execute_input":"2025-11-25T08:51:43.627043Z","iopub.status.idle":"2025-11-25T08:51:43.641642Z","shell.execute_reply.started":"2025-11-25T08:51:43.627023Z","shell.execute_reply":"2025-11-25T08:51:43.641111Z"}}
def train_epoch(model, loader, optimizer, criterion, tf_ratio, clip):
    model.train()

    epoch_loss = 0
    tp = pp = tp_true = 0
    seq_match = 0
    total = 0

    for src, trg in loader:
        src, trg = src.to(device), trg.to(device)

        optimizer.zero_grad()
        output = model(src, trg, teacher_forcing_ratio=tf_ratio)

        output_dim = output.shape[-1]
        loss = criterion(output[:,1:].reshape(-1, output_dim),
                         trg[:,1:].reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

        btp, bpp, btp_true, bmatch, bsz = batch_metrics(output, trg)
        tp += btp; pp += bpp; tp_true += btp_true
        seq_match += bmatch; total += bsz

    p, r, f1 = compute_prf(tp, pp, tp_true)
    seq_acc = seq_match / total

    return epoch_loss/len(loader), p, r, f1, seq_acc


def eval_epoch(model, loader, criterion):
    model.eval()

    epoch_loss = 0

    # token-level metrics
    tp = pp = tp_true = 0

    # sequence accuracy metrics
    seq_match = 0
    total = 0

    with torch.no_grad():
        for src, trg in loader:
            src, trg = src.to(device), trg.to(device)

            # ---------- LOSS (Teacher Forcing ON) ----------
            output = model(src, trg, teacher_forcing_ratio=1.0)
            output_dim = output.shape[-1]
            loss = criterion(output[:,1:].reshape(-1, output_dim),
                             trg[:,1:].reshape(-1))
            epoch_loss += loss.item()

            # ---------- ACCURACY (Teacher Forcing OFF) ----------
            preds = generate(model, src)   # <-- AUTOREGRESSIVE

            # Convert to python lists
            preds = preds[:, 1:]  
            trg_cut = trg[:, 1:]

            for i in range(src.size(0)):
                pred_seq = preds[i].tolist()
                true_seq = trg_cut[i].tolist()

                # Remove padding
                pred_seq = [x for x in pred_seq if x != PAD_IDX]
                true_seq = [x for x in true_seq if x != PAD_IDX]

                # Trim at EOS/PATH_END
                if EOS_IDX in pred_seq:
                    pred_seq = pred_seq[:pred_seq.index(EOS_IDX)]
                if EOS_IDX in true_seq:
                    true_seq = true_seq[:true_seq.index(EOS_IDX)]

                # token-level F1
                for p, t in zip(pred_seq, true_seq):
                    if p == t:
                        tp += 1
                    pp += 1
                    tp_true += 1

                # sequence exact match
                if pred_seq == true_seq:
                    seq_match += 1
                total += 1

    precision, recall, f1 = compute_prf(tp, pp, tp_true)
    seq_acc = seq_match / total
    return epoch_loss/len(loader), precision, recall, f1, seq_acc

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-11-25T08:51:43.642306Z","iopub.execute_input":"2025-11-25T08:51:43.64255Z","iopub.status.idle":"2025-11-25T08:51:43.65653Z","shell.execute_reply.started":"2025-11-25T08:51:43.642535Z","shell.execute_reply":"2025-11-25T08:51:43.655843Z"}}
def epoch_time(start_time, end_time):
    elapsed = end_time - start_time
    return int(elapsed // 60), int(elapsed % 60)

# %% [code] {"execution":{"iopub.status.busy":"2025-11-25T08:51:43.657373Z","iopub.execute_input":"2025-11-25T08:51:43.65792Z","iopub.status.idle":"2025-11-25T08:51:43.669993Z","shell.execute_reply.started":"2025-11-25T08:51:43.657902Z","shell.execute_reply":"2025-11-25T08:51:43.669505Z"},"jupyter":{"outputs_hidden":false}}
# Metric tracking lists
train_losses = []
valid_losses = []

train_f1 = []
valid_f1 = []

train_seq_acc = []
valid_seq_acc = []

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-11-25T08:51:43.670763Z","iopub.execute_input":"2025-11-25T08:51:43.671055Z","iopub.status.idle":"2025-11-25T11:22:19.839274Z","shell.execute_reply.started":"2025-11-25T08:51:43.671033Z","shell.execute_reply":"2025-11-25T11:22:19.838616Z"}}
import copy

best_val_seq = -1.0
best_state = None

print("Training RNN with Attention...")

for epoch in range(N_EPOCHS):
    start = time.time()

    # --- Training ---
    tr_loss, tr_p, tr_r, tr_f1, tr_seq = train_epoch(
        model_rnn,
        train_loader,
        optimizer,
        criterion,
        TEACHER_FORCING_RATIO,
        CLIP
    )

    # --- Validation ---
    val_loss, val_p, val_r, val_f1, val_seq = eval_epoch(
        model_rnn,
        val_loader,
        criterion
    )

    # STORE METRICS FOR PLOTTING
    train_losses.append(tr_loss)
    valid_losses.append(val_loss)

    train_f1.append(tr_f1)
    valid_f1.append(val_f1)

    train_seq_acc.append(tr_seq)
    valid_seq_acc.append(val_seq)

    # SAVE BEST MODEL (based on sequence accuracy)
    if val_seq > best_val_seq:
        best_val_seq = val_seq
        best_state = copy.deepcopy(model_rnn.state_dict())
        torch.save(best_state, "best_rnn_attn.pt")

    mm, ss = epoch_time(start, time.time())
    print(f"Epoch {epoch+1:02d} | Time: {mm}m {ss}s")
    print(f"  Train Loss: {tr_loss:.4f} | F1: {tr_f1:.4f} | SeqAcc: {tr_seq:.4f}")
    print(f"  Val   Loss: {val_loss:.4f} | F1: {val_f1:.4f} | SeqAcc: {val_seq:.4f}")
    print("-" * 80)

print("\nTraining Complete.")
print("Best Validation Sequence Accuracy =", best_val_seq)

# Load the best model for plotting & test evaluation
model_rnn.load_state_dict(best_state)

# %% [code] {"execution":{"iopub.status.busy":"2025-11-25T11:22:19.840115Z","iopub.execute_input":"2025-11-25T11:22:19.840388Z","iopub.status.idle":"2025-11-25T11:22:20.356115Z","shell.execute_reply.started":"2025-11-25T11:22:19.840363Z","shell.execute_reply":"2025-11-25T11:22:20.355415Z"},"jupyter":{"outputs_hidden":false}}
# Part 7: TRAINING / VALIDATION PLOTS

epochs = range(1, N_EPOCHS + 1)

plt.figure(figsize=(16,4))

# 1. LOSS CURVE
plt.subplot(1,3,1)
plt.plot(epochs, train_losses, label="Train Loss")
plt.plot(epochs, valid_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend()

# 2. TOKEN F1 CURVE
plt.subplot(1,3,2)
plt.plot(epochs, train_f1, label="Train F1")
plt.plot(epochs, valid_f1, label="Validation F1")
plt.xlabel("Epoch")
plt.ylabel("F1 Score")
plt.title("Token-level F1 Curve")
plt.legend()

# 3. SEQUENCE ACCURACY CURVE
plt.subplot(1,3,3)
plt.plot(epochs, train_seq_acc, label="Train Seq Accuracy")
plt.plot(epochs, valid_seq_acc, label="Validation Seq Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Exact Match Accuracy")
plt.title("Sequence Accuracy Curve")
plt.legend()

plt.tight_layout()
plt.show()

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Part 8: TEST SET EVALUATION

# %% [code] {"execution":{"iopub.status.busy":"2025-11-25T11:22:20.356977Z","iopub.execute_input":"2025-11-25T11:22:20.357262Z","iopub.status.idle":"2025-11-25T11:25:08.152045Z","shell.execute_reply.started":"2025-11-25T11:22:20.357239Z","shell.execute_reply":"2025-11-25T11:25:08.151162Z"},"jupyter":{"outputs_hidden":false}}
print("\nEvaluating on Test Set...")

test_loss, test_p, test_r, test_f1, test_seq = eval_epoch(
    model_rnn, test_loader, criterion
)

print("\n===== TEST METRICS =====")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Precision: {test_p:.4f}")
print(f"Test Recall: {test_r:.4f}")
print(f"Test F1 Score: {test_f1:.4f}")
print(f"Test Sequence Accuracy (Exact Match): {test_seq:.4f}")

# %% [code] {"execution":{"iopub.status.busy":"2025-11-25T11:25:08.152973Z","iopub.execute_input":"2025-11-25T11:25:08.153576Z","iopub.status.idle":"2025-11-25T11:25:09.6358Z","shell.execute_reply.started":"2025-11-25T11:25:08.153541Z","shell.execute_reply":"2025-11-25T11:25:09.635144Z"},"jupyter":{"outputs_hidden":false}}
# Part 9: VISUALIZE 5 RANDOM MAZES WITH PREDICTED PATHS

model_rnn.eval()

print("\nVisualizing 5 Random Validation Mazes...\n")

# pick 5 random indices from validation data
indices = np.random.choice(len(val_dataset), 5, replace=False)

for idx in indices:
    print(f"--- Visualization for Validation Sample {idx} ---")

    # --- Get input and target ---
    inp, trg = val_dataset[idx]
    inp = inp.unsqueeze(0).to(device)   # add batch dimension

    # --- Generate predicted output ---
    pred = generate(model_rnn, inp, max_len=MAX_DECODING_LEN)
    pred = pred.squeeze(0).tolist()

    # Convert indices to tokens
    pred_tokens = [vocab.itos[t] for t in pred]
    if "<eos>" in pred_tokens:
        pred_tokens = pred_tokens[:pred_tokens.index("<eos>")]

    true_tokens = [vocab.itos[t] for t in trg.tolist()]
    if "<eos>" in true_tokens:
        true_tokens = true_tokens[:true_tokens.index("<eos>")]

    print("Predicted Path:", pred_tokens)
    print("Ground Truth Path:", true_tokens)

    # --- Rebuild the full input sequence for plotting ---
    inp_tokens = [vocab.itos[t] for t in val_dataset[idx][0].tolist()]

    plot_tokens = (
        inp_tokens +
        ["<PATH_START>"] +
        pred_tokens +
        ["<PATH_END>"]
    )

    plot_maze(plot_tokens)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2025-11-25T11:25:09.636483Z","iopub.execute_input":"2025-11-25T11:25:09.636706Z","iopub.status.idle":"2025-11-25T11:25:09.66227Z","shell.execute_reply.started":"2025-11-25T11:25:09.63669Z","shell.execute_reply":"2025-11-25T11:25:09.661395Z"}}
torch.save(best_state, "/kaggle/working/best_rnn_attn.pt")