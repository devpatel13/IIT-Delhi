import pandas as pd
from torch.utils.data import Dataset, DataLoader
import ast
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support

import os, time, json, random
from torch.utils.data import Subset
from torch.cuda.amp import autocast, GradScaler

#  DATA LOADING 

# Vocabulary Class

class Vocabulary:
    def __init__(self):
        self.token2idx = {}
        self.idx2token = []

        # Add special tokens
        specials = ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<PATH_START>", "<PATH_END>"]
        for tok in specials:
            self.add_token(tok)

        self.pad_index = self.token2idx["<PAD>"]
        self.unk_index = self.token2idx["<UNK>"]
        self.bos_index = self.token2idx["<BOS>"]
        self.eos_index = self.token2idx["<EOS>"]
        self.path_start_index = self.token2idx["<PATH_START>"]
        self.path_end_index = self.token2idx["<PATH_END>"]

    def add_token(self, tok):
        if tok not in self.token2idx:
            self.token2idx[tok] = len(self.idx2token)
            self.idx2token.append(tok)

    def encode(self, seq):
        return [self.token2idx.get(tok, self.unk_index) for tok in seq]

    def decode(self, idxs):
        return [self.idx2token[i] for i in idxs]


# Maze Dataset Class

class MazeDataset(Dataset):
    def __init__(self, dataframe, vocab):
        self.vocab = vocab
        self.inputs = [ast.literal_eval(x) for x in dataframe["input_sequence"]]
        self.outputs = [ast.literal_eval(x) for x in dataframe["output_path"]]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        src_tokens = self.inputs[idx]
        tgt_tokens = self.outputs[idx]

        src_ids = self.vocab.encode(src_tokens)

        tgt_in = [self.vocab.path_start_index] + self.vocab.encode(tgt_tokens[:-1])
        tgt_out = self.vocab.encode(tgt_tokens)

        return (
            torch.tensor(src_ids),
            torch.tensor(tgt_in),
            torch.tensor(tgt_out)
        )


# Collate Function (padding)

def collate_fn(batch):
    src, tgt_in, tgt_out = zip(*batch)

    pad = vocab.pad_index

    src = torch.nn.utils.rnn.pad_sequence(src, batch_first=True, padding_value=pad)
    tgt_in = torch.nn.utils.rnn.pad_sequence(tgt_in, batch_first=True, padding_value=pad)
    tgt_out = torch.nn.utils.rnn.pad_sequence(tgt_out, batch_first=True, padding_value=pad)

    src_mask = (src == pad)
    tgt_mask = (tgt_in == pad)

    return src, tgt_in, tgt_out, src_mask, tgt_mask


# LOAD TRAIN 

train_path = "/kaggle/input/mazerunner1/train_6x6_mazes.csv"
test_path  = "/kaggle/input/mazerunner1/test_6x6_mazes.csv"

df_train = pd.read_csv(train_path)
df_test  = pd.read_csv(test_path)

# Build vocabulary from FULL TRAIN only

vocab = Vocabulary()

for seq in df_train["input_sequence"]:
    for tok in ast.literal_eval(seq):
        vocab.add_token(tok)

for seq in df_train["output_path"]:
    for tok in ast.literal_eval(seq):
        vocab.add_token(tok)


# TRAIN / VALIDATION SPLIT

train_df, val_df = train_test_split(df_train, test_size=0.1, random_state=42)

train_dataset = MazeDataset(train_df, vocab)
val_dataset   = MazeDataset(val_df, vocab)

# Test dataset (no split)
test_dataset  = MazeDataset(df_test, vocab)

# DATALOADERS

train_loader = DataLoader(
    train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn
)

val_loader = DataLoader(
    val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn
)

test_loader = DataLoader(
    test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn
)

print("Train samples :", len(train_dataset))
print("Val samples   :", len(val_dataset))
print("Test samples  :", len(test_dataset))
print("Vocab size    :", len(vocab.idx2token))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Sinusoidal Positional Encoding

def positional_encoding(max_len, d_model):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2) *
                    (-np.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div)
    pe[:, 1::2] = torch.cos(position * div)
    return pe.unsqueeze(0)  # [1, max_len, d_model]


# Transformer Model
class MazeTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=8, 
                 num_layers=6, dim_ff=512, dropout=0.1, max_len=300):
        super().__init__()

        self.d_model = d_model

        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = positional_encoding(max_len, d_model).to(device)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_ff, dropout=dropout,
            batch_first=True
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_ff, dropout=dropout,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt, src_mask, tgt_mask, src_pad_mask, tgt_pad_mask):
        src_emb = self.embed(src) * np.sqrt(self.d_model) + self.pos[:, :src.size(1)]
        tgt_emb = self.embed(tgt) * np.sqrt(self.d_model) + self.pos[:, :tgt.size(1)]

        memory = self.encoder(src_emb, mask=src_mask, src_key_padding_mask=src_pad_mask)
        output = self.decoder(
            tgt_emb, memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask
        )

        return self.fc_out(output)


# Greedy Auto-Regressive Decode
def greedy_decode(model, src, src_pad_mask, start_token, end_token, max_len=60):
    model.eval()
    with torch.no_grad():
        src_emb = model.embed(src) * np.sqrt(model.d_model) + model.pos[:, :src.size(1)]
        memory = model.encoder(src_emb, src_key_padding_mask=src_pad_mask)

        ys = torch.tensor([[start_token]], device=device)

        for _ in range(max_len):
            tgt_emb = model.embed(ys) * np.sqrt(model.d_model) + model.pos[:, :ys.size(1)]

            tgt_mask = nn.Transformer.generate_square_subsequent_mask(ys.size(1)).to(device)

            out = model.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
            logits = model.fc_out(out[:, -1])
            next_tok = logits.argmax(dim=-1)

            ys = torch.cat([ys, next_tok.unsqueeze(0)], dim=1)
            if next_tok.item() == end_token:
                break
        return ys.squeeze(0)


# Training Loop
def train_epoch(model, loader, criterion, optimizer, pad_idx, start_token):
    model.train()
    total_loss = 0
    total_token_correct = 0
    total_tokens = 0
    total_seq_correct = 0

    for batch in loader:
        src, tgt_in, tgt_out, src_pad_mask, tgt_pad_mask = batch
        src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
        src_pad_mask, tgt_pad_mask = src_pad_mask.to(device), tgt_pad_mask.to(device)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_in.size(1)).to(device)

        optimizer.zero_grad()
        output = model(src, tgt_in, None, tgt_mask, src_pad_mask, tgt_pad_mask)

        loss = criterion(output.reshape(-1, output.size(-1)), tgt_out.reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        #AUTOREGRESSIVE accuracy (required)
        preds = greedy_decode(model, src[0:1], src_pad_mask[0:1], start_token, pad_idx)
        gold = tgt_out[0]

        # Token accuracy
        L = min(len(preds), len(gold))
        total_token_correct += (preds[:L] == gold[:L]).sum().item()
        total_tokens += L

        # Sequence accuracy
        if torch.equal(preds, gold[:len(preds)]):
            total_seq_correct += 1

    token_acc = total_token_correct / total_tokens
    seq_acc = total_seq_correct / len(loader)
    return total_loss / len(loader), token_acc, seq_acc


# VALIDATION LOOP (same as training but no backward)
def eval_epoch(model, loader, criterion, pad_idx, start_token):
    model.eval()
    total_loss = 0
    total_token_correct = 0
    total_tokens = 0
    total_seq_correct = 0

    with torch.no_grad():
        for batch in loader:
            src, tgt_in, tgt_out, src_pad_mask, tgt_pad_mask = batch
            src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
            src_pad_mask, tgt_pad_mask = src_pad_mask.to(device), tgt_pad_mask.to(device)

            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_in.size(1)).to(device)
            output = model(src, tgt_in, None, tgt_mask, src_pad_mask, tgt_pad_mask)

            loss = criterion(output.reshape(-1, output.size(-1)), tgt_out.reshape(-1))
            total_loss += loss.item()

            # AUTOREGRESSIVE accuracy
            preds = greedy_decode(model, src[0:1], src_pad_mask[0:1], start_token, pad_idx)
            gold = tgt_out[0]

            L = min(len(preds), len(gold))
            total_token_correct += (preds[:L] == gold[:L]).sum().item()
            total_tokens += L

            if torch.equal(preds, gold[:len(preds)]):
                total_seq_correct += 1

    token_acc = total_token_correct / total_tokens
    seq_acc = total_seq_correct / len(loader)
    return total_loss / len(loader), token_acc, seq_acc

def evaluate_test(model, test_loader, pad_idx, start_token):
    all_preds = []
    all_targets = []

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            src, tgt_in, tgt_out, src_pad_mask, tgt_pad_mask = batch
            src = src.to(device)
            src_pad_mask = src_pad_mask.to(device)

            preds = greedy_decode(model, src[0:1], src_pad_mask[0:1], start_token, pad_idx)
            gold = tgt_out[0]

            # Collect for F1
            L = min(len(preds), len(gold))
            all_preds.extend(preds[:L].cpu().numpy())
            all_targets.extend(gold[:L].cpu().numpy())

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average="micro"
    )

    # Sequence accuracy
    seq_correct = 0
    for batch in test_loader:
        src, tgt_in, tgt_out, src_pad_mask, tgt_pad_mask = batch
        src = src.to(device)
        src_pad_mask = src_pad_mask.to(device)

        preds = greedy_decode(model, src[0:1], src_pad_mask[0:1], start_token, pad_idx)
        gold = tgt_out[0].to(device)

        if torch.equal(preds, gold[:len(preds)]):
            seq_correct += 1

    seq_acc = seq_correct / len(test_loader)

    return precision, recall, f1, seq_acc

# Plot Loss & Acc Curves
def plot_curves(train_loss, val_loss, train_acc, val_acc, train_seq, val_seq, test_acc, test_seq):
    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(16,5))
    plt.subplot(1,3,1)
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Val Loss")
    plt.legend()
    plt.title("Loss vs Epochs")

    plt.subplot(1,3,2)
    plt.plot(epochs, train_acc, label="Train Token Acc")
    plt.plot(epochs, val_acc, label="Val Token Acc")
    plt.legend()
    plt.title("Token Accuracy vs Epochs")

    plt.subplot(1,3,3)
    plt.plot(epochs, train_seq, label="Train Seq Acc")
    plt.plot(epochs, val_seq, label="Val Seq Acc")
    plt.axhline(test_seq, color='r', linestyle='--', label="Test Seq Acc")
    plt.legend()
    plt.title("Sequence Accuracy vs Epochs")

    plt.show()


# Visualize 3 Predictions
def visualize_random_predictions(model, dataset, visualize_func, pad_idx, start_token):
    for _ in range(3):
        idx = random.randint(0, len(dataset)-1)
        src, tgt_in, tgt_out, src_pad_mask, tgt_pad_mask = dataset[idx]
        src = src.unsqueeze(0).to(device)
        src_pad_mask = src_pad_mask.unsqueeze(0).to(device)

        pred = greedy_decode(model, src, src_pad_mask, start_token, pad_idx)

        # Convert tokens back to strings using your vocab
        gt_tokens = [dataset.vocab.get_token(t.item()) for t in tgt_out]
        pred_tokens = [dataset.vocab.get_token(t.item()) for t in pred]

        print("Ground Truth:", gt_tokens)
        print("Predicted:", pred_tokens)

        visualize_func(gt_tokens, pred_tokens)

# TRAIN + VALIDATION (batched greedy decoding)

# Config 
EPOCHS = 20
LR = 1e-4
BATCH_SIZE = 64           # tune for GPU memory
NUM_WORKERS = 4
PIN_MEMORY = True
TRAIN_GREEDY_SAMPLE = 256   # number of train samples to evaluate greedily per epoch
CHECKPOINT_DIR = "/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Re-create dataloaders with performance flags
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          collate_fn=collate_fn, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                          collate_fn=collate_fn, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                          collate_fn=collate_fn, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

# Instantiate model 
vocab_size = len(vocab.idx2token)
model = MazeTransformer(vocab_size=vocab_size, d_model=128, nhead=8,
                        num_layers=6, dim_ff=512, dropout=0.1, max_len=300).to(device)

# Batched (vectorized) greedy decode function for MazeTransformer (batch-first=True)
def greedy_decode_batched(model, src, src_pad_mask, start_token, end_token, pad_idx, max_len=300, device=device):
    """
    Vectorized greedy decode for an entire batch.
    Inputs:
      - model: MazeTransformer (uses batch_first=True)
      - src: LongTensor [B, S]
      - src_pad_mask: Bool tensor [B, S] (True where pad)
      - start_token, end_token, pad_idx: ints
      - max_len: max decode length (including start, excluding maybe)
    Returns:
      - pred_batch: LongTensor [B, T_dec] (includes start token as first column)
    """
    real_model = model.module if hasattr(model, "module") else model
    real_model.eval()
    B, S = src.size()
    with torch.no_grad():
        # Encode once
        src = src.to(device)
        src_pad_mask = src_pad_mask.to(device)
        # Embedding + positional
        src_emb = real_model.embed(src) * (real_model.d_model ** 0.5) + real_model.pos[:, :S, :].to(device)
        # memory: shape [B, S, d_model]
        memory = real_model.encoder(src_emb, src_key_padding_mask=src_pad_mask)

        # Initialize decoder input with start tokens
        ys = torch.full((B, 1), start_token, dtype=torch.long, device=device)  # [B, 1]
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        # collect tokens step by step
        for step in range(max_len):
            # Prepare tgt_emb for current ys
            T = ys.size(1)
            tgt_emb = real_model.embed(ys) * (real_model.d_model ** 0.5) + real_model.pos[:, :T, :].to(device)  # [B, T, d]
            # subsequent mask (PyTorch expects shape [T, T])
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(T).to(device)
            # No tgt_key_padding_mask needed because ys contains no padding
            out = real_model.decoder(tgt_emb, memory, tgt_mask=tgt_mask,
                                     memory_key_padding_mask=src_pad_mask)
            # out: [B, T, d_model] -> take last time-step
            last = out[:, -1, :]  # [B, d_model]
            logits = real_model.fc_out(last)  # [B, V]
            next_tok = logits.argmax(dim=-1)  # [B]
            ys = torch.cat([ys, next_tok.unsqueeze(1)], dim=1)  # [B, T+1]

            # mark finished sequences and optionally stop early if all finished
            newly_finished = (next_tok == end_token)
            finished = finished | newly_finished
            if finished.all():
                break

        # return predictions (includes start_token at position 0)
        return ys  # [B, T_out]

# Batched greedy evaluation over a loader (full val or subset)
def run_batched_greedy_eval(model, loader, start_token, end_token, pad_idx, device, max_len=300):
    """
    For each batch in loader, run greedy_decode_batched and compute:
      - token_correct, total_tokens
      - seq_exact_matches, total_examples
      - accumulate tokens for micro precision/recall/f1 if needed (returns lists)
    """
    model.eval()
    total_token_correct = 0
    total_tokens = 0
    total_seq_correct = 0
    total_examples = 0

    all_preds_flat = []
    all_targets_flat = []

    with torch.no_grad():
        for batch in loader:
            src_batch, tgt_in_batch, tgt_out_batch, src_pad_mask_batch, tgt_pad_mask_batch = batch
            # shapes: src [B,S], tgt_out [B, T]
            src_batch = src_batch.to(device)
            src_pad_mask_batch = src_pad_mask_batch.to(device)
            tgt_out_batch = tgt_out_batch.to(device)

            # Run vectorized greedy decode
            pred_batch = greedy_decode_batched(model, src_batch, src_pad_mask_batch,
                                              start_token, end_token, pad_idx, max_len=max_len, device=device)
            # pred_batch: [B, Tpred] with start_token at [:,0]
            B = pred_batch.size(0)
            for i in range(B):
                pred_seq = pred_batch[i].cpu().tolist()
                # trim leading start token if present
                if len(pred_seq) > 0 and pred_seq[0] == start_token:
                    pred_seq = pred_seq[1:]
                # Trim pred at path_end if present
                if vocab.path_end_index in pred_seq:
                    idx = pred_seq.index(vocab.path_end_index)
                    pred_trim = pred_seq[: idx + 1]
                else:
                    pred_trim = pred_seq

                # gold trimming: tgt_out_batch includes PATH_END and pads
                gold = tgt_out_batch[i].cpu().tolist()
                # trim gold at PATH_END (include it)
                if vocab.path_end_index in gold:
                    gi = gold.index(vocab.path_end_index)
                    gold_trim = gold[: gi + 1]
                else:
                    # remove pad tokens from end
                    gold_trim = [t for t in gold if t != pad_idx]

                # Token-level comparison up to min length
                L = min(len(pred_trim), len(gold_trim))
                for k in range(L):
                    if pred_trim[k] == gold_trim[k]:
                        total_token_correct += 1
                    all_preds_flat.append(pred_trim[k])
                    all_targets_flat.append(gold_trim[k])
                    total_tokens += 1

                # Sequence exact-match: lengths equal and all tokens equal
                if (len(pred_trim) == len(gold_trim)) and (pred_trim == gold_trim):
                    total_seq_correct += 1

                total_examples += 1

    return total_token_correct, total_tokens, total_seq_correct, total_examples, all_preds_flat, all_targets_flat

# Training loop (no greedy inside batch loop)
pad_idx = vocab.pad_index
start_token = vocab.path_start_index
end_token = vocab.path_end_index

optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
scaler = GradScaler()  # AMP

best_val_seq = -1.0
best_epoch = -1

train_losses, val_losses = [], []
train_tf_token_accs, val_tf_token_accs = [], []
train_seq_accs_sample, val_seq_accs = [], []

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    model.train()
    running_loss = 0.0
    num_batches = 0

    for batch in train_loader:
        src, tgt_in, tgt_out, src_pad_mask, tgt_pad_mask = batch
        src = src.to(device, non_blocking=True)
        tgt_in = tgt_in.to(device, non_blocking=True)
        tgt_out = tgt_out.to(device, non_blocking=True)
        src_pad_mask = src_pad_mask.to(device, non_blocking=True)
        tgt_pad_mask = tgt_pad_mask.to(device, non_blocking=True)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_in.size(1)).to(device)

        optimizer.zero_grad()
        with autocast():
            logits = model(src, tgt_in, None, tgt_mask, src_pad_mask, tgt_pad_mask)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        num_batches += 1

    avg_train_loss = running_loss / max(1, num_batches)
    train_losses.append(avg_train_loss)

    # Quick teacher-forced token accuracy (proxy) computed on last batch if available
    with torch.no_grad():
        try:
            preds_tf = logits.argmax(dim=-1)  # [B,T]
            mask = (tgt_out != pad_idx)
            correct = ((preds_tf == tgt_out) & mask).sum().item()
            total = mask.sum().item()
            train_tf_token_acc = correct / total if total > 0 else 0.0
        except Exception:
            train_tf_token_acc = 0.0
    train_tf_token_accs.append(train_tf_token_acc)

    # Validation: teacher-forced loss computed quickly in eval mode
    model.eval()
    val_running_loss = 0.0
    val_batches = 0
    with torch.no_grad():
        for batch in val_loader:
            src, tgt_in, tgt_out, src_pad_mask, tgt_pad_mask = batch
            src = src.to(device, non_blocking=True)
            tgt_in = tgt_in.to(device, non_blocking=True)
            tgt_out = tgt_out.to(device, non_blocking=True)
            src_pad_mask = src_pad_mask.to(device, non_blocking=True)
            tgt_pad_mask = tgt_pad_mask.to(device, non_blocking=True)

            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_in.size(1)).to(device)
            logits = model(src, tgt_in, None, tgt_mask, src_pad_mask, tgt_pad_mask)
            val_running_loss += criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1)).item()
            val_batches += 1

    avg_val_loss = val_running_loss / max(1, val_batches)
    val_losses.append(avg_val_loss)

    #Autoregressive metrics via batched greedy for validation 
    val_token_correct, val_total_tokens, val_seq_correct, val_total_examples, val_preds_flat, val_targets_flat = \
        run_batched_greedy_eval(model, val_loader, start_token, end_token, pad_idx, device, max_len=300)

    val_token_acc = val_token_correct / val_total_tokens if val_total_tokens > 0 else 0.0
    val_seq_acc = val_seq_correct / val_total_examples if val_total_examples > 0 else 0.0
    val_tf_token_accs.append(val_token_acc)
    val_seq_accs.append(val_seq_acc)

    # Autoregressive metrics on small training subset 
    n_train = len(train_dataset)
    sample_n = min(TRAIN_GREEDY_SAMPLE, n_train)
    sample_indices = random.sample(range(n_train), sample_n)
    sample_loader = DataLoader(Subset(train_dataset, sample_indices), batch_size=BATCH_SIZE,
                               shuffle=False, collate_fn=collate_fn, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    tr_token_correct, tr_total_tokens, tr_seq_correct, tr_total_examples, _, _ = \
        run_batched_greedy_eval(model, sample_loader, start_token, end_token, pad_idx, device, max_len=300)
    tr_seq_acc_sample = tr_seq_correct / tr_total_examples if tr_total_examples > 0 else 0.0
    train_seq_accs_sample.append(tr_seq_acc_sample)

    # Save checkpoint and best model by val seq acc
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"transformer_epoch_{epoch}.pt")
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if 'optimizer' in locals() else None,
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss,
        "vocab": vocab.token2idx
    }, ckpt_path)

    if val_seq_acc > best_val_seq:
        best_val_seq = val_seq_acc
        best_epoch = epoch
        best_path = os.path.join(CHECKPOINT_DIR, "best_transformer.pt")
        torch.save(model.state_dict(), best_path)

    # Logging
    epoch_time = time.time() - t0
    print(f"Epoch {epoch}/{EPOCHS}  time={epoch_time:.1f}s")
    print(f"  Train Loss: {avg_train_loss:.4f}  Train TF TokenAcc(proxy): {train_tf_token_acc:.4f}  Train SeqAcc(sample): {tr_seq_acc_sample:.4f}")
    print(f"  Val   Loss: {avg_val_loss:.4f}  Val TokenAcc: {val_token_acc:.4f}  Val SeqAcc: {val_seq_acc:.4f}")
    print(f"  Best Val SeqAcc so far: {best_val_seq:.4f} (epoch {best_epoch})")
    print("-" * 80)

# Save metrics
metrics = {
    "train_losses": train_losses, "val_losses": val_losses,
    "train_tf_token_accs": train_tf_token_accs, "train_seq_accs_sample": train_seq_accs_sample,
    "val_token_accs": val_tf_token_accs, "val_seq_accs": val_seq_accs,
    "best_val_seq": best_val_seq, "best_epoch": best_epoch
}
with open(os.path.join(CHECKPOINT_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

print("Training complete. Best val seq acc:", best_val_seq, " (epoch", best_epoch, ")")
print("Best model saved to:", os.path.join(CHECKPOINT_DIR, "best_transformer.pt"))

# SAVE FINAL TRANSFORMER WEIGHTS

# Load best checkpoint weights 
best_transformer_path = os.path.join(CHECKPOINT_DIR, "best_transformer.pt")

# Save a second copy
submission_transformer_path = os.path.join(CHECKPOINT_DIR, "transformer_submission_weights.pt")

# Copy weights
state_dict = torch.load(best_transformer_path, map_location="cpu")
torch.save(state_dict, submission_transformer_path)

print("Saved submission transformer weights to:", submission_transformer_path)

# CREATE EMPTY RNN WEIGHTS FILE
rnn_submission_path = os.path.join(CHECKPOINT_DIR, "rnn_submission_weights.pt")
torch.save({}, rnn_submission_path)  # empty dictionary as placeholder

print("Saved submission RNN weights placeholder to:", rnn_submission_path)


# TEST + VISUALIZE (batched greedy decoding)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "/checkpoints"
best_ckpt = os.path.join(CHECKPOINT_DIR, "best_transformer.pt")
assert os.path.exists(best_ckpt), "Best checkpoint not found. Run training cell first."

# Recreate model and load state
vocab_size = len(vocab.idx2token)
model = MazeTransformer(vocab_size=vocab_size, d_model=128, nhead=8,
                        num_layers=6, dim_ff=512, dropout=0.1, max_len=300).to(device)

model.load_state_dict(torch.load(best_ckpt, map_location=device))
model.eval()

pad_idx = vocab.pad_index
start_token = vocab.path_start_index
end_token = vocab.path_end_index

# Evaluate test set using batched greedy decode
t0 = time.time()
test_token_correct, test_total_tokens, test_seq_correct, test_total_examples, all_preds_flat, all_targets_flat = \
    run_batched_greedy_eval(model, test_loader, start_token, end_token, pad_idx, device, max_len=300)

test_token_acc = test_token_correct / test_total_tokens if test_total_tokens > 0 else 0.0
test_seq_acc = test_seq_correct / test_total_examples if test_total_examples > 0 else 0.0

# micro precision/recall/f1 on token-level
precision, recall, f1, _ = precision_recall_fscore_support(all_targets_flat, all_preds_flat, average="micro", zero_division=0)

elapsed = time.time() - t0
print("---- TEST METRICS (batched greedy) ----")
print(f"Time: {elapsed:.1f}s for test set decoding")
print(f"Token Accuracy : {test_token_acc:.6f}")
print(f"Seq Accuracy   : {test_seq_acc:.6f}")
print(f"Precision (micro): {precision:.6f}")
print(f"Recall    (micro): {recall:.6f}")
print(f"F1        (micro): {f1:.6f}")
print("Test examples:", test_total_examples)

# PLOT TRAIN+VAL LINES + TEST ACC
metrics_path = os.path.join(CHECKPOINT_DIR, "metrics.json")
if os.path.exists(metrics_path):
    with open(metrics_path, "r") as f:
        M = json.load(f)
else:
    raise FileNotFoundError("metrics.json not found. Ensure training cell saved it.")


epochs = list(range(1, len(M["val_losses"]) + 1))

plt.figure(figsize=(16,5))

# Loss
plt.subplot(1,3,1)
plt.plot(epochs, M["train_losses"], label="Train Loss")
plt.plot(epochs, M["val_losses"], label="Val Loss")
plt.xlabel("Epoch")
plt.title("Loss Curve")
plt.legend()

# Token Acc
plt.subplot(1,3,2)
plt.plot(epochs, M["train_tf_token_accs"], label="Train Token Acc (TF)")
plt.plot(epochs, M["val_token_accs"], label="Val Token Acc (Greedy)")
plt.xlabel("Epoch")
plt.title("Token Accuracy")
plt.legend()

# Sequence Accuracy
plt.subplot(1,3,3)
plt.plot(epochs, M["train_seq_accs_sample"], label="Train Seq Acc (sampled)")
plt.plot(epochs, M["val_seq_accs"], label="Val Seq Acc (full)")
plt.axhline(test_seq_acc, color="red", linestyle="--", label="Test Seq Acc")
plt.xlabel("Epoch")
plt.title("Sequence Accuracy")
plt.legend()

plt.show()


# VISUALIZE 3 RANDOM VALIDATION EXAMPLES

print("\n========== VISUALIZATION SAMPLES ==========")

def decode_trim_pred(pred):
    """Remove <PATH_START> and trim at <PATH_END>"""
    if len(pred) > 0 and pred[0] == start_token:
        pred = pred[1:]
    if end_token in pred:
        pred = pred[: pred.index(end_token)+1]
    return pred

def decode_trim_gt(gt):
    """Trim ground truth at PATH_END"""
    if end_token in gt:
        gt = gt[: gt.index(end_token)+1]
    return gt

for k in range(3):
    idx = random.randint(0, len(val_dataset) - 1)

    # dataset returns ONLY: (src, tgt_in, tgt_out)
    src, tgt_in, tgt_out = val_dataset[idx]

    src      = src.unsqueeze(0).to(device)
    src_mask = (src == pad_idx).to(device)

    pred_batch = greedy_decode_batched(
        model,
        src,
        src_mask,
        start_token,
        end_token,
        pad_idx,
        max_len=300,
        device=device
    )

    pred = decode_trim_pred(pred_batch[0].cpu().tolist())
    gt   = decode_trim_gt(tgt_out.tolist())

    pred_tokens = [vocab.idx2token[x] for x in pred]
    gt_tokens   = [vocab.idx2token[x] for x in gt]

    print(f"\nSample {k+1} (index={idx}):")
    print("GT:   ", gt_tokens)
    print("Pred: ", pred_tokens)

    # If user has a visualize() function in notebook
    if "visualize_func" in globals() and visualize_func is not None:
        try:
            visualize_func(gt_tokens, pred_tokens)
        except Exception as e:
            print("Visualization failed:", e)

print("\n========== DONE ==========\n")