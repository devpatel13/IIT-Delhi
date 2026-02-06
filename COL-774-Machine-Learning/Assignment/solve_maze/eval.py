import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import ast

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# VOCABULARIES

# RNN VOCAB 
class RNNVocabulary:
    def __init__(self):
        self.stoi = {"<pad>":0, "<sos>":1, "<eos>":2}
        self.itos = {0:"<pad>", 1:"<sos>", 2:"<eos>"}
        coords = [f"({r},{c})" for r in range(6) for c in range(6)]
        tags = [
            "<ADJLIST_START>", "<ADJLIST_END>",
            "<ORIGIN_START>", "<ORIGIN_END>",
            "<TARGET_START>", "<TARGET_END>",
            "<PATH_START>", "<PATH_END>",
            "<-->", ";"
        ]
        for tok in sorted(coords + tags):
            if tok not in self.stoi:
                idx = len(self.stoi)
                self.stoi[tok] = idx
                self.itos[idx] = tok

# TRANSFORMER VOCAB
class TransformerVocabulary:
    def __init__(self):
        self.token2idx = {}
        self.idx2token = []

        specials = ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<PATH_START>", "<PATH_END>"]
        for tok in specials:
            self.add(tok)

        self.pad_index = self.token2idx["<PAD>"]
        self.unk_index = self.token2idx["<UNK>"]
        self.path_start_index = self.token2idx["<PATH_START>"]
        self.path_end_index = self.token2idx["<PATH_END>"]

        coords = [f"({r},{c})" for r in range(6) for c in range(6)]
        tags = [
            "<ADJLIST_START>", "<ADJLIST_END>",
            "<ORIGIN_START>", "<ORIGIN_END>",
            "<TARGET_START>", "<TARGET_END>",
            "<-->", ";"
        ]
        tokens = sorted(list(set(coords + tags) - set(specials)))
        for t in tokens: self.add(t)

    def add(self, tok):
        if tok not in self.token2idx:
            self.token2idx[tok] = len(self.idx2token)
            self.idx2token.append(tok)

    def encode(self, seq):
        return [self.token2idx.get(tok, self.unk_index) for tok in seq]

    def get_token(self, idx):
        return self.idx2token[idx] if 0 <= idx < len(self.idx2token) else "<UNK>"



# ---------- RNN ----------
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.RNN(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        emb = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(emb)
        return outputs, hidden

class Attention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.attn = nn.Linear(hid_dim * 2, hid_dim)
        self.v = nn.Linear(hid_dim, 1, bias=False)

    def forward(self, hidden, enc_outputs):
        B, S, H = enc_outputs.shape
        hidden = hidden.unsqueeze(1).repeat(1, S, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, enc_outputs), dim=2)))
        scores = self.v(energy).squeeze(2)
        return torch.softmax(scores, dim=1)

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.RNN(hid_dim + emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hid_dim * 2 + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_tok, hidden, enc_outputs):
        input_tok = input_tok.unsqueeze(1)
        emb = self.dropout(self.embedding(input_tok))
        attn = self.attention(hidden[-1], enc_outputs).unsqueeze(1)
        context = torch.bmm(attn, enc_outputs)
        rnn_in = torch.cat((emb, context), dim=2)
        output, hidden = self.rnn(rnn_in, hidden)
        pred = self.fc_out(torch.cat((output.squeeze(1), context.squeeze(1), emb.squeeze(1)), dim=1))
        return pred, hidden

class Seq2Seq(nn.Module):
    def __init__(self, enc, dec):
        super().__init__()
        self.encoder = enc
        self.decoder = dec

# Greedy decode RNN
def rnn_generate(model, src, vocab, max_len=200):
    model.eval()
    with torch.no_grad():
        enc_out, hidden = model.encoder(src)
        inp = torch.tensor([vocab.stoi["<sos>"]], device=device)
        preds = []
        for _ in range(max_len):
            out, hidden = model.decoder(inp, hidden, enc_out)
            nxt = out.argmax(1)
            if nxt.item() == vocab.stoi["<eos>"]: break
            preds.append(nxt.item())
            inp = nxt
        return preds


# TRANSFORMER
def positional_encoding(max_len, d_model):
    pe = torch.zeros(max_len, d_model)
    pos = torch.arange(0, max_len).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.unsqueeze(0)

class MazeTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=8, 
                 num_layers=6, dim_ff=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = positional_encoding(300, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_ff, dropout, batch_first=True)
        dec_layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_ff, dropout, batch_first=True)

        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

# Greedy decode transformer
def transformer_generate(model, src, vocab, max_len=300):
    model.eval()
    with torch.no_grad():
        src_emb = model.embed(src) * np.sqrt(model.d_model) + model.pos[:, :src.size(1)]
        memory = model.encoder(src_emb)

        ys = torch.tensor([[vocab.path_start_index]], device=device)
        for _ in range(max_len):
            tgt_emb = model.embed(ys) * np.sqrt(model.d_model) + model.pos[:, :ys.size(1)]
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(ys.size(1)).to(device)
            out = model.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
            nxt = model.fc_out(out[:, -1]).argmax(dim=-1)
            if nxt.item() == vocab.path_end_index: break
            ys = torch.cat([ys, nxt.unsqueeze(0)], dim=1)

        return ys.squeeze(0).tolist()[1:]

# 3. MAIN SCRIPT

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python eval.py model.pt rnn/transformer input.csv output.csv")
        sys.exit(1)

    model_path = sys.argv[1]
    model_type = sys.argv[2].lower()
    data_path  = sys.argv[3]
    out_path   = sys.argv[4]

    print("Device:", device)

    # Load CSV
    DATA = pd.read_csv(data_path)
    DATA["input_sequence"] = DATA["input_sequence"].apply(ast.literal_eval)

    predictions = []

    # RNN INFERENCE
    if model_type == "rnn":
        vocab = RNNVocabulary()
        INPUT_DIM = len(vocab.stoi)
        OUTPUT_DIM = len(vocab.stoi)
        enc = Encoder(INPUT_DIM, 128, 512, 2, 0.3)
        attn = Attention(512)
        dec = Decoder(OUTPUT_DIM, 128, 512, 2, 0.3, attn)
        model = Seq2Seq(enc, dec).to(device)

        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        model.eval()

        for _, row in DATA.iterrows():
            seq = row["input_sequence"]
            idxs = [vocab.stoi.get(t, 0) for t in seq]
            src = torch.tensor(idxs, dtype=torch.long).unsqueeze(0).to(device)
            out = rnn_generate(model, src, vocab)
            predictions.append([vocab.itos[i] for i in out])

    # TRANSFORMER INFERENCE
    elif model_type == "transformer":
        vocab = TransformerVocabulary()
        model = MazeTransformer(len(vocab.idx2token)).to(device)

        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        model.eval()

        for _, row in DATA.iterrows():
            seq = row["input_sequence"]
            idxs = vocab.encode(seq)
            src = torch.tensor(idxs, dtype=torch.long).unsqueeze(0).to(device)
            out = transformer_generate(model, src, vocab)
            predictions.append([vocab.get_token(i) for i in out])

    else:
        print("Invalid model type:", model_type)
        sys.exit(1)

    # SAVE
    DATA["output_path"] = predictions
    DATA.to_csv(out_path, index=False)
    print("Saved output to:", out_path)
