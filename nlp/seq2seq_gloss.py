"""
ISL Pipeline — Seq2Seq Transformer: English → ISL Gloss
========================================================
Replaces the rule-based gloss_converter.py with a learned model.

Pipeline:
  1. generate_data()   — creates training pairs using rule-based converter
  2. train()           — trains the Transformer
  3. translate()       — converts new English sentences to ISL gloss

Usage:
  python seq2seq_gloss.py --mode train    # train the model
  python seq2seq_gloss.py --mode test     # test on sample sentences
  python seq2seq_gloss.py --mode convert --sentence "i do not understand"
"""

import json
import math
import random
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
MODEL_PATH  = BASE_DIR / "seq2seq_model.pt"
DATA_PATH   = BASE_DIR / "gloss_output.json"   # from rule-based converter
VOCAB_PATH  = BASE_DIR / "seq2seq_vocab.json"

# ── Special tokens ─────────────────────────────────────────────────────────────
PAD   = "<PAD>"   # padding
SOS   = "<SOS>"   # start of sequence
EOS   = "<EOS>"   # end of sequence
UNK   = "<UNK>"   # unknown word

# ── Hyperparameters ────────────────────────────────────────────────────────────
EMB_DIM     = 256    # embedding dimension
N_HEADS     = 4      # attention heads (EMB_DIM must be divisible by N_HEADS)
N_LAYERS    = 3      # encoder/decoder layers
FF_DIM      = 512    # feedforward layer size
DROPOUT     = 0.1    # reverted — 0.3 caused underfitting
MAX_LEN     = 50     # max sequence length
BATCH_SIZE  = 64
EPOCHS      = 150
LR          = 0.0003
BEAM_SIZE   = 5      # beam search width

# ═══════════════════════════════════════════════════════════════════════════════
# VOCABULARY
# ═══════════════════════════════════════════════════════════════════════════════

class Vocabulary:
    def __init__(self):
        self.word2idx = {PAD: 0, SOS: 1, EOS: 2, UNK: 3}
        self.idx2word = {0: PAD, 1: SOS, 2: EOS, 3: UNK}
        self.n_words   = 4

    def add_sentence(self, sentence: str):
        for word in sentence.lower().split():
            self.add_word(word)

    def add_word(self, word: str):
        if word not in self.word2idx:
            self.word2idx[word] = self.n_words
            self.idx2word[self.n_words] = word
            self.n_words += 1

    def encode(self, sentence: str) -> list[int]:
        return [self.word2idx.get(w.lower(), self.word2idx[UNK])
                for w in sentence.split()]

    def decode(self, indices: list[int]) -> str:
        words = []
        for idx in indices:
            word = self.idx2word.get(idx, UNK)
            if word in (PAD, SOS, EOS):
                continue
            words.append(word)
        return " ".join(words)

    def save(self, path: Path):
        data = {"word2idx": self.word2idx, "idx2word": {str(k): v for k,v in self.idx2word.items()}, "n_words": self.n_words}
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path):
        v = cls()
        data = json.loads(path.read_text())
        v.word2idx = data["word2idx"]
        v.idx2word = {int(k): val for k, val in data["idx2word"].items()}
        v.n_words  = data["n_words"]
        return v

# ═══════════════════════════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════════════════════════

class GlossDataset(Dataset):
    def __init__(self, pairs, src_vocab, tgt_vocab, max_len=MAX_LEN):
        self.pairs     = pairs
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len   = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src_str, tgt_str = self.pairs[idx]

        src = self.src_vocab.encode(src_str)[:self.max_len]
        tgt = self.tgt_vocab.encode(tgt_str)[:self.max_len]

        # Add SOS and EOS to target
        tgt = [self.tgt_vocab.word2idx[SOS]] + tgt + [self.tgt_vocab.word2idx[EOS]]

        return torch.tensor(src, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)


def collate_fn(batch):
    """Pad sequences in a batch to the same length."""
    srcs, tgts = zip(*batch)
    src_pad = torch.zeros(len(srcs), max(s.size(0) for s in srcs), dtype=torch.long)
    tgt_pad = torch.zeros(len(tgts), max(t.size(0) for t in tgts), dtype=torch.long)
    for i, (s, t) in enumerate(zip(srcs, tgts)):
        src_pad[i, :s.size(0)] = s
        tgt_pad[i, :t.size(0)] = t
    return src_pad, tgt_pad

# ═══════════════════════════════════════════════════════════════════════════════
# POSITIONAL ENCODING
# ═══════════════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """Adds position information to embeddings so the model knows word order."""
    def __init__(self, emb_dim, max_len=MAX_LEN, dropout=DROPOUT):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, emb_dim)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, emb_dim, 2).float() * (-math.log(10000.0) / emb_dim))

        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)  # (1, max_len, emb_dim)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# ═══════════════════════════════════════════════════════════════════════════════
# TRANSFORMER MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class ISLTransformer(nn.Module):
    """
    Seq2Seq Transformer for English → ISL Gloss translation.

    Architecture:
      - Source embedding + positional encoding
      - Transformer encoder (N_LAYERS layers, each with multi-head self-attention + FFN)
      - Transformer decoder (N_LAYERS layers, each with masked self-attention + cross-attention + FFN)
      - Linear projection to target vocabulary size
    """
    def __init__(self, src_vocab_size, tgt_vocab_size,
                 emb_dim=EMB_DIM, n_heads=N_HEADS,
                 n_layers=N_LAYERS, ff_dim=FF_DIM,
                 dropout=DROPOUT, max_len=MAX_LEN):
        super().__init__()

        # Embeddings
        self.src_embed = nn.Embedding(src_vocab_size, emb_dim, padding_idx=0)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, emb_dim, padding_idx=0)
        self.pos_enc   = PositionalEncoding(emb_dim, max_len, dropout)

        # Core Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=n_heads,
            dim_feedforward=ff_dim, dropout=dropout, batch_first=True
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=emb_dim, nhead=n_heads,
            dim_feedforward=ff_dim, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # Output projection
        self.fc_out = nn.Linear(emb_dim, tgt_vocab_size)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_src_key_padding_mask(self, src):
        """Mask padding tokens in source."""
        return src == 0  # True where padded

    def make_tgt_mask(self, tgt):
        """Causal mask — decoder can't see future tokens."""
        sz = tgt.size(1)
        return torch.triu(torch.ones(sz, sz, device=tgt.device), diagonal=1).bool()

    def forward(self, src, tgt):
        src_pad_mask = self.make_src_key_padding_mask(src)
        tgt_mask     = self.make_tgt_mask(tgt)

        # Encode source
        src_emb = self.pos_enc(self.src_embed(src))
        memory  = self.encoder(src_emb, src_key_padding_mask=src_pad_mask)

        # Decode
        tgt_emb = self.pos_enc(self.tgt_embed(tgt))
        out     = self.decoder(
            tgt_emb, memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=src_pad_mask
        )

        return self.fc_out(out)  # (batch, tgt_len, tgt_vocab_size)

# ═══════════════════════════════════════════════════════════════════════════════
# DATA GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def load_pairs_from_gloss_output() -> list[tuple[str, str]]:
    """Load English-ISL pairs from training_pairs.json + gloss_output.json"""
    pairs = []

    # Load from training_pairs.json (Wikipedia scraped data)
    training_path = BASE_DIR / "training_pairs.json"
    if training_path.exists():
        data = json.loads(training_path.read_text())
        for seg in data:
            src = seg.get("english", "").strip()
            tgt = seg.get("gloss", "").strip()
            # Filter: keep only clean short sentences
            if src and tgt and 3 <= len(src.split()) <= 12 and len(tgt.split()) >= 2:
                pairs.append((src, tgt.lower()))
        print(f"[SEQ2SEQ] Loaded {len(pairs)} pairs from training_pairs.json")

    # Also load from gloss_output.json (lecture data)
    if DATA_PATH.exists():
        data = json.loads(DATA_PATH.read_text())
        lecture_pairs = 0
        for seg in data:
            src = seg.get("clean", "").strip()
            tgt = seg.get("gloss", "").strip()
            if src and tgt and len(src.split()) >= 2:
                pairs.append((src, tgt.lower()))
                lecture_pairs += 1
        print(f"[SEQ2SEQ] Added {lecture_pairs} pairs from gloss_output.json")

    print(f"[SEQ2SEQ] Total: {len(pairs)} pairs")
    return pairs


def augment_pairs(pairs: list[tuple[str,str]], factor: int = 5) -> list[tuple[str,str]]:
    """
    Bootstrap: augment training data by importing the rule-based converter
    and generating gloss for additional sentences derived from the corpus.
    This multiplies training data without needing manual annotation.
    """
    import sys
    sys.path.insert(0, str(BASE_DIR))

    try:
        from gloss_converter import convert_sentence
        augmented = list(pairs)

        for src, _ in pairs:
            words = src.split()
            if len(words) < 3:
                continue
            # Generate sub-sentence variants
            for _ in range(factor - 1):
                start = random.randint(0, len(words) - 2)
                end   = random.randint(start + 1, min(start + 8, len(words)))
                sub   = " ".join(words[start:end])
                gloss = convert_sentence(sub)
                if gloss and gloss != sub.upper():
                    augmented.append((sub, gloss.lower()))

        random.shuffle(augmented)
        print(f"[SEQ2SEQ] Augmented to {len(augmented)} pairs")
        return augmented

    except ImportError:
        print("[SEQ2SEQ] Could not import gloss_converter — using original pairs only")
        return pairs

# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def train():
    print("[SEQ2SEQ] Starting training…")

    # Load and augment data
    pairs = load_pairs_from_gloss_output()
    if not pairs:
        print("[SEQ2SEQ] No training data found. Run the pipeline first.")
        return

    pairs = augment_pairs(pairs, factor=6)

    # Build vocabularies
    src_vocab = Vocabulary()
    tgt_vocab = Vocabulary()
    for src, tgt in pairs:
        src_vocab.add_sentence(src)
        tgt_vocab.add_sentence(tgt)

    print(f"[SEQ2SEQ] Source vocab: {src_vocab.n_words} words")
    print(f"[SEQ2SEQ] Target vocab: {tgt_vocab.n_words} words")

    # Save vocabularies
    src_vocab.save(BASE_DIR / "src_vocab.json")
    tgt_vocab.save(BASE_DIR / "tgt_vocab.json")

    # Split train/val
    random.shuffle(pairs)
    split     = int(0.9 * len(pairs))
    train_set = pairs[:split]
    val_set   = pairs[split:]

    train_ds = GlossDataset(train_set, src_vocab, tgt_vocab)
    val_ds   = GlossDataset(val_set,   src_vocab, tgt_vocab)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SEQ2SEQ] Training on {device}")

    model = ISLTransformer(src_vocab.n_words, tgt_vocab.n_words).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_loss = float('inf')

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        train_loss = 0
        for src_batch, tgt_batch in train_loader:
            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)

            # Teacher forcing: feed ground truth as decoder input
            tgt_input  = tgt_batch[:, :-1]   # all except last
            tgt_output = tgt_batch[:, 1:]    # all except first (SOS)

            logits = model(src_batch, tgt_input)
            # Reshape for cross entropy: (batch*seq, vocab)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt_output.reshape(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for src_batch, tgt_batch in val_loader:
                src_batch = src_batch.to(device)
                tgt_batch = tgt_batch.to(device)
                tgt_input  = tgt_batch[:, :-1]
                tgt_output = tgt_batch[:, 1:]
                logits = model(src_batch, tgt_input)
                loss   = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
                val_loss += loss.item()
        val_loss /= max(len(val_loader), 1)

        scheduler.step(val_loss)

        if epoch % 5 == 0:
            print(f"  Epoch {epoch:3d}/{EPOCHS} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state": model.state_dict(),
                "src_vocab":   src_vocab.word2idx,
                "tgt_vocab":   tgt_vocab.word2idx,
                "src_n_words": src_vocab.n_words,
                "tgt_n_words": tgt_vocab.n_words,
            }, MODEL_PATH)

    print(f"\n[SEQ2SEQ] Training complete. Best val loss: {best_val_loss:.4f}")
    print(f"[SEQ2SEQ] Model saved to {MODEL_PATH}")

# ═══════════════════════════════════════════════════════════════════════════════
# INFERENCE
# ═══════════════════════════════════════════════════════════════════════════════

def load_model():
    """Load trained model and vocabularies."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found — run with --mode train first")

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")

    src_vocab         = Vocabulary()
    src_vocab.word2idx = checkpoint["src_vocab"]
    src_vocab.idx2word = {v: k for k, v in checkpoint["src_vocab"].items()}
    src_vocab.n_words  = checkpoint["src_n_words"]

    tgt_vocab         = Vocabulary()
    tgt_vocab.word2idx = checkpoint["tgt_vocab"]
    tgt_vocab.idx2word = {v: k for k, v in checkpoint["tgt_vocab"].items()}
    tgt_vocab.n_words  = checkpoint["tgt_n_words"]

    model = ISLTransformer(src_vocab.n_words, tgt_vocab.n_words)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    return model, src_vocab, tgt_vocab


def translate(sentence: str, model=None, src_vocab=None, tgt_vocab=None, beam_size=BEAM_SIZE) -> str:
    """Translate one English sentence to ISL gloss using beam search."""
    if model is None:
        model, src_vocab, tgt_vocab = load_model()

    device = next(model.parameters()).device
    model.eval()

    # Encode source
    src_ids    = src_vocab.encode(sentence.lower())
    src_tensor = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(device)

    sos_idx = tgt_vocab.word2idx[SOS]
    eos_idx = tgt_vocab.word2idx[EOS]

    # Each beam: (log_prob, token_ids)
    beams     = [(0.0, [sos_idx])]
    completed = []

    with torch.no_grad():
        for _ in range(MAX_LEN):
            new_beams = []

            for log_prob, token_ids in beams:
                # If this beam already ended, keep it
                if token_ids[-1] == eos_idx:
                    completed.append((log_prob, token_ids))
                    continue

                tgt_tensor = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(device)
                logits     = model(src_tensor, tgt_tensor)

                # Get log probabilities for next token
                log_probs  = torch.log_softmax(logits[0, -1, :], dim=-1)

                # Take top beam_size candidates
                top_probs, top_ids = log_probs.topk(beam_size)

                for prob, idx in zip(top_probs.tolist(), top_ids.tolist()):
                    new_beams.append((log_prob + prob, token_ids + [idx]))

            if not new_beams:
                break

            # Keep top beam_size beams by score, normalized by length
            new_beams.sort(key=lambda x: x[0] / len(x[1]), reverse=True)
            beams = new_beams[:beam_size]

            # Stop if all beams ended
            if all(ids[-1] == eos_idx for _, ids in beams):
                completed.extend(beams)
                break

    # Pick best completed sequence, fall back to best beam
    all_candidates = completed if completed else beams
    all_candidates.sort(key=lambda x: x[0] / len(x[1]), reverse=True)
    best_ids = all_candidates[0][1]

    gloss = tgt_vocab.decode(best_ids[1:])  # skip SOS
    return gloss.upper()


def test_model():
    """Test the trained model on sample sentences."""
    model, src_vocab, tgt_vocab = load_model()

    test_sentences = [
        "i am going home",
        "she does not know the answer",
        "where are you going",
        "i do not understand neural networks",
        "the teacher is explaining the concept",
        "do you want to learn sign language",
        "what is machine learning",
        "i have been studying computer science",
    ]

    print("\n[SEQ2SEQ] Model predictions vs rule-based:\n")
    print(f"{'English':<45} {'Seq2Seq':<35} {'Rule-based'}")
    print("─" * 110)

    try:
        from gloss_converter import convert_sentence
        for sent in test_sentences:
            seq2seq   = translate(sent, model, src_vocab, tgt_vocab)
            rule_based = convert_sentence(sent).upper()
            print(f"{sent:<45} {seq2seq:<35} {rule_based}")
    except ImportError:
        for sent in test_sentences:
            seq2seq = translate(sent, model, src_vocab, tgt_vocab)
            print(f"{sent:<45} {seq2seq}")

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ISL Seq2Seq Gloss Model")
    parser.add_argument("--mode", choices=["train", "test", "convert"], default="train")
    parser.add_argument("--sentence", type=str, default="", help="Sentence to convert (use with --mode convert)")
    args = parser.parse_args()

    if args.mode == "train":
        train()
    elif args.mode == "test":
        test_model()
    elif args.mode == "convert":
        if not args.sentence:
            print("Provide a sentence with --sentence")
        else:
            result = translate(args.sentence)
            print(f"Input : {args.sentence}")
            print(f"Output: {result}")