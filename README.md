# Assignment 06: RNN Foundations in PyTorch

## Goal

Learn what an RNN hidden state is, how PyTorch represents RNN inputs/outputs, and how to use a final hidden state for sequence classification.

## What you will submit

- One notebook or Python script that runs end-to-end
- Manual RNN calculation from Part A
- Printed tensor shapes from Part B
- DNA classifier results from Part C
- Answers to the follow-up questions for understanding

---

## Part A: Manual RNN Calculation

Use:

```text
x_1 = [1.0, 0.0, 0.0]
x_2 = [0.0, 1.0, 0.0]
x_3 = [0.0, 0.0, 1.0]
x_4 = [1.0, 1.0, 0.0]
h_0 = [0.0, 0.0]
```

Update:

```text
h_t = tanh(W_x x_t + W_h h_{t-1} + b)
```

Weights:

```text
W_x = [[ 0.5, -0.2,  0.1],
       [ 0.0,  0.3, -0.4]]

W_h = [[ 0.2,  0.1],
       [-0.3,  0.4]]

b = [0.0, 0.1]
```

Compute `h_1`, `h_2`, `h_3`, and `h_4`.

---

## Part B: Reproduce the Same RNN in PyTorch

Create an input with shape:

```text
(batch_size, seq_len, input_size) = (1, 4, 3)
```

Use `nn.RNN(input_size=3, hidden_size=2, batch_first=True)` and print:

```python
print(x.shape)
print(output.shape)
print(h_n.shape)
print(output)
print(h_n)
```

Answer:

1. What does `output[:, -1, :]` represent?
2. What does `h_n[-1]` represent?
3. Why are they usually the same for a one-layer RNN?
4. What changes if `batch_first=False`?

---

## Part C: DNA Sequence Classifier

Generate synthetic DNA sequences. The label is:

```text
1 = sequence contains a motif
0 = sequence does not contain the motif
```

Use nucleotides:

```text
A, C, G, T
```

Choose one motif, for example `ATG`.

### C1. Generate Data

Create random DNA sequences with length between 30 and 80.

Generate:

- 2000 training examples
- 500 validation examples
- 500 test examples

Use:

```python
char2idx = {"A": 0, "C": 1, "G": 2, "T": 3, "<PAD>": 4}
```

Positive examples must contain the motif. Negative examples must not contain the motif.

### C2. Encode and Pad

Convert strings to integer IDs:

```text
"ATGCA" -> [0, 3, 2, 1, 0]
```

Pad variable-length sequences in each batch.

Required shapes:

```text
x_batch: (batch_size, max_seq_len)
lengths: (batch_size,)
y_batch: (batch_size,)
```

### C3. Train a Classifier

Model:

```text
DNA ids -> embedding -> RNN -> final hidden state -> classifier
```

Suggested model:

```python
class DNAClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_classes, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.rnn = nn.RNN(embed_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        packed_output, h_n = self.rnn(packed)
        last_hidden = h_n[-1]
        return self.fc(last_hidden)
```

Packing prevents the model from treating padding tokens as part of the DNA sequence.

Report train, validation, and test accuracy.

### C4. Generalization Tests

Test examples where the motif appears:

- near the beginning
- in the middle
- near the end
- in a longer sequence
- with noisy repeated characters

Print sequence, true label, predicted label, and probability.

### Follow-Up Questions for Understanding

Answer:

1. Why do we use `nn.Embedding` before the RNN?
2. Why is this a many-to-one task?
3. Why do we use the final hidden state for classification?
4. Why do we pad sequences in a batch?
5. Why does `pack_padded_sequence` help when sequences have different lengths?
6. What kind of information might a vanilla RNN forget on long sequences?
7. What problem are LSTMs and GRUs designed to reduce compared with a vanilla RNN?
8. What do gates help an LSTM or GRU decide while reading a sequence?
9. If you replaced `nn.RNN` with `nn.LSTM`, what extra value does PyTorch return?
