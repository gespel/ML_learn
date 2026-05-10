# Small LLM (sllm.py) - Kleines Transformer-basiertes Language Model

Ein **funktionsfähiges Mini-Sprachmodell** basierend auf der Transformer-Architektur mit mehreren Tokenizer-Optionen und vollständiger Training/Generierungs-Pipeline.

---

## 📋 Inhaltsverzeichnis

1. [Features](#features)
2. [Tokenizer](#tokenizer)
3. [Architektur](#architektur)
4. [Training](#training)
5. [Text-Generierung](#text-generierung)
6. [Checkpoint-System](#checkpoint-system)

---

## Features

✅ **Mehrere Tokenizer-Optionen:**
- `SimpleTokenizer` - Character-Level (auf einzigartigen Zeichen)
- `WordTokenizer` - Word-Level (Split nach Satzzeichen)
- `BPETokenizer` - Byte Pair Encoding (effiziente Subword-Tokenisierung)

✅ **Transformer-Architektur:**
- Multi-Head Self-Attention
- Feed-Forward Networks
- Layer Normalization & Residual Connections
- Positional Embeddings

✅ **Robustes Training:**
- Checkpoint-System mit Mid-Epoch & End-of-Epoch Saves
- Automatische Wiederaufnahme bei Unterbruch
- Erhaltung von Loss-Metriken
- Progress-Bar mit tqdm

✅ **Text-Generierung:**
- Top-K Sampling für vielfältigere Texte
- Temperature-Kontrolle für Diversität

---

## Tokenizer

### SimpleTokenizer (Character-Level)

```python
tokenizer = SimpleTokenizer(text)
tokens = tokenizer.encode("Hello")  # → [7, 4, 2, 2, 12]
text = tokenizer.decode(tokens)      # → "Hello"
```

**Eigenschaften:**
- Vokabulgröße: Anzahl der einzigartigen Zeichen im Text
- **Vorteil:** Einfach, kann jeden Text darstellen
- **Nachteil:** Viele Tokens nötig, weniger Kontext pro Token

### WordTokenizer (Word-Level)

```python
tokenizer = WordTokenizer(text)
tokens = tokenizer.encode("Hello world")
```

**Eigenschaften:**
- Vokabulgröße: Anzahl der einzigartigen Wörter
- Split nach: `,` `;` `|` `.` ` ` `\n`
- **Vorteil:** Weniger Tokens, besseres Verständnis von Semantik
- **Nachteil:** Out-of-Vocabulary-Probleme möglich

### BPETokenizer (Byte Pair Encoding)

```python
tokenizer = BPETokenizer(text, vocab_size=10000, num_merges=None)
tokens = tokenizer.encode("Hello world")
```

**Wie BPE funktioniert:**

1. **Initialisierung:** Start mit 256 ASCII-Bytes
2. **Iteratives Merging:** Finde häufigsten Token-Pair → merge zu neuem Token
3. **Wiederholung:** Bis gewünschte Vokabulgröße erreicht

**Beispiel:**
```
Text: "hello"
Bytes: [104, 101, 108, 108, 111]

Merge 1: (108, 108) → 256
Bytes: [104, 101, 256, 111]

Merge 2: (101, 256) → 257
Bytes: [104, 257, 111]
...
```

**Eigenschaften:**
- Vokabulgröße: Konfigurierbar (Standard: 10000)
- **Vorteil:** Perfekte Balance zwischen Effizienz und Flexibilität (wie GPT)
- **Nachteil:** Trainingszeit für BPE erforderlich

---

## Architektur

### Attention Mechanism

```python
class Attention(nn.Module):
    def forward(self, query, key, value, mask=None):
        # Scaled Dot-Product Attention
        scores = Q @ K.T / sqrt(d_k)
        attention_weights = softmax(scores)
        output = attention_weights @ V
```

**Multi-Head Attention:** Mehrere Attention-Operationen parallel für verschiedene "Ansichten" des Inputs.

### TransformerBlock

```python
class TransformerBlock(nn.Module):
    # Self-Attention + Residual + LayerNorm
    # Feed-Forward + Residual + LayerNorm
```

**Aufbau:**
1. Multi-Head Self-Attention
2. Residual Connection + Layer Normalization
3. Feed-Forward Network (MLP)
4. Residual Connection + Layer Normalization

### SmallLLM (Komplettes Modell)

```python
model = SmallLLM(
    vocab_size=tokenizer.vocab_size,
    d_model=128,           # Embedding Dimension
    num_heads=4,           # Attention Heads
    num_layers=6,          # Transformer Blocks
    d_ff=512,              # Feed-Forward Dimension
    max_seq_len=128        # Maximale Sequenzlänge
)
```

**Architektur:**
```
Token Embedding + Positional Embedding
         ↓
[TransformerBlock] × 6
         ↓
Linear Layer → Output Logits (vocab_size)
```

---

## Training

### Verwendung

```bash
python sllm.py
```

### Trainingsablauf

1. **Text laden** - Alle `.txt` Dateien aus `../.storage/`
2. **Tokenizer trainieren** - SimpleTokenizer/WordTokenizer/BPETokenizer
3. **Dataset erstellen** - TextDataset mit Sequenzlänge 128
4. **Modell initialisieren** - SmallLLM Transformer
5. **Training Loop** - 100 Epochen mit Adam Optimizer

### Checkpoint-System

**Mid-Epoch Checkpoints** (alle 1000 Batches):
```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'batch_idx': batch_idx,
    'total_loss': total_loss  # Für korrekten avg_loss bei Resume
}
```

**End-of-Epoch Checkpoints:**
```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'batch_idx': 0,
    'total_loss': 0  # Neue Epoche startet bei 0
}
```

### Loss-Tracking

**Average Loss Berechnung:**
```python
avg_loss = total_loss / batch_idx
```

- Wird im Checkpoint gespeichert (`total_loss`)
- Bei Resume wird `total_loss` wiederhergestellt → avg_loss startet korrekt
- Stabiler als EMA, echtes Durchschnitts-Metric

### Metriken

```
Epoch 5/100 | Loss: 3.2451 | Avg: 3.1245 | Batch 450/2843
```

- **Loss**: Aktueller Batch Loss
- **Avg**: Durchschnittlicher Loss ab Epoch-Start (oder nach Mid-Epoch Resume)
- **Batch**: Aktueller Batch / Batches pro Epoche

---

## Text-Generierung

### Generierungscode im Training integriert

Nach dem Training werden automatisch 100 Tokens generiert:

```python
seed_text = "Baden"
for _ in range(100):
    logits = model(seed_tokens)
    next_token_logits = logits[0, -1, :] / temperature  # Temperature-Scaling
    
    # Top-K Filtering
    values, indices = torch.topk(next_token_logits, k=10)
    filtered_logits = torch.full_like(next_token_logits, float('-inf'))
    filtered_logits[indices] = values
    
    probs = torch.softmax(filtered_logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
```

### Parameter

| Parameter | Default | Effekt |
|-----------|---------|--------|
| `temperature` | 0.9 | Höher = vielfältiger, niedriger = konsistenter |
| `top_k` | 10 | Nur top-10 Tokens mit höchster Wahrscheinlichkeit berücksichtigen |

### Beispiel-Output

```
Input: "Baden"
Output: "Baden ist eine schöne Stadt an der Donau. Die Altstadt mit ihren..."
```

---

## Checkpoint-System Details

### Automatische Wiederaufnahme

```python
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1 if checkpoint['batch_idx'] == 0 else checkpoint['epoch']
    old_batch_idx = checkpoint.get('batch_idx', 0)
    last_total_loss = checkpoint.get('total_loss', 0)
```

**Szenarien:**

1. **Mid-Epoch Unterbruch** (batch_idx > 0):
   - Startet in gleicher Epoche weiter
   - Überspringt bereits trainierte Batches
   - avg_loss startet bei gespeichertem Wert

2. **End-of-Epoch Checkpoint** (batch_idx == 0):
   - Startet nächste Epoche
   - avg_loss bei 0 (neue Epoche)

---

## Konfiguration

Hyperparameter in `train_model()`:

```python
model = SmallLLM(
    vocab_size=tokenizer.vocab_size,
    d_model=128,           # Erhöhen für bessere Qualität (64, 128, 256, ...)
    num_heads=4,           # Muss d_model teilen (4, 8, ...)
    num_layers=6,          # Mehr Layer = größeres Modell (3, 6, 12, ...)
    d_ff=512,              # Typisch 2-4x d_model
    max_seq_len=128        # Kontext-Fenster für Training
)

dataset = TextDataset(text, tokenizer, seq_len=128)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)  # Batch Size
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Learning Rate
```

---

## Tokenizer Vergleich

| Feature | SimpleTokenizer | WordTokenizer | BPETokenizer |
|---------|-----------------|---------------|--------------|
| **Vokabulgröße** | ~100-200 | ~500-2000 | ~10000 |
| **Tokens pro Text** | Sehr hoch | Mittel | Niedrig |
| **Training Zeit** | <1s | <1s | Minuten |
| **OOV Handling** | Keine | Limited | Perfekt |
| **Empfehlung** | Einfache Tests | Baseline | Production |

---

## Speicherte Dateien

- `small_llm_checkpoint.pt` - Letzter Checkpoint (Model, Optimizer, Metadaten)
- `tokenizer.pkl` - Trainierter Tokenizer (SimpleTokenizer/WordTokenizer/BPETokenizer)
- `small_llm_model.pt` - Finales trainiertes Modell (nach Training)
