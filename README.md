# Small LLM - Kleines Language Model

Dieses Projekt implementiert ein funktionsfähiges **Mini-Sprachmodell** basierend auf der Transformer-Architektur. Es ist speziell dafür konzipiert, die Grundlagen von modernen Language Models wie ChatGPT zu verstehen.

---

## 📋 Inhaltsverzeichnis

1. [Imports](#imports)
2. [SimpleTokenizer](#simpletokenizer)
3. [Attention Mechanism](#attention-mechanism)
4. [TransformerBlock](#transformerblock)
5. [SmallLLM Modell](#smallllm-modell)
6. [TextDataset](#textdataset)
7. [Training Loop](#training-loop)
8. [Text Generierung](#text-generierung)

---

## Imports

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
```

**Warum diese Imports?**

| Import | Zweck |
|--------|-------|
| `torch` | Deep Learning Framework - Grundlage für Tensoren und Berechnungen |
| `torch.nn` | Neural Network Module (Layer, Aktivierungen, Verlustfunktionen) |
| `torch.optim` | Optimierer (Adam) zum Trainieren des Modells |
| `DataLoader, Dataset` | Tools zum Laden von Trainingsdaten in Batches |
| `numpy` | Numerische Berechnungen (z.B. für Skalierungsfaktoren) |

---

## SimpleTokenizer

```python
class SimpleTokenizer:
    def __init__(self, text):
        self.chars = sorted(list(set(text)))
        self.char_to_idx = {ch: idx for idx, ch in enumerate(self.chars)}
        self.idx_to_char = {idx: ch for ch, idx in self.char_to_idx.items()}
        self.vocab_size = len(self.chars)
```

### Warum brauchen wir einen Tokenizer?

**Problem:** Neuronale Netze verstehen nur Zahlen, keine Buchstaben.

**Lösung:** Der Tokenizer wandelt Text in Zahlen um (und zurück).

### Funktionsweise:

| Methode | Aufgabe | Beispiel |
|---------|---------|---------|
| `__init__` | Erstellt Wörterbuch aller einzigartigen Zeichen | Text: "Hello" → chars: [' ', 'H', 'e', 'l', 'o'] |
| `encode()` | Konvertiert Text → Zahlen | "Hi" → [1, 2] |
| `decode()` | Konvertiert Zahlen → Text | [1, 2] → "Hi" |

### Praktisches Beispiel:

```
Text: "ab"
chars = ['a', 'b']
char_to_idx = {'a': 0, 'b': 1}
idx_to_char = {0: 'a', 1: 'b'}

encode("ab") → [0, 1]
decode([0, 1]) → "ab"
```

---

## Attention Mechanism

### 🧠 Das Herzstück von Transformers

```python
class Attention(nn.Module):
    def __init__(self, d_model, num_heads):
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
```

### Warum Attention?

**Problem:** Das Modell braucht einen Mechanismus, um zu entscheiden, welche Wörter wichtig sind.

**Beispiel:** Im Satz "Der Kater trinkt Milch" ist das Wort "Kater" für "trinkt" wichtiger als der Punkt am Ende.

### Wie funktioniert Attention?

1. **Query (Q):** "Was suche ich?"
2. **Key (K):** "Was kann ich finden?"
3. **Value (V):** "Was ist der Inhalt?"

```
Satz: "Der Kater trinkt"

Für "trinkt":
- Query: Was brauche ich zum Verstehen?
- Key: Welche anderen Wörter sind relevant?
- Value: Was ist der Gehalt dieser Wörter?

Resultat: "trinkt" konzentriert sich stark auf "Kater"
```

### Mathematik dahinter:

```
Attention(Q, K, V) = softmax(Q·K^T / √d) · V
```

| Komponente | Zweck |
|------------|-------|
| `Q·K^T` | Ähnlichkeitsscores zwischen Wörtern berechnen |
| `/ √d` | Skalierung für Stabilität |
| `softmax()` | Conversion zu Wahrscheinlichkeiten (alle = 100%) |
| `· V` | Gewichtete Kombination der Werte |

### Multi-Head Attention:

```python
# Split in multiple heads
Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
```

**Warum mehrere Heads?**
- Kopf 1 lernt: "Verben konzentrieren sich auf Substantive"
- Kopf 2 lernt: "Adjektive modifizieren Substantive"
- Kopf 3 lernt: "Pronomen beziehen sich auf frühere Substantive"

→ Verschiedene Perspektiven gleichzeitig!

---

## TransformerBlock

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        self.attention = Attention(d_model, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
```

### Was ist ein Transformer Block?

Ein Block besteht aus **2 Hauptkomponenten**:

```
┌─────────────────────────────────┐
│   Eingabe (Text-Embeddings)     │
└────────────┬────────────────────┘
             │
        ┌────▼──────────────────────────┐
        │   Self-Attention              │
        │   (Was sind relevante Wörter) │
        └────┬──────────────────────────┘
             │
    ┌────────▼────────────┐
    │  + Residual Add     │  ← Original + Attention Output
    └────────┬────────────┘
             │
        ┌────▼──────────────────────────┐
        │  Layer Normalization          │  ← Stabilisiert Training
        └────┬──────────────────────────┘
             │
        ┌────▼──────────────────────────┐
        │   Feed-Forward Network        │
        │   (Tiefere Verarbeitung)      │
        └────┬──────────────────────────┘
             │
    ┌────────▼────────────┐
    │  + Residual Add     │
    └────────┬────────────┘
             │
        ┌────▼──────────────────────────┐
        │  Layer Normalization          │
        └────┬──────────────────────────┘
             │
             ▼
        Ausgabe
```

### Komponenten im Detail:

| Teil | Funktion | Warum? |
|------|----------|-------|
| **Self-Attention** | Findet Beziehungen zwischen Wörtern | Das Modell braucht Kontext |
| **Residual Connection** (`x + output`) | Ursprüngliche Information bleibt erhalten | Verhindert Informationsverlust beim Training |
| **Layer Norm** | Normalisiert Aktivierungen | Stabilisiert Training, schneller Lernen |
| **Feed-Forward** | 2 lineare Layer mit ReLU | Fügt Nichtlinearität hinzu, ermöglicht komplexere Patterns |

### Residual Connections erklärt:

```
Ohne Residual:
Eingabe → Layer 1 → Layer 2 → Layer 3 → ??? 
(Ursprüngliche Info ist verloren gegangen)

Mit Residual:
Eingabe → Layer 1 → + Eingabe → Layer 2 → + Layer1-Output → Layer 3
```

---

## SmallLLM Modell

```python
class SmallLLM(nn.Module):
    def __init__(self, vocab_size, d_model=128, num_heads=4, 
                 num_layers=3, d_ff=512, max_seq_len=100):
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(d_model, vocab_size)
```

### Hyperparameter:

| Parameter | Standard | Bedeutung |
|-----------|----------|-----------|
| `vocab_size` | Abhängig | Wie viele unterschiedliche Zeichen/Wörter |
| `d_model` | 128 | **Dimensionalität** - wie groß ist jeder "Gedanke"? |
| `num_heads` | 4 | Wie viele parallele Aufmerksamkeitsperspektiven? |
| `num_layers` | 3 | Wie viele Transformer Blöcke hintereinander? |
| `d_ff` | 512 | Größe des Feed-Forward Netzwerks |
| `max_seq_len` | 100 | Max. Textlänge die das Modell verarbeiten kann |

### Embedding Layer erklärt:

```python
self.token_embedding = nn.Embedding(vocab_size, d_model)
```

**Was macht ein Embedding?**

Ein Embedding wandelt einen Token (Zahl) in einen dichten Vektor um:

```
Token: 5 (Character 'e')
       ↓
Embedding: [0.2, -0.1, 0.8, 0.5, -0.3, ...]  ← 128 Dimensionen
           
Diese Zahlen lernen während des Trainings!
Ähnliche Zeichen bekommen ähnliche Embeddings.
```

### Positional Encoding:

```python
positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
x = self.token_embedding(x) + self.position_embedding(positions)
```

**Warum ist das wichtig?**

```
Satz: "Der Kater trinkt Milch"

Token-Embeddings: 
- Position 0: [... Embedding von "Der"]
- Position 1: [... Embedding von "Kater"] 
- Position 2: [... Embedding von "trinkt"]
- Position 3: [... Embedding von "Milch"]

Positionalen Embeddings werden addiert!

Finales Embedding = Token-Embedding + Position-Embedding

→ Das Modell weiß nicht nur WAS, sondern auch WO!
```

### Forward Pass:

```python
def forward(self, x, mask=None):
    # 1. Embeddings + Position
    x = self.token_embedding(x) + self.position_embedding(positions)
    
    # 2. Durch alle Transformer Blöcke
    for block in self.transformer_blocks:
        x = block(x, mask)
    
    # 3. Klassifizierung: Welcher Token kommt als nächstes?
    logits = self.fc_out(x)  # Von 128 Dimensionen → vocab_size
    return logits
```

---

## TextDataset

```python
class TextDataset(Dataset):
    def __init__(self, text, tokenizer, seq_len=50):
        self.tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    
    def __getitem__(self, idx):
        x = self.tokens[idx:idx + self.seq_len]
        y = self.tokens[idx + 1:idx + self.seq_len + 1]
        return x, y
```

### Wie funktioniert das Training?

**Idee:** Das Modell lernt, den nächsten Character vorherzusagen.

**Praktisches Beispiel:**

```
Text: "Hallo"
Tokens: [7, 0, 11, 11, 14]  (nach Tokenisierung)

Trainingspaar 1:
x = [7, 0]          (Input: "Ha")
y = [0, 11]         (Target: "al" - nächste Zeichen)

Trainingspaar 2:
x = [7, 0, 11]      (Input: "Hal")
y = [0, 11, 11]     (Target: "all")

Trainingspaar 3:
x = [7, 0, 11, 11]  (Input: "Hall")
y = [0, 11, 11, 14] (Target: "allo")
```

### Training-Prozess:

```
1. Gib [7, 0] ins Modell → Modell sagt: "wahrscheinlich 0" ✓
2. Gib [7, 0, 11] ins Modell → Modell sagt: "wahrscheinlich 11" ✓
3. Etc...

Wenn Modell richtig liegt: Loss klein
Wenn Modell falsch liegt: Loss groß → Gewichte anpassen
```

---

## Training Loop

```python
# Training setup
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# Training loop
for epoch in range(num_epochs):
    for x_batch, y_batch in dataloader:
        logits = model(x_batch)
        loss = loss_fn(logits.reshape(-1, vocab_size), y_batch.reshape(-1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Was passiert hier?

```
EPOCH 1:
┌──────────────────────────────────┐
│ For each Batch:                  │
│  1. Forward Pass                 │
│     x_batch → Modell → logits   │
│                                  │
│  2. Loss berechnen               │
│     Wie falsch ist die Vorhersage?│
│                                  │
│  3. Backward Pass                │
│     Berechne Gradienten          │
│     (Wie sollten sich Gewichte   │
│      ändern?)                    │
│                                  │
│  4. Optimizer Step               │
│     Passe Gewichte an            │
│                                  │
└──────────────────────────────────┘
         ↓
    Loss nimmt ab
         ↓
    Modell wird besser!
```

### Optimizer (Adam):

```python
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

**Was ist Adam?**

- **Adaptive Learning Rate Optimizer**
- Passt die Lernrate für jeden Parameter einzeln an
- Besser als simpler Gradient Descent
- Standard in modernen Deep Learning

### CrossEntropyLoss:

```python
loss_fn = nn.CrossEntropyLoss()
```

**Warum diese Loss-Funktion?**

```
Modell Output für Token: [0.1, 2.5, 0.3, 1.2, ...]
                         (Logits - unnormalisiert)

CrossEntropyLoss macht:
1. Softmax: [0.05, 0.6, 0.1, 0.2, ...]  (Wahrscheinlichkeiten)
2. Vergleicht mit Target: [0, 1, 0, 0, ...]  (True Label)
3. Berechnet -log(0.6) = 0.51 (Loss)

Wenn Modell richtig: Loss ≈ 0 ✓
Wenn Modell falsch: Loss groß ✗
```

---

## Text Generierung

```python
model.eval()
seed_text = "Das ist"
seed_tokens = torch.tensor([tokenizer.encode(seed_text)], device=device)

generated = seed_text
for _ in range(50):
    with torch.no_grad():
        logits = model(seed_tokens)
        next_token_logits = logits[0, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1)
        generated += tokenizer.decode([next_token.item()])
```

### Schritt-für-Schritt:

```
Startext: "Das ist"

Step 1:
┌─────────────────────────────────┐
│ Modell Input: [D, a, s, , i, s]│
│        ↓                         │
│ Modell Output für letztes Token: │
│ [0.1, 0.8, 0.05, ...] (Logits) │
│        ↓                         │
│ argmax → Token 1 (Charakter 'a')│
│ Ausgabe: "Das ista"             │
└─────────────────────────────────┘

Step 2:
┌─────────────────────────────────┐
│ Modell Input: [a, s, , i, s, a]│
│        ↓                         │
│ Vorhersage: Token 4 (Charakter 'l')?│
│ Ausgabe: "Das istal"            │
└─────────────────────────────────┘

... (50 mal)
```

### `model.eval()` und `torch.no_grad()`:

| Code | Funktion |
|------|----------|
| `model.eval()` | Schaltet Modell in Evaluations-Modus (Dropout deaktiviert) |
| `torch.no_grad()` | Keine Gradienten berechnen (spart Speicher & Zeit) |
| `argmax()` | Wählt Token mit höchster Wahrscheinlichkeit |

---

## 🚀 Wie man das Modell startet

```bash
python small_llm.py
```

### Output:

```
Using device: cuda (oder cpu)
Vokabulgröße: 23
Epoch 10/500, Loss: 2.3456
Epoch 20/500, Loss: 1.8234
...
Epoch 500/500, Loss: 0.1234

=== Text Generierung ===
Das ist ein kleines Sprachmodell das lernt Text zu generieren
```

---

## 📊 Hyperparameter zum Experimentieren

```python
# Im Code: train_model()

model = SmallLLM(
    vocab_size=tokenizer.vocab_size,
    d_model=64,          # ← Erhöhen für komplexere Muster
    num_heads=2,         # ← Muss d_model teilen
    num_layers=2,        # ← Mehr Layer = tieferes Modell
    d_ff=256,            # ← Feed-Forward Größe
    max_seq_len=100
)

# Training
num_epochs = 500        # ← Mehr Epochen = besseres Lernen
# seq_len=20 im Dataset  # ← Längere Sequenzen = mehr Kontext
```

---

## 🎓 Zusammenfassung

| Komponente | Aufgabe |
|------------|---------|
| **SimpleTokenizer** | Text ↔ Zahlen |
| **Attention** | Findet wichtige Wörter |
| **TransformerBlock** | Verarbeitet Text in mehreren Schichten |
| **SmallLLM** | Hauptmodell mit Embeddings + Transformer |
| **TextDataset** | Bereitet Trainingsdaten vor |
| **Training Loop** | Lehrt das Modell, den nächsten Character vorherzusagen |
| **Text Generation** | Nutzt trainiertes Modell zur Vorhersage |

---

## 💡 Weitere Konzepte zum Lernen

- **Batch Normalization vs Layer Normalization**
- **Positional Encoding** (Sinusoidale vs gelernte)
- **Masking** (Causal Attention für Language Models)
- **Temperature Sampling** (vs Greedy Decoding)
- **Beam Search** (bessere Text-Generierung)

---

Viel Spaß beim Lernen! 🚀
