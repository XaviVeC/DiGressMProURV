# GINE Regressor Architecture

```mermaid
flowchart TD
    NF["Node features<br/>(atom: symbol, hybridization,<br/>degree, Hs, charge,<br/>aromaticity, ring, mass)"]
    EF["Edge features<br/>(bond: type, stereo,<br/>conjugation, ring)"]

    NE["Node Encoder<br/>Linear → hidden_dim"]
    EI["Edge Input<br/>Linear → hidden_dim"]

    NF --> NE
    EF --> EI

    subgraph LAYER ["GINELayer  ×  num_layers"]
        direction TB
        ENC["Edge Encoder<br/>Linear → ReLU → Linear"]
        MSG["Messages<br/>edge_enc(edge_attr) + x[src]  → ReLU"]
        AGG["Aggregation<br/>Σ messages at each destination node"]
        RES["Residual combination<br/>(1 + ε) · x  +  aggregated"]
        UPD["Update MLP<br/>Linear → ReLU → Dropout → Linear"]
        LN["LayerNorm + Dropout<br/>x  =  x  +  dropout(update_mlp(...))"]

        ENC --> MSG
        MSG --> AGG
        AGG --> RES
        RES --> UPD
        UPD --> LN
    end

    NE --> LAYER
    EI --> LAYER

    subgraph READOUT ["Graph Readout"]
        direction TB
        MEAN["Global Mean Pool<br/>mean of node embeddings per graph"]
        MAX["Global Max Pool<br/>max of node embeddings per graph"]
        CAT["Concatenate<br/>[mean || max]  →  2 × hidden_dim"]

        MEAN --> CAT
        MAX --> CAT
    end

    LN --> READOUT

    FF["FeedForward Readout<br/>Linear → ReLU → Dropout → Linear<br/>2×hidden_dim → hidden_dim"]
    HEAD["Prediction Head<br/>ReLU → Dropout → Linear<br/>hidden_dim → 1"]
    OUT["pIC50 prediction<br/>(scalar)"]

    CAT --> FF
    FF --> HEAD
    HEAD --> OUT
```

## Component Summary

| Component | Structure | Purpose |
|---|---|---|
| Node Encoder | Linear | Projects raw atom features to hidden_dim |
| Edge Input | Linear | Projects raw bond features to hidden_dim |
| Edge Encoder (per layer) | Linear → ReLU → Linear | Transforms bond embeddings into message context |
| Update MLP (per layer) | Linear → ReLU → Dropout → Linear | Refines node embeddings after aggregation |
| GINELayer residual | (1 + ε) · x + aggregated | Learnable skip connection; ε is a per-layer parameter |
| Layer normalization | LayerNorm + Dropout | Stabilizes activations after each message-passing step |
| Global Mean Pool | Mean over nodes | Captures average molecular properties |
| Global Max Pool | Max over nodes | Captures the most prominent local features |
| FeedForward Readout | Linear → ReLU → Dropout → Linear | Compresses concatenated pooled representation |
| Prediction Head | ReLU → Dropout → Linear | Maps to a single scalar pIC50 value |
