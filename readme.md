
# Multi-Scale Feature Extraction Network

## Layers

| Layer                                       | Kernel Size | Resolution Behavior      | Purpose                                                                 |
| ------------------------------------------- | ----------- | ------------------------ | ----------------------------------------------------------------------- |
| **Layer 1 – Local Detail Extraction**       | **3 × 3**   | Stride 1, padding = same | Detects fine edges, lines, and small textures at full input resolution. |
| **Layer 2 – Structural Aggregation**        | **5 × 5**   | Stride 1                 | Captures corners and medium-scale structures; slightly smooths details. |
| **Layer 3 – Contextual Abstraction**        | **7 × 7**   | Stride 2                 | Expands receptive field to connect regions and shapes.                  |
| **Layer 4 – Global Integration** | **9 × 9**   | Stride 2                 | Extracts coarse, global patterns and overall image context.             |
