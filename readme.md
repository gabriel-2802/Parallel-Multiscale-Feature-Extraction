
# Multi-Scale Feature Extraction Network

## Layers

| Layer                                       | Kernel Size | Resolution Behavior      | Purpose                                                                 |
| ------------------------------------------- | ----------- | ------------------------ | ----------------------------------------------------------------------- |
| **Layer 1 – Local Detail Extraction**       | **3 × 3**   | Stride 1, padding = same | Detects fine edges, lines, and small textures at full input resolution. |
| **Layer 2 – Structural Aggregation**        | **5 × 5**   | Stride 1                 | Captures corners and medium-scale structures; slightly smooths details. |
| **Layer 3 – Contextual Abstraction**        | **7 × 7**   | Stride 2                 | Expands receptive field to connect regions and shapes.                  |

## Layer Details

| Layer | Kernel       | Stride | Padding | Effect                              |
| ----- | ------------ | ------ | ------- | ----------------------------------- |
| 1     | 3×3 Gaussian | 1      | 1       | Edge detection & fine texture       |
| 2     | 5×5 Gaussian | 1      | 2       | Mid-scale smoothing & shape merging |
| 3     | 7×7 Gaussian | 2      | 3       | Structural abstraction              |

| **Layer**         | **Kernel Size** | **Matrix (Normalized)**                                                                                                                                                                                                    | **Notes**                                                               |
| ----------------- | --------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| **Layer 1 – 3×3** | 3 × 3           | [[1, 2, 1], [2, 4, 2], [1, 2, 1]] ÷ 16                                                                                                                                                                                     | Light smoothing, edge-preserving. Excellent for fine-detail extraction. |
| **Layer 2 – 5×5** | 5 × 5           | [[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, 36, 24, 6], [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]] ÷ 256                                                                                                                       | Broader Gaussian; merges nearby structures, reduces noise.              |
| **Layer 3 – 7×7** | 7 × 7           | [[1, 6, 15, 20, 15, 6, 1], [6, 36, 90, 120, 90, 36, 6], [15, 90, 225, 300, 225, 90, 15], [20, 120, 300, 400, 300, 120, 20], [15, 90, 225, 300, 225, 90, 15], [6, 36, 90, 120, 90, 36, 6], [1, 6, 15, 20, 15, 6, 1]] ÷ 4096 | Large receptive field; captures object shapes and context.              |
