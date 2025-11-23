### Vision Transformer (ViT) Architecture Overview

The notebook implements a Vision Transformer (ViT) model for image classification, specifically designed for the MNIST dataset. The architecture follows the standard ViT approach, breaking down the input image into patches, embedding them, processing them through a series of Transformer encoder blocks, and finally classifying the output using a Multi-Layer Perceptron (MLP) head.

### Key Architectural Parameters:

- Input Image Size: img_size = 28x28 pixels
- Number of Channels: num_channels = 1 (grayscale)
- Patch Size: patch_size = 7x7 pixels
- Number of Patches: number_patches = 16 ((28/7)^2)
- Token Dimension (Embedding Size): token_dim = 32
- Number of Attention Heads: num_heads = 4
- Number of Transformer Encoder Blocks: transformer_bloks = 4
- MLP Hidden Dimension in Transformer Blocks: mlp_hidden_dim = 64
- Number of Output Classes: num_classes = 10 (for MNIST digits 0-9)

### Visual Diagram (Conceptual Flow):

```
Input Image (28x28x1)
      |
      V
Patch Embedding (Conv2d 7x7, stride 7, to 32 dims)
      |
      V
Sequence of 16 Patch Embeddings (each 32-dim)
      |
      V
Prepend Class Token (1x32) + Add Positional Embeddings (17x32)
      |
      V
Input to Transformer Encoder (17x32)
      |
      V
[ Transformer Encoder Block 1 (MHA + MLP) ] -- 4 Blocks Stacked --> [ Transformer Encoder Block 4 ]
      | (Output for Class Token)
      V
MLP Classification Head (LayerNorm + Linear to 10 classes)
      |
      V
Output Logits (10-dim) for Classification
```
