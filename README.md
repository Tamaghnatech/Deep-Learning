# Deep Learning

This repository serves as a structured path to master Deep Learning — from perceptrons to modern LLMs — using theory, projects, and visualizations.

---

## 🧠 Topics Covered

### 🔥 Introduction to Deep Learning
- What is Deep Learning?
- Traditional ML vs Neural Networks
- When to Use DL (Data vs Compute tradeoff)
- Layers, Neurons, Tensors
- Forward Pass and Loss Computation
- Activation Functions: ReLU, Sigmoid, Tanh

### 🔁 Core Neural Network Concepts
- Perceptron → MLP (Multi-Layer Perceptron)
- Loss Functions: MSE, Cross-Entropy
- Backpropagation and Chain Rule
- Optimizers: SGD, Adam, RMSprop
- Overfitting & Underfitting, Bias–Variance
- BatchNorm, LayerNorm, Dropout, Regularization

### 📊 Visualizing Learning
- Loss Landscapes
- Gradient Descent Path Visuals
- Training Curves and Validation Monitoring

---

### 🧱 Convolutional Neural Networks (CNNs)
- Convolutions, Filters, Padding, Stride
- Max/Avg Pooling, Feature Maps
- ResNet, DenseNet, EfficientNet
- Depthwise & Dilated Convs
- Transfer Learning
- Attention-Augmented CNNs
- Object Detection Basics (YOLO, SSD)

#### 📌 Projects
- CIFAR-10 Classifier (from scratch and with PyTorch)
- Fine-tune a pre-trained CNN on custom data
- Feature Visualization & Filter Heatmaps

---

### 🔄 Sequence Models
- RNNs: Vanishing Gradient Problem
- LSTM, GRU, Bi-directional RNNs
- Seq2Seq + Attention Mechanisms
- Time Series Forecasting (LSTM-based)
- Temporal Convolutional Networks

#### 📌 Projects
- Char-Level Text Generation (LSTM)
- Weather Prediction (Time Series with LSTM)
- Neural Machine Translation (Seq2Seq + Attention)

---

### 🔭 Transformers & Attention
- Self-Attention, Positional Encoding
- Transformer Blocks: Encoder/Decoder
- BERT, GPT, T5, BART Architectures
- HuggingFace Transformers
- Efficient Transformers: FlashAttention, ALiBi, RoPE
- Fine-tuning with LoRA, QLoRA

#### 📌 Projects
- Sentiment Classifier using BERT
- Mini GPT Text Generator
- Q&A with T5 or BART

---

### 🎨 Generative Models
- Autoencoders & Variational AEs (VAE)
- Generative Adversarial Networks (GAN, DCGAN, StyleGAN)
- Diffusion Models
- Neural Radiance Fields (NeRF)

#### 📌 Projects
- Digit Generation with VAE
- GAN to generate human-like faces (CelebA)
- Denoising Autoencoder (Fashion-MNIST)

---

### 🛠️ Training Engineering & Efficiency
- Optimizers: AdamW, SAM
- Learning Rate Schedulers: One-Cycle, Cosine
- Data Augmentation: MixUp, CutMix
- Early Stopping & Checkpointing
- DDP, ZeRO, DeepSpeed
- Mixed Precision (FP16, BF16)
- Model Quantization & Pruning
- ONNX, TensorRT Deployment

---

## 📁 Folder Structure

- `notes/`  
  📚 Concept explanations, formulas, and intuition-first theory notes.

- `projects/`  
  📓 Google Colab notebooks for building, training, and visualizing models.

- `diagrams/`  
  🧠 Visual illustrations of networks, attention, convolutions, backpropagation, and more.

---

## 🧪 Bonus Project Ideas
- Create an MLP from scratch using NumPy
- Train ResNet on CIFAR-100 and visualize confusion matrix
- Fine-tune BERT and log training with Weights & Biases
- Build a GAN that generates artistic images from sketch
- Implement Grad-CAM for model explainability

---

> “Neural networks are not just functions. They are differentiable universes of understanding.”  
Welcome to the cosmos, Lord Nag. 🌌🔥
