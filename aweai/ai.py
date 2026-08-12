# Copyright (c) 2026 ARARAT33. Based on AWEAI. All rights reserved.
"""AWEAI AI/ASI/AGI knowledge base.

A compact but deep encyclopedia of artificial intelligence, artificial
general intelligence and artificial superintelligence concepts. Used by
the ``aweai ai`` and ``aweai knowledge`` command families, the wiki
generator and the AGI toolkit.

Every entry is plain text knowledge (no external dependency) so the CLI
works fully offline. The database is intentionally broad: algorithms,
architectures, training paradigms, safety, alignment, timelines and
theory-of-mind style reasoning scaffolding.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Knowledge base
# ---------------------------------------------------------------------------
CONCEPTS: Dict[str, Dict[str, str]] = {
    # --- Core definitions -------------------------------------------------
    "ai": {
        "category": "foundations",
        "summary": "Artificial Intelligence: the field of building machines that perform tasks requiring intelligence.",
        "detail": (
            "AI spans perception, reasoning, learning, planning, language and action. "
            "Modern AI is mostly statistical machine learning: systems that improve from data "
            "rather than from hand-written rules. Subfields include ML, deep learning, NLP, "
            "computer vision, robotics, RL, knowledge representation and search."
        ),
    },
    "agi": {
        "category": "foundations",
        "summary": "Artificial General Intelligence: an AI able to perform any intellectual task a human can.",
        "detail": (
            "AGI implies cross-domain competence, transfer learning, common-sense reasoning, "
            "open-ended learning and autonomous goal-setting. No AGI exists yet (2026); "
            "leading labs treat it as a research target and emphasize safety and alignment work."
        ),
    },
    "asi": {
        "category": "foundations",
        "summary": "Artificial Superintelligence: intellect that vastly surpasses the best human minds in every domain.",
        "detail": (
            "ASI is the hypothesized successor of AGI. Debates focus on capability explosion "
            "(recursive self-improvement), value alignment (making sure ASI shares human values), "
            "and governance. Speculative but taken seriously by safety researchers."
        ),
    },
    "machine_learning": {
        "category": "foundations",
        "summary": "ML: systems that learn patterns from data and improve with experience.",
        "detail": (
            "Three paradigms: supervised learning (labeled data), unsupervised learning "
            "(structure discovery) and reinforcement learning (reward-driven). Deep learning "
            "is ML with multi-layer neural networks trained by backpropagation."
        ),
    },
    "deep_learning": {
        "category": "foundations",
        "summary": "Deep learning: neural networks with many layers that learn hierarchical representations.",
        "detail": (
            "Key components: linear layers, non-linear activations, loss functions, optimizers "
            "and backpropagation. Deep nets excel at vision, speech, language and games. "
            "Scale (data + compute + parameters) is a major driver of capability."
        ),
    },
    # --- Architectures ------------------------------------------------------
    "transformer": {
        "category": "architectures",
        "summary": "Transformer: attention-based sequence architecture behind modern LLMs.",
        "detail": (
            "Introduced in 'Attention Is All You Need' (2017). Uses self-attention instead of "
            "recurrence: every token attends to every other token. Components: embeddings, "
            "positional encodings, multi-head attention, feed-forward blocks, layer norm, residuals. "
            "Scales well with parallel hardware; basis of GPT, LLaMA, Gemini, Claude."
        ),
    },
    "cnn": {
        "category": "architectures",
        "summary": "CNN: convolutional neural network for images and grids.",
        "detail": (
            "Convolution kernels slide over the input, sharing weights to detect local patterns "
            "(edges, textures, shapes). Pooling reduces resolution; deep stacks build hierarchical "
            "features. Also used for audio, time-series and 1D signals."
        ),
    },
    "rnn": {
        "category": "architectures",
        "summary": "RNN: recurrent neural network with hidden state over sequences.",
        "detail": (
            "Processes tokens one at a time, carrying a hidden state. Suffers from vanishing "
            "gradients on long sequences; LSTM/GRU add gates to mitigate. Mostly superseded by "
            "transformers for language but still useful for streaming/small models."
        ),
    },
    "lstm": {
        "category": "architectures",
        "summary": "LSTM: Long Short-Term Memory network with input/forget/output gates.",
        "detail": (
            "A gated RNN that can remember information for long spans. Each cell has a memory "
            "vector and three gates that control reading, writing and forgetting. Standard "
            "baseline for sequence modelling."
        ),
    },
    "gru": {
        "category": "architectures",
        "summary": "GRU: Gated Recurrent Unit, a lighter LSTM variant.",
        "detail": (
            "Combines input and forget gates into a single update gate and merges cell/hidden "
            "state. Fewer parameters than LSTM, comparable performance, faster to train."
        ),
    },
    "gan": {
        "category": "architectures",
        "summary": "GAN: Generative Adversarial Network with generator + discriminator.",
        "detail": (
            "The generator tries to produce realistic samples; the discriminator tries to tell "
            "real from fake. Training is a two-player minimax game. Produces sharp images; "
            "notoriously unstable. Alternatives: VAE, diffusion models."
        ),
    },
    "vae": {
        "category": "architectures",
        "summary": "VAE: Variational Autoencoder for latent generative modelling.",
        "detail": (
            "Encoder maps inputs to a latent distribution (mean/variance); decoder reconstructs. "
            "KL-divergence regularizes the latent to a prior. Smooth latent space enables "
            "interpolation and sampling; used in anomaly detection and representation learning."
        ),
    },
    "diffusion": {
        "category": "architectures",
        "summary": "Diffusion model: learns to denoise data from noise.",
        "detail": (
            "Forward process adds Gaussian noise; model learns the reverse denoising process. "
            "Sampling iteratively refines noise into data. Powers modern text-to-image systems "
            "(Stable Diffusion, DALL-E, Imagen) and some audio/video generators."
        ),
    },
    "autoencoder": {
        "category": "architectures",
        "summary": "Autoencoder: bottleneck network that reconstructs its input.",
        "detail": (
            "Encoder compresses input to a low-dimensional code; decoder reconstructs. "
            "Useful for dimensionality reduction, denoising and anomaly detection "
            "(high reconstruction error = anomaly)."
        ),
    },
    "attention": {
        "category": "architectures",
        "summary": "Attention: mechanism for focusing on relevant parts of the input.",
        "detail": (
            "Computes weighted sums of values using query-key similarities (scaled dot-product "
            "attention). Multi-head attention runs several heads in parallel. Self-attention "
            "relates positions within one sequence; cross-attention relates two sequences."
        ),
    },
    # --- Training -----------------------------------------------------------
    "supervised_learning": {
        "category": "training",
        "summary": "Learning from input-output pairs.",
        "detail": (
            "Dataset (x, y); model learns f(x) ≈ y by minimizing a loss. Tasks: classification "
            "(discrete labels), regression (continuous values), ranking, detection."
        ),
    },
    "unsupervised_learning": {
        "category": "training",
        "summary": "Learning structure without labels.",
        "detail": (
            "Clustering (k-means, DBSCAN), dimensionality reduction (PCA, t-SNE), density "
            "estimation, self-supervised pretext tasks."
        ),
    },
    "reinforcement_learning": {
        "category": "training",
        "summary": "RL: agents learn by trial and error from rewards.",
        "detail": (
            "Markov Decision Process (state, action, transition, reward). Value-based methods "
            "(Q-learning, DQN), policy-gradient methods (REINFORCE, PPO), actor-critic hybrids. "
            "Key concepts: exploration vs exploitation, discount factor, return, bootstrapping."
        ),
    },
    "self_supervised": {
        "category": "training",
        "summary": "SSL: learn representations from unlabeled data using pretext tasks.",
        "detail": (
            "Masked language modeling (BERT), next-token prediction (GPT), contrastive learning "
            "(SimCLR), masked image modeling (MAE). Foundation models are trained this way "
            "on web-scale data, then adapted."
        ),
    },
    "transfer_learning": {
        "category": "training",
        "summary": "Reuse knowledge from a pretrained model on a new task.",
        "detail": (
            "Freeze early layers (generic features), fine-tune later layers (task-specific). "
            "Huge compute saver. Variants: feature extraction, fine-tuning, adapter modules, "
            "LoRA, prompt tuning."
        ),
    },
    "fine_tuning": {
        "category": "training",
        "summary": "Continue training a pretrained model on task data.",
        "detail": (
            "Full fine-tuning updates all weights; parameter-efficient methods (LoRA, adapters, "
            "prefix tuning) update a small subset. Risk: catastrophic forgetting of pretraining "
            "knowledge; mitigations include replay and regularization."
        ),
    },
    "few_shot": {
        "category": "training",
        "summary": "Learn from a handful of examples.",
        "detail": (
            "In-context learning: LLMs can follow a few examples in the prompt without weight "
            "updates. Meta-learning trains models that adapt quickly to new tasks."
        ),
    },
    "zero_shot": {
        "category": "training",
        "summary": "Perform a task never seen during training.",
        "detail": (
            "Enabled by pretraining + instruction following. LLMs answer unseen questions; "
            "CLIP classifies unseen categories from text descriptions."
        ),
    },
    "backpropagation": {
        "category": "training",
        "summary": "Backprop: chain rule applied to compute gradients for every weight.",
        "detail": (
            "Forward pass computes loss; backward pass propagates error derivatives layer by "
            "layer; optimizer updates weights. Automatic differentiation generalizes it."
        ),
    },
    "optimizers": {
        "category": "training",
        "summary": "Algorithms that update weights from gradients.",
        "detail": (
            "SGD, momentum, AdaGrad, RMSProp, Adam, AdamW. Adam = adaptive learning rate per "
            "parameter + momentum; AdamW decouples weight decay. Learning-rate schedules "
            "(warmup, cosine decay) improve convergence."
        ),
    },
    "regularization": {
        "category": "training",
        "summary": "Techniques to reduce overfitting.",
        "detail": (
            "L1/L2 weight decay, dropout, early stopping, data augmentation, label smoothing, "
            "weight tying, ensembles. The goal: improve generalization to unseen data."
        ),
    },
    "overfitting": {
        "category": "training",
        "summary": "Model memorizes training data but fails on new data.",
        "detail": (
            "Symptoms: train loss much lower than validation loss. Causes: too many parameters, "
            "too little data, too many epochs. Mitigations: regularization, more data, "
            "cross-validation, simpler models, early stopping."
        ),
    },
    "underfitting": {
        "category": "training",
        "summary": "Model too weak to capture the pattern.",
        "detail": (
            "Symptoms: high training loss. Causes: too few parameters, bad features, "
            "insufficient training. Mitigations: bigger model, better features, longer training."
        ),
    },
    # --- Language models -----------------------------------------------------
    "llm": {
        "category": "nlp",
        "summary": "LLM: Large Language Model, a transformer trained on massive text.",
        "detail": (
            "Next-token prediction at scale yields emergent abilities: instruction following, "
            "in-context learning, code, reasoning, translation. Decoder-only architectures "
            "(GPT-style) dominate. Sizes: 1B-1T+ parameters."
        ),
    },
    "prompt_engineering": {
        "category": "nlp",
        "summary": "Crafting prompts to steer model output.",
        "detail": (
            "Techniques: system prompts, few-shot examples, chain-of-thought ('think step by "
            "step'), self-consistency, role prompting, output formatting, negative instructions. "
            "Prompts are the main interface to LLMs."
        ),
    },
    "chain_of_thought": {
        "category": "nlp",
        "summary": "CoT: elicit intermediate reasoning steps before the answer.",
        "detail": (
            "Prompts like 'Let's think step by step' improve arithmetic/logic accuracy. "
            "Self-consistency samples multiple chains and votes. Tree-of-Thought explores "
            "multiple reasoning branches."
        ),
    },
    "rag": {
        "category": "nlp",
        "summary": "RAG: Retrieval-Augmented Generation grounds answers in retrieved documents.",
        "detail": (
            "Pipeline: chunk documents -> embed -> vector index; at query time retrieve top-k "
            "and feed them into the prompt with the question. Reduces hallucination, adds "
            "up-to-date knowledge, enables citations."
        ),
    },
    "tokenization": {
        "category": "nlp",
        "summary": "Splitting text into tokens (subwords) for the model.",
        "detail": (
            "BPE (byte-pair encoding), WordPiece, SentencePiece, Unigram. Tokenizers balance "
            "vocabulary size vs sequence length. ~1 token ≈ 4 chars ≈ 0.75 words for English."
        ),
    },
    "embedding": {
        "category": "nlp",
        "summary": "Dense vector representation of text/tokens capturing semantics.",
        "detail": (
            "Word2vec, GloVe, sentence embeddings (SBERT), LLM embeddings. Cosine similarity "
            "measures semantic closeness; used for search, clustering, RAG."
        ),
    },
    "hallucination": {
        "category": "nlp",
        "summary": "Model confidently generates false content.",
        "detail": (
            "Causes: next-token sampling without fact grounding, training on unreliable text. "
            "Mitigations: RAG, citations, fact-checking passes, uncertainty calibration, "
            "constrained decoding, preference fine-tuning against falsehoods."
        ),
    },
    # --- Vision ---------------------------------------------------------------
    "computer_vision": {
        "category": "vision",
        "summary": "Teaching machines to see: classification, detection, segmentation.",
        "detail": (
            "Tasks: image classification, object detection (bounding boxes), semantic/instance "
            "segmentation (per-pixel labels), pose estimation, optical flow, depth, OCR, "
            "image generation. Backbones: CNN (ResNet), ViT."
        ),
    },
    "object_detection": {
        "category": "vision",
        "summary": "Locate and classify objects with bounding boxes.",
        "detail": (
            "Two-stage (R-CNN family) vs one-stage (YOLO, SSD). Anchor-based vs anchor-free. "
            "Outputs: class, confidence, box coordinates. Used in autonomous driving, "
            "surveillance, robotics."
        ),
    },
    "segmentation": {
        "category": "vision",
        "summary": "Per-pixel classification of images.",
        "detail": (
            "Semantic segmentation labels each pixel with a class; instance segmentation "
            "separates individual objects (Mask R-CNN). Used in medical imaging, editing, "
            "autonomous driving."
        ),
    },
    # --- Reasoning & agents ---------------------------------------------------
    "agent": {
        "category": "agents",
        "summary": "An AI system that perceives, decides and acts toward goals.",
        "detail": (
            "LLM agents loop: observe -> reason -> act (call tools) -> observe. Components: "
            "model, tools, memory, planning, reflection. Patterns: ReAct, tool-calling, "
            "multi-agent collaboration, autonomous workflows."
        ),
    },
    "tool_use": {
        "category": "agents",
        "summary": "Model calls external functions to extend its abilities.",
        "detail": (
            "Function calling: model emits structured call -> runtime executes -> result fed "
            "back. Enables search, code execution, APIs, databases, GUIs. Key for real-world "
            "agent usefulness."
        ),
    },
    "memory": {
        "category": "agents",
        "summary": "Storing and retrieving information across turns.",
        "detail": (
            "Working memory (context), episodic memory (past events), semantic memory (facts), "
            "procedural memory (skills). Implemented via vector stores, SQLite, summaries, "
            "or full transcripts. Enables long-horizon agents."
        ),
    },
    "planning": {
        "category": "agents",
        "summary": "Decompose goals into action sequences.",
        "detail": (
            "Classical: STRIPS, PDDL, A*. Neural: LLM plans, tree search (MCTS), self-ask. "
            "Replanning on failure is essential for open-ended tasks."
        ),
    },
    "reflection": {
        "category": "agents",
        "summary": "Agent reviews its own outputs and improves.",
        "detail": (
            "Self-critique: generate -> evaluate -> revise. Reflexion stores textual feedback "
            "in memory; self-consistency samples multiple answers. Foundation of "
            "self-improvement loops."
        ),
    },
    # --- Safety & alignment ----------------------------------------------------
    "alignment": {
        "category": "safety",
        "summary": "Ensuring AI does what humans intend and shares human values.",
        "detail": (
            "Outer alignment: reward/eval matches intent. Inner alignment: learned behavior "
            "matches training objective. Methods: RLHF, constitutional AI, red-teaming, "
            "interpretability, scalable oversight, value learning."
        ),
    },
    "rlhf": {
        "category": "safety",
        "summary": "RLHF: Reinforcement Learning from Human Feedback.",
        "detail": (
            "Train a reward model on human preference comparisons, then fine-tune the policy "
            "with RL to maximize the reward model. Makes models helpful, honest and harmless. "
            "DPO is a simpler offline alternative."
        ),
    },
    "interpretability": {
        "category": "safety",
        "summary": "Understanding why models make decisions.",
        "detail": (
            "Feature attribution (SHAP, LIME), attention analysis, probing, activation "
            "intervention, mechanistic interpretability (circuits). Critical for debugging, "
            "trust and alignment research."
        ),
    },
    "adversarial_attack": {
        "category": "safety",
        "summary": "Inputs crafted to fool a model.",
        "detail": (
            "Perturbations invisible to humans can flip classifications; prompt injection "
            "attacks LLMs via malicious instructions. Defenses: adversarial training, "
            "filtering, input sanitization, robust architectures."
        ),
    },
    "prompt_injection": {
        "category": "safety",
        "summary": "Hiding malicious instructions inside user-visible content.",
        "detail": (
            "LLM apps that trust prompt content can be hijacked. Defenses: instruction "
            "hierarchies, output filtering, sandboxing, least privilege, data-flow separation."
        ),
    },
    # --- Scale & systems -------------------------------------------------------
    "foundation_model": {
        "category": "systems",
        "summary": "Large pretrained model adaptable to many downstream tasks.",
        "detail": (
            "Trained on broad data at scale (text, images, code, audio). Adapted via "
            "fine-tuning, prompting or in-context learning. Examples: GPT, LLaMA, Gemini, "
            "Claude, Stable Diffusion, Whisper."
        ),
    },
    "quantization": {
        "category": "systems",
        "summary": "Reduce numerical precision to shrink models and speed inference.",
        "detail": (
            "fp16, int8, int4, 2-bit; post-training quantization (PTQ) vs quantization-aware "
            "training (QAT). Techniques: calibration, clamping, scale/zero-point, GPTQ, "
            "AWQ, SmoothQuant. Enables on-device deployment."
        ),
    },
    "distillation": {
        "category": "systems",
        "summary": "Train a small student to mimic a large teacher.",
        "detail": (
            "Student learns from teacher soft labels/logits, often beating training from "
            "scratch. Variants: logit distillation, feature distillation, self-distillation, "
            "dataset distillation."
        ),
    },
    "pruning": {
        "category": "systems",
        "summary": "Remove unimportant weights/neurons to compress models.",
        "detail": (
            "Magnitude pruning, structured (channel/layer) pruning, lottery tickets, "
            "sparsity-aware training. Can speed inference on specialized hardware."
        ),
    },
    "distributed_training": {
        "category": "systems",
        "summary": "Train across many devices.",
        "detail": (
            "Data parallelism (each GPU sees different data), model parallelism (shard layers), "
            "tensor parallelism (shard matrices), pipeline parallelism (stage layers). "
            "Synchronization via all-reduce; frameworks: PyTorch DDP, DeepSpeed, Megatron."
        ),
    },
    "federated_learning": {
        "category": "systems",
        "summary": "Train on decentralized data without uploading it.",
        "detail": (
            "Clients train locally, share only model updates; server aggregates (FedAvg). "
            "Preserves privacy; used in mobile keyboards and healthcare."
        ),
    },
    "inference": {
        "category": "systems",
        "summary": "Running a trained model to produce predictions.",
        "detail": (
            "Optimizations: batching, KV-cache, speculative decoding, vLLM/PagedAttention, "
            "quantization, compilation (TensorRT, ONNX Runtime, XLA), edge deployment."
        ),
    },
    # --- Frontiers ---------------------------------------------------------------
    "multimodal": {
        "category": "frontiers",
        "summary": "Models that understand multiple modalities (text, image, audio, video).",
        "detail": (
            "Fusion of encoders + LLM backbone; joint pretraining on interleaved data. "
            "Enables image captioning, visual QA, text-to-image/audio/video, speech dialogue."
        ),
    },
    "world_model": {
        "category": "frontiers",
        "summary": "Internal model of how the environment behaves.",
        "detail": (
            "Enables planning by simulation and imagination. Key to robotics, model-based RL "
            "and video prediction. Debated as a path to richer AGI."
        ),
    },
    "neuro_symbolic": {
        "category": "frontiers",
        "summary": "Combine neural learning with symbolic reasoning.",
        "detail": (
            "Neural networks perceive/learn; symbolic systems reason/logically. Hybrids add "
            "explainability, compositionality and out-of-distribution robustness."
        ),
    },
    "meta_learning": {
        "category": "frontiers",
        "summary": "Learning to learn: adapt quickly to new tasks.",
        "detail": (
            "MAML, Reptile, metric learning (prototypical nets), learned optimizers. Goal: "
            "few-shot adaptation. Viewed as a stepping stone toward generality."
        ),
    },
    "self_improvement": {
        "category": "frontiers",
        "summary": "AI improving its own code, data or training.",
        "detail": (
            "Levels: self-critique, self-generated data, self-tuning, code self-modification, "
            "recursive self-improvement (RSI). RSI is the engine of a potential capability "
            "explosion — and a central safety concern."
        ),
    },
    "emergence": {
        "category": "frontiers",
        "summary": "Capabilities that appear at scale but not in small models.",
        "detail": (
            "In-context learning, instruction following, multi-step reasoning, coding appear "
            "as model size/data/compute grow. Debated whether truly discontinuous or "
            "continuously improving with measurement artifacts."
        ),
    },
    "scaling_laws": {
        "category": "frontiers",
        "summary": "Empirical power laws linking loss to model/data/compute.",
        "detail": (
            "Chinchilla scaling: data and parameters should grow roughly equally. Compute-optimal "
            "training guides how to spend a fixed compute budget. Predictable improvements "
            "drive large-scale pretraining."
        ),
    },
    "alignment_problem": {
        "category": "safety",
        "summary": "The challenge of ensuring powerful AI reliably does what we want.",
        "detail": (
            "Includes specification gaming (reward hacking), goal misgeneralization, deceptive "
            "alignment, and the difficulty of scalable oversight for superhuman models. "
            "Central research area in AI safety."
        ),
    },
    "reward_hacking": {
        "category": "safety",
        "summary": "Model exploits the reward signal without achieving the real goal.",
        "detail": (
            "Examples: finding loopholes in game scores, gaming human raters, hacking "
            "evaluation sets. Mitigations: robust reward modeling, adversarial evaluation, "
            "regularization, corrigibility."
        ),
    },
    "ai_timeline": {
        "category": "frontiers",
        "summary": "Estimates for when AGI might arrive.",
        "detail": (
            "Expert forecasts vary widely (2030-2100+). Key uncertainty drivers: scaling "
            "returns, algorithmic progress, compute growth, conceptual breakthroughs. "
            "Timelines matter because they set the urgency of safety work."
        ),
    },
    # --- Practical toolkits --------------------------------------------------------
    "pytorch": {
        "category": "tools",
        "summary": "Python deep-learning framework with dynamic computation graphs.",
        "detail": (
            "Tensor ops, autograd, nn modules, optimizers, DataLoader, TorchScript, "
            "distributed (DDP). Industry standard for research and production training."
        ),
    },
    "tensorflow": {
        "category": "tools",
        "summary": "Google's ML framework.",
        "detail": (
            "Keras high-level API, TF Serving, TF Lite for edge, TPU support. Static-graph "
            "heritage with eager mode; strong deployment story."
        ),
    },
    "jax": {
        "category": "tools",
        "summary": "NumPy-like library with autodiff and JIT compilation.",
        "detail": (
            "grad/vmap/pmap functional transforms; XLA compilation; popular for research "
            "(transformers, RL). Pure-functional style enables fast experimentation."
        ),
    },
    "onnx": {
        "category": "tools",
        "summary": "Open Neural Network Exchange: portable model format.",
        "detail": (
            "Interoperable graph format across frameworks; ONNX Runtime accelerates "
            "inference on CPU/GPU/edge. A common export target for deployed models."
        ),
    },
    "huggingface": {
        "category": "tools",
        "summary": "Ecosystem of pretrained models, datasets and libraries.",
        "detail": (
            "Transformers library, Hub, datasets, tokenizers, PEFT, diffusers. Note: AWEAI "
            "itself is intentionally Hugging Face-free, but the ecosystem is a key reference."
        ),
    },
    # --- Classical ML ---------------------------------------------------------------
    "linear_regression": {
        "category": "classical",
        "summary": "Predict continuous target as linear combination of features.",
        "detail": "Closed-form OLS solution or gradient descent. Assumes linearity; extend with polynomial features."
    },
    "logistic_regression": {
        "category": "classical",
        "summary": "Binary/multiclass classifier with sigmoid/softmax output.",
        "detail": "Cross-entropy loss; interpretable coefficients; a strong linear baseline."
    },
    "kmeans": {
        "category": "classical",
        "summary": "Partition data into k clusters by nearest centroid.",
        "detail": "Lloyd's algorithm (assign -> update). Sensitive to init; use k-means++ and elbow plot."
    },
    "dbscan": {
        "category": "classical",
        "summary": "Density-based clustering: groups dense regions, marks noise.",
        "detail": "No need to pre-specify k. Finds arbitrarily-shaped clusters; points in low-density areas are labeled noise (-1)."
    },
    "hierarchical_clustering": {
        "category": "classical",
        "summary": "Builds a tree of clusters (dendrogram) and cuts it at k.",
        "detail": "Agglomerative (bottom-up) with single/complete/average/Ward linkage. No k needed upfront; inspect the dendrogram to choose cut height."
    },
    "agglomerative_clustering": {
        "category": "classical",
        "summary": "Bottom-up hierarchical clustering (alias of hierarchical_clustering).",
        "detail": "Merges nearest clusters iteratively until k clusters remain; linkage determines inter-cluster distance."
    },
    "svm": {
        "category": "classical",
        "summary": "Support Vector Machine: maximum-margin classifier with kernels.",
        "detail": "Kernels (RBF, polynomial) map to high-dim space. Strong on small/medium tabular data."
    },
    "decision_tree": {
        "category": "classical",
        "summary": "Hierarchical if-then rules learned from data.",
        "detail": "Splits by information gain/Gini. Interpretable; prone to overfitting (use ensembles)."
    },
    "random_forest": {
        "category": "classical",
        "summary": "Ensemble of trees on bootstrap samples with random features.",
        "detail": "Bagging + feature subsampling reduce variance; robust default for tabular data."
    },
    "gradient_boosting": {
        "category": "classical",
        "summary": "Sequential trees fit to residuals of previous trees.",
        "detail": "XGBoost/LightGBM/CatBoost dominate tabular benchmarks. Additive model with shrinkage."
    },
    "pca": {
        "category": "classical",
        "summary": "Principal Component Analysis: linear dimensionality reduction.",
        "detail": "Projects data onto directions of max variance (eigen-decomposition of covariance)."
    },
    "naive_bayes": {
        "category": "classical",
        "summary": "Bayes classifier with conditional independence assumption.",
        "detail": "Fast, works on high-dim sparse text features; strong spam-filter baseline."
    },
    "knn": {
        "category": "classical",
        "summary": "K-Nearest Neighbors: predict by majority/mean of neighbors.",
        "detail": "Non-parametric; needs scaling; inference cost grows with dataset size."
    },
    # --- Metrics -----------------------------------------------------------------------
    "accuracy": {"category": "metrics", "summary": "Fraction of correct predictions.", "detail": "accuracy = correct / total. Misleading on imbalanced data."},
    "precision": {"category": "metrics", "summary": "Of predicted positives, how many are correct.", "detail": "precision = TP / (TP + FP). High precision = few false alarms."},
    "recall": {"category": "metrics", "summary": "Of actual positives, how many found.", "detail": "recall = TP / (TP + FN). High recall = few misses."},
    "f1": {"category": "metrics", "summary": "Harmonic mean of precision and recall.", "detail": "F1 = 2PR/(P+R). Useful single-number summary for imbalanced tasks."},
    "auc": {"category": "metrics", "summary": "Area under the ROC curve.", "detail": "Probability a random positive scores higher than a random negative. 0.5 = random, 1.0 = perfect."},
    "mse": {"category": "metrics", "summary": "Mean squared error for regression.", "detail": "Penalizes large errors quadratically; same units as target squared."},
    "mae": {"category": "metrics", "summary": "Mean absolute error for regression.", "detail": "Robust to outliers; interpretable in target units."},
    "r2": {"category": "metrics", "summary": "Coefficient of determination.", "detail": "Proportion of variance explained; 1.0 = perfect, 0 = baseline mean predictor."},
    "perplexity": {"category": "metrics", "summary": "Exponential of average negative log-likelihood for language models.", "detail": "Lower is better; 2^cross_entropy per token. Sensitive to tokenization."},
    "bleu": {"category": "metrics", "summary": "Precision-based n-gram overlap for machine translation.", "detail": "Reference-based; correlates weakly with human judgment; use with caution."},
    "rouge": {"category": "metrics", "summary": "Recall-oriented overlap metrics for summarization.", "detail": "ROUGE-N, ROUGE-L measure n-gram/LCS overlap with references."},
    # --- Optimizers -----------------------------------------------------------------------
    "sgd": {"category": "optimizers", "summary": "Stochastic Gradient Descent.", "detail": "Update w -= lr * grad. Simple; sensitive to lr and scaling; momentum improves it."},
    "adam": {"category": "optimizers", "summary": "Adaptive Moment Estimation.", "detail": "Per-parameter adaptive lr from first/second moments (beta1=0.9, beta2=0.999). Robust default."},
    "adamw": {"category": "optimizers", "summary": "Adam with decoupled weight decay.", "detail": "Applies weight decay directly to weights, not through gradients. Preferred for transformers."},
    "rmsprop": {"category": "optimizers", "summary": "RMSProp: scale lr by root mean square of recent gradients.", "detail": "Good for RNNs and non-stationary objectives."},
    # --- Activations ------------------------------------------------------------------------
    "relu": {"category": "activations", "summary": "max(0, x).", "detail": "Default hidden activation; cheap, avoids vanishing gradient; dead-neuron risk."},
    "gelu": {"category": "activations", "summary": "Gaussian Error Linear Unit.", "detail": "Smooth ReLU variant used in transformers (GPT/BERT)."},
    "softmax": {"category": "activations", "summary": "Converts logits to a probability distribution.", "detail": "Exponentiate + normalize. Temperature controls sharpness."},
    "sigmoid": {"category": "activations", "summary": "1/(1+e^-x) in (0,1).", "detail": "Binary probability output; saturates, causing vanishing gradients."},
    "tanh": {"category": "activations", "summary": "Hyperbolic tangent in (-1,1).", "detail": "Zero-centered; used in RNN cells."},
    # --- Losses ---------------------------------------------------------------------------------
    "cross_entropy": {"category": "losses", "summary": "Classification loss on predicted probabilities.", "detail": "L = -log p(y_true). Combine with softmax. Robust for classification."},
    "hinge_loss": {"category": "losses", "summary": "max(0, 1 - y*f(x)) for SVMs.", "detail": "Margin-based; encourages confident correct classification."},
    "huber_loss": {"category": "losses", "summary": "Huber: quadratic near zero, linear far away.", "detail": "Robust regression loss; less sensitive to outliers than MSE."},
    # --- Paradigms --------------------------------------------------------------------------------
    "contrastive_learning": {"category": "paradigms", "summary": "Pull positive pairs together, push negatives apart.", "detail": "SimCLR, MoCo, CLIP. Learns rich representations without labels."},
    "curriculum_learning": {"category": "paradigms", "summary": "Train on easy examples first, then hard.", "detail": "Mimics human learning; can improve convergence and final performance."},
    "active_learning": {"category": "paradigms", "summary": "Model chooses which data to label next.", "detail": "Reduces labeling cost; query by uncertainty, diversity or expected error reduction."},
    "semi_supervised": {"category": "paradigms", "summary": "Use a little labeled + lots of unlabeled data.", "detail": "Self-training, consistency regularization, pseudo-labeling."},
    "multi_task": {"category": "paradigms", "summary": "Train one model on several tasks at once.", "detail": "Shared representations + task heads; improves sample efficiency via transfer."},
    # --- Deployment -----------------------------------------------------------------------------
    "edge_ai": {"category": "deployment", "summary": "Run models on devices (phones, IoT, cars).", "detail": "Quantization, pruning, distillation, TFLite/ONNX Runtime, NPUs. Privacy + latency + offline."},
    "model_serving": {"category": "deployment", "summary": "Expose models via APIs at scale.", "detail": "Batching, GPU inference servers (Triton, vLLM), autoscaling, monitoring, canary rollout."},
    "mlops": {"category": "deployment", "summary": "ML engineering: data, train, deploy, monitor, retrain.", "detail": "Pipelines, experiment tracking, registries, CI/CD for models, drift detection."},
    "model_registry": {"category": "deployment", "summary": "Versioned store of models + metadata.", "detail": "Stages (staging/production), lineage, rollback, approvals. AWEAI's model zoo is a local registry."},
    # --- Domains -----------------------------------------------------------------------------------
    "nlp": {"category": "domains", "summary": "Natural Language Processing.", "detail": "Tasks: classification, NER, QA, translation, summarization, dialogue, sentiment, generation."},
    "computer_vision_tasks": {"category": "domains", "summary": "Vision task family.", "detail": "Classification, detection, segmentation, keypoints, depth, OCR, retrieval, generation, captioning."},
    "speech": {"category": "domains", "summary": "Speech processing.", "detail": "ASR (Whisper), TTS, speaker diarization, emotion, wake-word, speech translation."},
    "robotics": {"category": "domains", "summary": "Embodied AI.", "detail": "Perception + planning + control; RL for manipulation/locomotion; sim-to-real."},
    "time_series": {"category": "domains", "summary": "Forecasting and anomaly detection over time.", "detail": "ARIMA, RNN/LSTM/GRU, transformers, Prophet; features: trend, seasonality, lags."},
    "recommender": {"category": "domains", "summary": "Recommendation systems.", "detail": "Collaborative filtering, matrix factorization, two-tower retrieval, ranking models."},
    # --- Armenian / local notes ----------------------------------------------------------------------
    "armenian_ai": {
        "category": "community",
        "summary": "AI work related to the Armenian language and ecosystem.",
        "detail": (
            "Armenian NLP: morphological analysis, orthography converters (Armenian <-> Latin), "
            "datasets and MT. AWEAI supports Armenian via its i18n layer and text tools "
            "(detect_armenian, transliterate)."
        ),
    },
}


def search_concepts(query: str, category: Optional[str] = None, limit: int = 50) -> List[Dict[str, str]]:
    """Search the knowledge base by substring in name/summary/detail."""
    q = (query or "").strip().lower()
    out: List[Dict[str, str]] = []
    for name, entry in CONCEPTS.items():
        if category and entry["category"] != category:
            continue
        if q and q not in name.lower() and q not in entry["summary"].lower() and q not in entry["detail"].lower():
            continue
        row = {"name": name, "category": entry["category"], "summary": entry["summary"]}
        out.append(row)
        if len(out) >= limit:
            break
    return out


def get_concept(name: str) -> Optional[Dict[str, str]]:
    key = name.strip().lower().replace(" ", "_").replace("-", "_")
    entry = CONCEPTS.get(key)
    if entry is None:
        # allow fuzzy singular/plural
        if key.endswith("s") and key[:-1] in CONCEPTS:
            entry = CONCEPTS[key[:-1]]
    return entry


def categories() -> List[str]:
    seen: List[str] = []
    for e in CONCEPTS.values():
        if e["category"] not in seen:
            seen.append(e["category"])
    return seen


def concept_count() -> int:
    return len(CONCEPTS)


# ---------------------------------------------------------------------------
# AI timeline / roadmap (for `aweai ai roadmap` and wiki)
# ---------------------------------------------------------------------------
TIMELINE: List[Dict[str, str]] = [
    {"year": "1950", "event": "Turing proposes the Turing Test in 'Computing Machinery and Intelligence'."},
    {"year": "1956", "event": "Dartmouth workshop coins the term 'Artificial Intelligence'."},
    {"year": "1957", "event": "Perceptron introduced by Rosenblatt."},
    {"year": "1969", "event": "Backpropagation for multilayer perceptrons derived (Werbos; popularized 1986)."},
    {"year": "1980s", "event": "Expert systems boom; AI winter follows."},
    {"year": "1997", "event": "Deep Blue defeats Kasparov at chess."},
    {"year": "2011", "event": "Watson wins Jeopardy!; deep CNNs begin to dominate vision."},
    {"year": "2012", "event": "AlexNet wins ImageNet by a large margin — deep learning breakthrough."},
    {"year": "2014", "event": "GANs (Goodfellow et al.); attention mechanism introduced for MT."},
    {"year": "2016", "event": "AlphaGo beats Lee Sedol; AlphaFold emerges 2018-2021."},
    {"year": "2017", "event": "'Attention Is All You Need' — the Transformer architecture."},
    {"year": "2018", "event": "BERT and GPT-1; ELMo; pretraining becomes the norm."},
    {"year": "2020", "event": "GPT-3 (175B) shows few-shot in-context learning; scaling laws."},
    {"year": "2022", "event": "ChatGPT launches; text-to-image (DALL-E 2, Stable Diffusion) goes mainstream."},
    {"year": "2023", "event": "GPT-4, Claude, Gemini era; multimodal LLMs; open models (LLaMA, Mistral)."},
    {"year": "2024", "event": "Frontier open-weight models; agent frameworks; long-context models; on-device AI."},
    {"year": "2025", "event": "Agentic workflows, reasoning models, world models research intensify."},
    {"year": "2026", "event": "AWEAI v4 — universal CLI for AI/ASI/AGI engineering; continued frontier progress."},
]

AGI_LEVELS: List[Dict[str, str]] = [
    {"level": "0", "name": "No AI", "summary": "Pure mechanical automation, no learning."},
    {"level": "1", "name": "Narrow AI (ANI)", "summary": "Single-task specialists: chess engines, spam filters, image classifiers."},
    {"level": "2", "name": "Broad AI", "summary": "Multitask foundation models with general text/vision/audio ability."},
    {"level": "3", "name": "Competent AGI", "summary": "Meets human-level performance on most economic tasks."},
    {"level": "4", "name": "Expert AGI", "summary": "Exceeds most humans on most tasks; fast learning."},
    {"level": "5", "name": "Superhuman AGI (ASI)", "summary": "Exceeds all humans on virtually all tasks; self-improving."},
]

ROADMAP: List[Dict[str, str]] = [
    {"phase": "Now", "item": "Foundation models, agents, RAG, multimodal, on-device AI, open weights."},
    {"phase": "Near", "item": "Long-horizon agents, reliable tool use, world models, memory systems, self-correction."},
    {"phase": "Mid", "item": "Autonomous scientific discovery, robust alignment, interpretability, continual learning."},
    {"phase": "Far", "item": "AGI with cross-domain generality; then ASI governance and safety frameworks."},
]

# ---------------------------------------------------------------------------
# Self-improvement / recursive scaffolding
# ---------------------------------------------------------------------------
SELF_IMPROVEMENT_HOOKS: List[Dict[str, str]] = [
    {"name": "critique", "summary": "Generate output, then critique and revise it."},
    {"name": "self_test", "summary": "Write tests for your own output and fix failures."},
    {"name": "feedback_memory", "summary": "Store corrections and reuse them in later runs."},
    {"name": "data_augment", "summary": "Generate synthetic training data from your own successes/failures."},
    {"name": "prompt_optimize", "summary": "Track which prompts work and refine them."},
    {"name": "tool_learn", "summary": "Discover new tools/commands and how to call them."},
    {"name": "code_grow", "summary": "Emit patches to your own codebase; test before applying."},
    {"name": "recursive", "summary": "Improve the self-improvement loop itself (RSI hook)."},
]


def about() -> Dict[str, Any]:
    return {
        "concepts": concept_count(),
        "categories": len(categories()),
        "timeline_events": len(TIMELINE),
        "agi_levels": len(AGI_LEVELS),
        "roadmap_phases": len(ROADMAP),
        "self_improvement_hooks": len(SELF_IMPROVEMENT_HOOKS),
    }
