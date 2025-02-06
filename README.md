# Text Summarization with BART and T5

## Introduction

Text summarization is a method of compressing a given document to summarize it while retaining the general idea of the original text. It has become a necessary tool in information processing due to the vast amount of textual data available. Summarization enhances productivity by reducing the time spent on reading long documents. For instance, it enables better decision-making in healthcare.

Text summarization is divided into two methods:
- **Abstractive Summarization**: Focuses on understanding the semantics of the original text and refining its ideas and concepts to achieve semantic reconstruction.
- **Extractive Summarization**: Selects key sentences or phrases directly from the original text to form a summary.

Since summaries are inherently abstractive, extractive summarization methods often produce long and redundant outputs, negatively impacting readability. In this project, we focus on **abstractive summarization**.

Abstractive summarization is often treated as a **sequence-to-sequence learning problem**. In this project, I implemented two state-of-the-art transformer-based models, **BART** and **T5**, for text summarization. Both models are pre-trained on large datasets, making them effective for summarization tasks.

Additionally, I fine-tuned BART using **Reinforcement Learning (RL)** to optimize performance beyond standard supervised fine-tuning. RL improves text generation by treating it as a sequential decision-making problem, fine-tuning model parameters to maximize evaluation metrics like **ROUGE**.

## Methodology

### Background

- **Seq2Seq Model**: First used by Rush et al. [1] for text summarization, it generates summaries by understanding text semantics, similar to manual summarization.
- **Attention Mechanism**: Introduced by Bahdanau et al. [2], it assigns weights to encoder hidden states to help the decoder focus on key words when generating summaries.

### Models Used

#### BART (Bidirectional and Auto-Regressive Transformers)

- Introduced by Lewis et al. [3], BART is a **transformer-based encoder-decoder (seq2seq) model**.
- The encoder is **bidirectional** (like BERT), and the decoder is **autoregressive** (like GPT).
- BART is known for producing **creative** and **coherent** abstractive summaries.

#### T5 (Text-to-Text Transfer Transformer)

- Introduced by Raffel et al. [5], T5 is also a **transformer-based encoder-decoder model**.
- Pre-trained on a combination of **unsupervised and supervised tasks**, making it highly **versatile**.
- Treats all NLP tasks in a unified **text-to-text** format, providing flexibility in text summarization.

Both BART and T5 models use an **encoder-decoder structure**:
- The **encoder** processes the input document to create a dense representation.
- The **decoder** generates the summary based on the encoded representation, predicting one token at a time while attending to previously generated tokens.

### Why These Models?

- **BART**: Chosen for its strong pre-training methods and excellent performance in abstractive summarization.
- **T5**: Used for comparison since it is commonly utilized in summarization research, offering a broader evaluation of model capabilities.

### Fine-Tuning with Reinforcement Learning

For the next step, I fine-tuned BART using **Reinforcement Learning (RL)** to enhance summarization quality.

- RL is commonly used to fine-tune large language models (LLMs) after self-supervised pretraining.
- The model learns to **optimize rewards**, which in this case is based on **ROUGE scores**.
- The **reward function** evaluates how well the generated summary aligns with the reference summary.
- The model updates its parameters to favor sequences with higher rewards, improving summarization quality.

## Discussion

### Strengths
- Pre-trained transformer models can generate high-quality summaries **without extensive computing resources**.
- Fine-tuning with RL allows models to focus on **quality-driven summarization metrics** beyond simple next-token prediction.

### Limitations
- **Limited training data**: Due to GPU constraints, the model was fine-tuned on only 50 documents, leading to **low ROUGE scores**.
- **Poor initialization**: If RL fine-tuning is applied without adequate supervised fine-tuning, the model can generate **meaningless outputs**.
- **Semantic faithfulness**: Both BART and T5 sometimes generate **inaccurate or hallucinated** information.
- **Readability issues**: The generated summaries may lack coherence and smoothness.

The **SeqCo** model in the reference paper provides a more reliable approach by addressing these issues with **contrastive learning**, making summarization more effective and readable.
### Before running
make sure to install these libraries
This is the list of Libraries that should be installed before running the code.

pip install transformers datasets torch torchvision torchaudio rouge-score sacrebleu
(copy and paste in terminal)
The data set will be load from datasets library.

https://github.com/EdinburghNLP/XSum/tree/master/XSum-Dataset

## References

1. Rush, A. M., Chopra, S., & Weston, J. (2015). A Neural Attention Model for Abstractive Sentence Summarization. *arXiv e-prints*.
2. Bahdanau, D., Brakel, P., Xu, K., et al. (2016). An Actor-Critic Algorithm for Sequence Prediction.
3. Lewis, M., Liu, Y., Goyal, N., Ghazvininejad, M., Mohamed, A., Levy, O., Stoyanov, V., & Zettlemoyer, L. (2019). BART: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension. *arXiv preprint arXiv:1910.13461*.
4. Ramadhan, M. R., Endah, S. N., & Mantau, A. B. J. (2020). Implementation of Textrank Algorithm in Product Review Summarization. *ICICoS 2020*.
5. Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., & Liu, P. J. (2019). Exploring the limits of transfer learning with a unified text-to-text transformer. *arXiv preprint arXiv:1910.10683*.
6. Badhe, S., Hasan, M., Rughwani, V., & Koshy, R. (2023). Synopsis Creation for Research Paper using Text Summarization Models. *4th International Conference for Emerging Technology (INCET 2023)*.
7. Cruz, D., Pona, E., Holness-Tofts, A., Schmied, E., et al. (2023). Reinforcement Learning Fine-Tuning of Language Models is Biased Towards More Extractable Features. *arXiv*.

---

This README provides a clear structure, making it easy for others to understand your project on GitHub. 🚀 Let me know if you want any further refinements!
