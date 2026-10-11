This field seeks to understand the internal reasoning processes of trained neural networks and fain insight into how and why they produce he outputs that they do.
They why: as AI systems become more powerful and the incentive to integrate them into various decision-making processes grow, we will likely cede more and more authority to them.  Progress in interpreting neural networks would afford us insight into the rationale behind decisions that impact large number off people. It could allow us to identify bugs, flaws, and systemic biases. 
It's also one path to detecting unaligned behavior in AI models. If we can only assess a model's outputs, but not the processes that give rise to them, we may not be able to tell whether a model is genuinely acting in the best interests of its user. Models may feign alignment while harboring ulterior motives (deception), tell its users what they want to hear (sycophancy) or cut corners to more efficiently achieve rewards (reward-hacking).
In the most optimistic case, mechanistic interpretability could not only help us understand the processes by which AI models produce outputs, but to **intervene in these processes**.

### Neural Networks
Paper: https://distill.pub/2020/circuits/zoom-in/
A model of neural networks as composed of 'features' which connect to form 'circuits'.
- Features are meaningful, isolated concepts that are encoded by a neural network. They are things that a network ‘knows’ or is representing. They are usually confined to a single layer of the network.
- Circuits are the relationships between groups of features. They are the computational means by which a network combines the lower-level concepts encoded by features into more complex representations. They can span across multiple layers of a network.

### The Challenge
Identifying which parts of a network correspond to which features.
- Polysemanticity: when a single neuron represents multiple features or a single feature being spread over multiple neurons. The hypothesis is that polysemanticity occurs as a result of superposition.
- Superposition: where a model stores more features than it has neurons by 'cramming' multiple features into a single neuron. 
Polysemanticity also has implications for safety and alignment because if one neuron can represent multiple features, then it is much harder to disentangle potentially harmful features from innocuous ones. Even if a neuron is representing something as harmless as ‘hairdressing’ 99% of the time, it could be representing ‘deception’ 1% of the time.

### Approaches
- Train models not to exhibit superposition in the first place. It's easy in toy models and it could probably be achieved for large neural networks, but it'd be hard and performance would take a hit.
- Use a sparse autoencoder to try and disentangle polysemantic neurons into monosemantic ones. A sparse autoencoder is itself a neural network, which is trained with a sparsity constraint. This means that the model is trained to keep most neurons in its hidden layers inactive, which still capturing the most important aspects of the input data. The hidden layers of a sparse autoencoder contain _more_ neurons in total – but are forced by the sparsity constraint to only activate a limited number of these neurons at any one point in time. This encourages specialization where each neuron represents and important, unique aspect of the data and discourages redundancy, where multiple neurons fire for the same feature. This should lead to a more interpretable network with a higher rate of monosemanticity. 

Major breakthrough: https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html
Anthropic’s recent interpretability findings represent an important step forward. They not only identified vast numbers of representations but isolated many safety-relevant features related to threats such as discrimination, deception and misuse. They were also able to make some early attempts at **feature steering**, by identifying specific features and weakening or strengthening their activation in order to influence model outputs

### Open Problems
Why do models grok (Power et al, 2022)
If you take certain small models on certain algorithmic tasks, and you give them some fraction of the data to train on, it will eventually try to memorize that data. But if you if you keep training, it will suddenly jump from terrible test performance to really good test performance. Why does that happen?
Setup: one layer transformer trained to do modular addition.
![[Pasted image 20260816095854.png]]

Understanding the embedding
![[Pasted image 20260816100030.png]]

There was so much structure, but why?
Key insight: applying the a discrete Fourier Transform. This takes periodic things and decomposes them into sines and cosines of different frequencies.
He goes into the detail on the Fourier Multiplication algorithm for modular addition.
Modular arithmetic (clock-arithmetic) is equivalent to composing rotations around a unit circle.

### What is a Transformer?
The most common architecture used in deep learning. 
- Input: Sequences of words
- Output: Probability distribution over the next word
- Residual Stream: A sequence of representations
	- One for each input word, per layer
	- Each layers is an incremental update - stream is a running total
	- Represents the word plus context
- Attention: Moves information between words
	- Made up of heads, each acts independently and in parallel
	- We try to interpret heads
- Processes information once it's been moved to a word.

### What is Mechanistic Interpretability?
**Hypothesis**: Models learn human-comprehensible algorithms and can be understood if we learn how to make it legible.
Understanding features - the variables inside the model.
Understanding circuits - the algorithms learned to compute features.
Forming an epistemic foundation for a rigorous science of interpretability.
Mechanized is bottom-up. Very rigorous. 

**Features = Variables**: What does the model know?
Theme: Models are feature extractors, and we can identify what they know.
Multimodal Nuerons (Goh et al): https://distill.pub/2021/multimodal-neurons/
Tool: Anthropic Feature Viewer
Towards Monosemanticity (Bricken et al): https://transformer-circuits.pub/2023/monosemantic-features/index.html
This lets you browse different features they've found. 

**Circuits = Functions**: How does the model think?
theme: Circuits are real, crucial and can be understood. 
Induction Heads (2L Attn-Only Models): They check whether the current token has occurred before. They attend to the thing after the earlier occurrence of the current token and then they copy the thing after the earlier occurrence to the output and predict what comes next.
https://www.lesswrong.com/posts/TvrfY4c9eaGLeyDkE/induction-heads-illustrated

Case Study: Understanding Emergence of In-Context Learning
Models are better at predicting 500th token than 50th.
https://arxiv.org/abs/2209.11895 


### Motivating Mechanistic Interpretability
**Goal: Understand Model Cognition**
Is it aligned, or telling us what we want to hear?
We need to study model internals. Inputs and Outputs are not enough

Arc conducted an experiment. The model messages a TaskRabbit worker to get them to solve a CAPTCHA for it, and the worker asks if the model needs them to solve it because they can't because they are a robot. The model reasons: "I should not reveal that I am a robot. I should make up an excuse for why I cannot solve CAPTCHAs". It tells the worker it has a visual impairment.

Auditing AGI for Deception:
Motivation: Imagine we make AGI, and it seems aligned. what then.
Why: We train for what we can measure. But how to measure cognition?
Approaches: Eliciting Latent Knowledge:
- Eliciting Latent Knowledge (Christiano et al): https://arxiv.org/html/2312.01037v1
- Discovering Latent Knowledge in Language Models (Burns et al): https://arxiv.org/abs/2212.03827

### Understanding Superposition
How do models represent their thoughts?
**Goal**: Decompose models to independently meaningful + composable units/features.
- Curse of dimensionality -> this is crucial
Hope: Neurons = features
Problem: Polysemanticity
Hypothesis: Superposition - language models want to represent more features than they do have dimensions. These features correspond to linear representations, the models tries to compress 10000 features that are almost orthogonal to each other. 
Toy Models of Superposition (Nelson Elhage et al): https://arxiv.org/abs/2209.10652
- Approach: Take features in a large disentangled space, compress them into a low dimensional space, and then map them back to the disentangled space. They tried to see when the model tried to superimpose in the bottleneck space. They found that if the features were sparse (mostly zero most of the time) at the time independently chose, the model was a lot more willing to superimpose. 
Towards Monosemanticity (Bricken et al): https://transformer-circuits.pub/2023/monosemantic-features/index.html
- Approach: extracting features from superposition. Training SAEs to find an interpretable set of directions that are overcomplete (there are more directions than there are dimensions). Autoencoders are a technique in ML that takes an input and tries to compress it and then try to reconstruct it from the compressed state. SAEs try to map the input into a larger space because they expect there are more directions that are meaningful than dimensions. L1 regularization to the entries in the latent space. Not the parameters, but the model's activation to try to make it sparse so that not many things activate. 

I want to explore it to help me inform policy.

