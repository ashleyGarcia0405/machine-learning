Blog: https://ae.studio/research/selfie
Paper: https://arxiv.org/abs/2602.10352

- Contrastive activation vectors: prompt the model with "Tell me about {{topic}}" for many topics, then subtract the mean activation. What's left isolates the direction for that specific topic.
- SAE features: Sparse autoencoders learn to decompose a model's activations into interpretable building blocks.

Both methods produce large datasets of (vector, label) pairs. Could we teach the model to write those labels itself. 
Approach: freeze the LLM entirely and train a tiny adapter (as few as d+1 embedding space). The model then autoregressively describes what the vector represents, interpreting it's own internals.
![[Pasted image 20260815201753.png]]
