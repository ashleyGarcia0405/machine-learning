https://www.youtube.com/watch?v=7jvWDF6ZXPc
https://yilundu.github.io/thesis.pdf

General Approach: Learn probabilistic generative (world) models that predict and plan adaptively at test time though search that compose to generalize where we have no training data.

changes in an environment can make a predicted action invalid. We can solve this issue by encoding "preferences" over each action. This corresponds to learning a probabilistic generative model over all of your action. Construct an energy landscape over actions using a probabilistic generative model. Scalar energy represents the negative utility for each possible action. Infer action to execute by searching for the action that minimizes the energy. Adapt to new test-time constraints such as costs on action (eg safety / battery life) by constructing new energy functions. Infer valid actions by optimizing ("searching") for predictions that minimize energy. 

### Why does GenAI work poorly on other settings?
I think a lot of this has to do with the differences of the training distributions. Compared to pixels in images (and many other continuous distributions), natural language is much simpler (structured for effective communication) and naturally compositional (infinite use of finite means).

Energy Based Models (EBMs) provide probabilistic manner to represent the real distribution as a composition of factors. You can compose factors together with this representation. So you can learn one model to capture one axis of the distribution and you can train a separate model to capture the second axis in the distribution.  

![[Pasted image 20261010003711.png]]

Should we just gather data to fill in all the gaps? 
BigGAN (2018) - We trained generative models conditions on class identities. Sample only high likelihood samples.
Stable Diffusion (2022) - We conditioned them on text identities. Sample from distribution p(x)(p(x|c)/p(x))^a. Very careful data filtering.

![[Pasted image 20261010004522.png]]

![[Pasted image 20261010005012.png]]

![[Pasted image 20261010005250.png]]

So the final expression is you draw some samples from your real data distribution, you decrease the energy of those samples, then you draw some samples from your model distribution and you increase the energy of those samples. 
Diffusion models are energy based models. 
### Composable Neural Network Construction with Loss Functions
Adding loss functions is a composable way to build neural networks with multiple properties. The issue is that often times at test time, you want a neural network that can generate predictions with a set of properties that you don't know at training. Adding predictions of neural networks cannot combine properties at prediction time. Learned neural networks are not composable!

EBMs are an intermediate loss compilation that enables composition through additional prediction time. 

### Compositional Sampling from Diffusion Models

![[Pasted image 20261010012546.png]]

![[Pasted image 20261010012832.png]]

  ### Unsupervised Discovery of Composable Components 
  Train generative model with factorized structure through maximum likelihood. 

### Planning
I think the bottleneck to this problem is a problem of demonstrations. We can reduce the data requirement necessary by decomposing demonstration data into a model capturing dynamics and a model capturing the goal of a task. This enables dynamics information solving one task to be stitched with goals from another task.

![[Pasted image 20261010203238.png]]

![[Pasted image 20261010203306.png]]

Possibility: Internet videos as a source of demonstrations. 

### Towards Intelligent Robots in the Physical World
Effective decision making requires many different aspects of knowledge.
- Semantic Knowledge: Need common sense knowledge about different tasks - where to look for the tea kettle or the high-level steps to steep the tea
- Visual Knowledge: Need visual knowledge about the world - the identity and intrinsic properties of objects and the 3D geometry of the world
- Physical Knowledge: Need physical knowledge about the world - the underlying material, physics, and masses of objects
- Control Knowledge: Need knowledge about how to make contact with objects and effective manipulate them to goals.

Idea - compose existing foundation models.
If you want to compose pre-trained model, you can view it as an energy function via its loss function.  e.g. you can derive an energy function from an LLM by just taking the likelihood loss the model has been trained on.

![[Pasted image 20261010205200.png]]

![[Pasted image 20261010205302.png]]

The previous consensus procedure is slow because language models often proposes plans not grounded in vision and the video likelihood is hard to optimize. 
Use a vision language model to propose visually grounded plans and as a heuristic to accelerate consensus. 

Long horizon decision making though hierarchical composition. Hierarchical planning substantially outperforms monolithic generative models. 

### Dissertation Text Notes
(any additional stuff that was unclear to me the first time)
A goal of AI is to construct an intelligent embodied agent that can autonomously operate and accomplish many unseen tasks in new environments the agent has not encountered before. For the agent to be successful, it's important that the agent can not only operate in distribution of tasks similar to the ones seen at training time but further generalize to new distributions seen at prediction time. 
In recent years, techniques from machine learning have led to remarkable advances in parts of artificial intelligence such as computer vision and natural language processing. However, such methods are derived from **statistical machine learning principles, which assume that both train and prediction distributions are identical**. To cover all the situations an embodied agent can find itself in, recent works have focused on collecting ever-larger training datasets. Such an approach does not address the fundamental problem of generalization and it is difficult to imagine being able to gather data to address every possible situation an agent can find itself in.