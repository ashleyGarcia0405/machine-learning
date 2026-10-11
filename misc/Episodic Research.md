https://pd-perry.github.io/posts/post-training.html
https://pd-perry.github.io/index.html
https://ai.stanford.edu/~cbfinn/_files/dissertation.pdf

Paper: https://gated-memory-policy.github.io/ 
A robot action policy with selective memory. It learns when historical information is useful and which parts to attend to. The authors find that simply giving a policy a longer history can hurt performance through overfitting and sensitivity to imperfect observations.
One example is casting an object across a surface with unknown friction. Earlier attempts reveal how the object moves, allowing the policy to adjust subsequent attempts. Another is returning a cup to its original location after that location leaves view. These test adaptation across attempts and memory within a task, respectively.
The policy adapts through its input context during execution; this is different from updating its trained weights after each attempt.



Paper (2023): https://arxiv.org/pdf/2307.09955
Cross-embodiment skill discovery as an unsupervised clustering problem.
Human demonstration videos are a widely available data source for robot learning and an intuitive user interface for expressing desired behavior. However, directly extracting reusable robot manipulation skills from unstructured human videos is challenging due to the big embodiment difference and unobserved action parameters. To bridge this embodiment gap, this paper introduces XSkill, an imitation learning framework that 1) discovers a cross-embodiment representation called skill prototypes purely from unlabeled human and robot manipulation videos, 2) transfers the skill representation to robot actions using conditional diffusion policy, and finally, 3) composes the learned skill to accomplish unseen tasks specified by a human prompt video.

Paper (2022): https://arxiv.org/pdf/2203.12601
We study how visual representations pre-trained on diverse human video data can enable data-efficient learning of downstream robotic manipulation tasks. Concretely, we pre-train a visual representation using the Ego4D human video dataset using a combination of time-contrastive learning, video-language alignment, and an L1 penalty to encourage sparse and compact representations. The resulting representation, R3M, can be used as a frozen perception module for downstream policy learning. Across a suite of 12 simulated robot manipulation tasks, we find that R3M improves task success by over 20% compared to training from scratch and by over 10% compared to state-of-the-art visual representations like CLIP and MoCo. Furthermore, R3M enables a Franka Emika Panda arm to learn a range of manipulation tasks in a real, cluttered apartment given just 20 demonstrations.