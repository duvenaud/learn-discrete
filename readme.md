STA 4273 / CSC 2547 Spring 2018:
# Learning Discrete Latent Structure
<img src="http://www.cs.toronto.edu/~duvenaud/pictures/svae/spiral-small.png" width="200"><img src="http://www.cs.toronto.edu/~duvenaud/pictures/autochem-icon.png" width="200"><img src="http://www.cs.toronto.edu/~duvenaud/pictures/neuralfp-icon.png" width="200">

## Overview
New inference methods allow us to train learn generative latent-variable models.
These models can generate novel images and text, find meaningful latent representations of data, take advantage of large unlabeled datasets, and even let us do analogical reasoning automatically.
However, most generative models such as GANs and variational autoencoders currently have pre-specified model structure, and represent data using fixed-dimensional continuous vectors.
This seminar course will develop extensions to these approaches to learn model structure, and represent data using mixed discrete and continuous data structures such as lists of vectors, graphs, or even programs.
The class will have a major project component, and will be run in a similar manner to [Differentiable Inference and Generative Models](https://www.cs.toronto.edu/~duvenaud/courses/csc2541/index.html)

## Prerequisites:
This course is designed to bring students to the current state of the art, so that ideally, their course projects can make a novel contribution. A previous course in machine learning such as CSC321, CSC411, CSC412, STA414, or ECE521 is strongly recommended. However, the only hard requirements are linear algebra, basic multivariate calculus, basics of working with probability, and basic programming skills.

To check if you have the background for this course, try taking this [Quiz](skill-quiz/skill-quiz.pdf).  If more than half the questions are too difficult, you might want to put some extra work into preparation.

### Where and When

* Spring term, 2018
* Instructor: [David Duvenaud](http://www.cs.toronto.edu/~duvenaud)
* Email: <duvenaud@cs.toronto.edu> (put "STA4273" in the subject)
* Location, office hours, teaching assistants: TBD

## What is discete latent structure?
Loosely speaking, it referes to any discrete quantity that we wish to estimate or optimize.
Concretely, in this course we'll consider using gradient-based stochastic optimization to train models like:
	* Variational autoencoders with latent binary vectors, mixture models, or lists of vectors
	* Differentiable versions of stacks, deques, and Turing machines
	* Generative models of text, graphs, and programs
	* Tree-structured recursive neural networks

## Why discrete latent struture?

 * **Computational efficency** - Making models fully differentiable sometimes requires us to sum over all possiblities to compute gradients, for instance in soft attention models.  Making hard choices about which computation to perform breaks differentiability, but is faster and requires less memory.
 * **Reinforcement learning** - In many domains, the set of possible actions is discrete.  Planning and learning in these domains requires integrating over possible future actions.
 * **Interpretability and Communication** - Models with millions of continuous parameters, or vector-valued latent states, are usually hard to interpret.  Discrete structure is easier to communicate using language.  Conversely, communicating using words is an example of learning and planning in a discrete domain. 
 
## Why not discrete latent struture?

 - **It's hard to compute gradients** - It's hard to estimate gradients through functions of discrete random variables.  It is so difficult that much of this course will be dedicated to investigating different techniques for doing so.  Developing these techniques are an active research area, with several large developments in the last few years.

## Course Structure

Aside from the first two and last two lectures, each week a different group of students will present on a set of related papers covering an aspect of these methods.  I'll provide guidance to each group about the content of these presentations.

In-class discussion will center around understanding the strengths and weaknesses of these methods, their relationships, possible extensions, and experiments that might better illuminate their properties.

The hope is that these discussions will lead to actual research papers, or resources that will help others understand these approaches.

Grades will be based on:

  * One assignment, designed to help you become familiar with unbiased gradient estimators, such as [REINFORCE](http://incompleteideas.net/sutton/williams-92.pdf) (also known as the score-function estimator) and [REBAR](https://arxiv.org/abs/1703.07370)
  * Class presentations
  * Project proposal
  * Project presentation
  * Project report and code

### Project
Students can work on projects individually,in pairs, or even in triplets. The grade will depend on the ideas, how well you present them in the report, how clearly you position your work relative to existing literature, how illuminating your experiments are, and well-supported your conclusions are.
Full marks will require a novel contribution.

Each group of students will write a short (around 2 pages) research project proposal, which ideally will be structured similarly to a standard paper.
It should include a description of a minimum viable project, some nice-to-haves if time allows, and a short review of related work.
You don't have to do what your project proposal says - the point of the proposal is mainly to have _a_ plan and to make it easy for me to give you feedback.

Towards the end of the course everyone will present their project in a short, roughly 5 minute, presentation.

At the end of the class you'll hand in a project report (around 4 to 8 pages), ideally in the format of a machine learning conference paper such as [NIPS](https://nips.cc/Conferences/2016/PaperInformation/StyleFiles).

Projects will be graded according to an updated version of [Last year's project report grading rubric](project-report-guidelines.html)



## Tentative Schedule

- **Dealing with non-differentiability** - Discrete variables makes gradient estimation harder.
    - The original [REINFORCE](http://incompleteideas.net/sutton/williams-92.pdf) paper.
    - [Gradient Estimation Using Stochastic Computation Graphs](https://arxiv.org/abs/1506.05254)
    - [REBAR: Low-variance, unbiased gradient estimates for discrete latent variable models](https://arxiv.org/abs/1703.07370)
    - [https://arxiv.org/abs/1308.3432](Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation)
    
- **Differentiable Data Structures**
    - [Neural Turing Machines](https://arxiv.org/abs/1410.5401)
    - [Reinforcement Learning Neural Turing Machines](https://arxiv.org/abs/1505.00521)
    - [Neural Networks, Types, and Functional Programming](http://colah.github.io/posts/2015-09-NN-Types-FP/)
    - [Pointer Networks](https://arxiv.org/abs/1506.03134)
    
-  **Discrete latent structures** - Variational autoencoders and GANs typically use continuous latent variables.
    - [The Helmholtz Machine](http://www.gatsby.ucl.ac.uk/~dayan/papers/hm95.pdf) - The forerunner of VAEs used binary latent variables.
    - [Attend, Infer, Repeat: Fast Scene Understanding with Generative Models](http://arxiv.org/abs/1603.08575) - The latent variables can be a list or set of vectors.
   - [Composing graphical models with neural networks for structured representations and fast inference](http://arxiv.org/abs/1603.06277) - the prior on latent variables can be any tractable graphical model, and we can use this inference as part of the recognition step.

- **Reinforcement learning**
    - [Connecting Generative Adversarial Networks and Actor-Critic Methods](https://arxiv.org/abs/1610.01945)

- **Adversarial training**
    - [Adversarially Regularized Autoencoders for Generating Discrete Structures)(https://arxiv.org/abs/1706.04223)

- **Latent-variable language models**
    - [Breaking Sticks and Ambiguities with Adaptive Skip-gram](http://arxiv.org/abs/1502.07257) - word2vec with multiple meanings for each word.
    - [Program Synthesis for Character-Level Language Modeling](http://openreview.net/pdf?id=ry_sjFqgx)

- **Program Induction**
   - [Sampling for Bayesian Program Learning](http://web.mit.edu/ellisk/www/programSample.pdf)
   - [Programming with a Differentiable Forth Interpreter](https://arxiv.org/abs/1605.06640)

- **Project Presentations I**

- **Project Presentations II**

  
  
