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
* Location: Galbraith 119
* Time: Fridays, 2-4pm
* Office hours: Mondays, 3:30-4:30pm, in 384 Pratt
* Piazza: [https://piazza.com/utoronto.ca/winter2018/csc2547/](https://piazza.com/utoronto.ca/winter2018/csc2547/)

## What is discrete latent structure?
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

  * [15%] One [assignment](assignments/assignment.pdf), designed to help you become familiar with unbiased gradient estimators, such as [REINFORCE](http://incompleteideas.net/sutton/williams-92.pdf) (also known as the score-function estimator) and [REBAR](https://arxiv.org/abs/1703.07370)
  * [15%] Class presentations
  * [15%] Project proposal
  * [15%] Project presentation
  * [40%] Project report and code

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
---
### Week 1 - Jan 12th - Optimization, integration, and the reparameterization trick

This lecture will set the scope of the course, the different settings where discrete structure must be estimated or chosen, and the main existing approaches.  As a warm-up, we'll look at the REINFORCE and reparameterization gradient estimators.

[Lecture 1 slides](slides/lec1-intro.pdf)

***
### Week 2 - Jan 19th - Gradient estimators for non-differentiable computation graphs

[Lecture 2 slides](slides/lec2-gradient-estimators.pdf)

Discrete variables makes gradient estimation hard, but there has been a lot of recent progress on developing unbiased gradient estimators.
  
Recommended reading:
  - [Gradient Estimation Using Stochastic Computation Graphs](https://arxiv.org/abs/1506.05254)
  - [Backpropagation through the Void: Optimizing control variates for black-box gradient estimation](https://arxiv.org/abs/1711.00123)
  
Material that will be covered:
  - The original [REINFORCE](http://www-anw.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf) paper.
  - [The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables](https://arxiv.org/abs/1611.00712)
  - [Categorical Reparameterization with Gumbel-Softmax](https://arxiv.org/abs/1611.01144)
  - [REBAR: Low-variance, unbiased gradient estimates for discrete latent variable models](https://arxiv.org/abs/1703.07370)
  - [Stochastic Backpropagation and Approximate Inference in Deep Generative Models](https://arxiv.org/abs/1401.4082)
  - [Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation](https://arxiv.org/abs/1308.3432)

Related work:
  - [MuProp: Unbiased Backpropagation for Stochastic Neural Networks](https://arxiv.org/abs/1511.05176)
  - [The Generalized Reparameterization Gradient](https://arxiv.org/abs/1610.02287)
  - [Developing Bug-Free Machine Learning Systems With Formal Mathematics](https://arxiv.org/pdf/1706.08605.pdf) - One can use formal tools to verify that a gradient estimator is unbiased.
    
***
### Week 3 - Jan 26th - Deep Reinforcement learning and Evolution Strategies
  - [A Visual Guide to Evolution Strategies](http://blog.otoro.net/2017/10/29/visual-evolution-strategies/)
  - [Evolution Strategies as a Scalable Alternative to Reinforcement Learning](https://arxiv.org/abs/1703.03864)
  - [Optimization by Variational Bounding](https://www.elen.ucl.ac.be/Proceedings/esann/esannpdf/es2013-65.pdf)
  - [Natural Evolution Strategies](http://www.jmlr.org/papers/volume15/wierstra14a/wierstra14a.pdf)
  - [Parallel Gaussian Perturbation](https://arxiv.org/abs/1703.03864)
  - [Emergence of Grounded Compositional Language in Multi-Agent Populations](https://arxiv.org/abs/1703.04908)
  - [Model-Based Planning in Discrete Action Spaces](https://arxiv.org/abs/1705.07177) - "it is in fact possible to effectively perform planning via backprop in discrete action spaces"
  - [Q-Prop: Sample-Efficient Policy Gradient with An Off-Policy Critic](https://arxiv.org/abs/1611.02247) - learns a linear surrogate function.
  
***
### Week 4 - Feb 2nd - Differentiable Data Structures and Adaptive Computation

Attempts learn programs using gradient-based methods, and program induction in general.
 - [Pointer Networks](https://arxiv.org/abs/1506.03134)
 - [Neural Turing Machines](https://arxiv.org/abs/1410.5401)
 - [Reinforcement Learning Neural Turing Machines](https://arxiv.org/abs/1505.00521)
 - [Recurrent Models of Visual Attention](https://arxiv.org/pdf/1406.6247.pdf) - Training a hard attention model inside an RNN.
 - [Programming with a Differentiable Forth Interpreter](https://arxiv.org/abs/1605.06640)
 - [Sampling for Bayesian Program Learning](http://web.mit.edu/ellisk/www/programSample.pdf)
 - [Neural Sketch Learning for Conditional Program Generation](https://openreview.net/pdf?id=HkfXMz-Ab)
 - [Adaptive Computation Time for Recurrent Neural Networks](https://arxiv.org/abs/1603.08983)
 - [The Case for Learned Index Structures](https://arxiv.org/abs/1712.01208)
 - [Reparameterization Gradients through Acceptance-Rejection
Sampling Algorithms](http://proceedings.mlr.press/v54/naesseth17a/naesseth17a.pdf)
 - [Divide and Conquer Networks](https://arxiv.org/abs/1611.02401)

***
### Week 5 - Feb 9th - Discrete latent structure

Variational autoencoders and GANs typically use continuous latent variables, but there is recent work on getting them to use discrete random variables.
  - [The Helmholtz Machine](http://www.gatsby.ucl.ac.uk/~dayan/papers/hm95.pdf) - The forerunner of VAEs used binary latent variables.
  - [Attend, Infer, Repeat: Fast Scene Understanding with Generative Models](http://arxiv.org/abs/1603.08575) - The latent variables can be a list or set of vectors.
  - [Composing graphical models with neural networks for structured representations and fast inference](http://arxiv.org/abs/1603.06277)  the prior on latent variables can be any tractable graphical model, and we can use this inference as part of the recognition step.
  - [Learning Hard Alignments with Variational Inference](https://arxiv.org/pdf/1705.05524.pdf) - in machine translation, the alignment between input and output words can be treated as a discrete latent variable.
  - [Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937) - trains an RNN with discrete hidden units, using the straigh-through estimator.

Related work:
  - [Neural Variational Inference and Learning in Belief Networks](https://arxiv.org/abs/1402.0030)
  - [Variational inference for Monte Carlo objectives](https://arxiv.org/abs/1602.06725)


***
### Week 6 - Feb 16th - Adversarial training and text models

It's not obvious how to train GANs to produce discrete structures, because this cuts off the gradient to the discriminator.
  - [Connecting Generative Adversarial Networks and Actor-Critic Methods](https://arxiv.org/abs/1610.01945)
  - [Adversarial Autoencoders](https://arxiv.org/abs/1511.05644) - One surprisingly effective hack for training discrete random variables is to let them be continuous, and have a discriminator check if they're discrete.
  - [Adversarially Regularized Autoencoders for Generating Discrete Structures](https://arxiv.org/abs/1706.04223)
  - [GANS for Sequences of Discrete Elements with the Gumbel-softmax Distribution](https://arxiv.org/abs/1611.04051)
  - [Program Synthesis for Character-Level Language Modeling](http://openreview.net/pdf?id=ry_sjFqgx)
  - [Hierarchical Multiscale Recurrent Neural Networks](https://arxiv.org/abs/1609.01704)
  - [Generating and designing DNA with deep generative models](https://arxiv.org/abs/1712.06148)
  

***
### Week 7 - Feb 23rd - Bayesian nonparametrics

Models of infinitely-large discrete objects.
  - [Slides on Bayesian nonparametrics](http://stat.columbia.edu/~porbanz/talks/nipstutorial.pdf)
  - [Lecture notes on Bayesian nonparametrics](http://stat.columbia.edu/~porbanz/papers/porbanz_BNP_draft.pdf)
  - [Warped Mixtures for Nonparametric Cluster Shapes](https://arxiv.org/abs/1206.1846)
  - [Structure Discovery in Nonparametric Regression through Compositional Kernel Search](https://arxiv.org/abs/1302.4922)
  - [Learning the Structure of Deep Sparse Graphical Models](https://arxiv.org/abs/1001.0160)
  - [Probabilistic programming](https://probmods.org/) - Automatic inference in arbitary models specified by a program.
  - [Breaking Sticks and Ambiguities with Adaptive Skip-gram](http://arxiv.org/abs/1502.07257) - word2vec with multiple meanings for each word. 
 
***
### Week 8 - March 2nd - Learning model structure
  - [The discovery of structural form](http://www.pnas.org/content/105/31/10687.full) - put a grammar on model structures and built a different model for each dataset automatically.
  - [Exploiting compositionality to explore a large space of model structures](https://arxiv.org/abs/1210.4856) - another systematic search through model structure using a grammar.
  - [Bayesian Compression for Deep Learning](https://arxiv.org/abs/1705.08665) - putting a sparse prior on a neural network's weights is a principled way to learn its structure.
  - [SMASH: One-Shot Model Architecture Search through HyperNetworks](https://arxiv.org/pdf/1708.05344.pdf)

***
### Week 9 - March 9th - Graphs, permutations and parse trees
  - [Learning Latent Permutations with Gumbel-Sinkhorn Networks](https://openreview.net/forum?id=Byt3oJ-0W)
  - [Reparameterizing the Birkhoff Polytope for Variational Permutation Inference](https://arxiv.org/abs/1710.09508)
  - [Grammar Variational Autoencoder](https://arxiv.org/pdf/1703.01925.pdf)
  - [Automatic chemical design using a data-driven continuous representation of molecules](https://arxiv.org/abs/1610.02415)
  - [Learning to Compose Words into Sentences with Reinforcement Learning](https://arxiv.org/abs/1611.09100)
  - [Learning to Represent Programs with Graphs](https://arxiv.org/abs/1711.00740)


***
### Week 10 - March 16th - Project presentations I

***
### Week 11 - March 23rd - Project presentations II

***
### Week 12 - March 30th - Good Friday (Holiday)

