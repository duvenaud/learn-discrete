STA 4273 / CSC 2547 Spring 2018:
# Learning Discrete Latent Structure

![A warped mixture model](http://www.cs.toronto.edu/~duvenaud/pictures/svae/spiral-small.png)

## Overview
New inference methods allow us to train learn generative latent-variable models.
These models can generate novel images and text, find meaningful latent representations of data, take advantage of large unlabeled datasets, and even let us do analogical reasoning automatically.
However, most generative models such as GANs and variational autoencoders currently have pre-specified model structure, and represent data using fixed-dimensional continuous vectors.
This seminar course will develop extensions to these approaches to learn model structure, and represent data using mixed discrete and continuous data structures such as lists of vectors, graphs, or even programs.
The class will have a major project component, and will be run in a similar manner to [Differentiable Inference and Generative Models](https://www.cs.toronto.edu/~duvenaud/courses/csc2541/index.html)

## Prerequisites:
This course is designed to bring students to the current state of the art, so that ideally, their course projects can make a novel contribution. A previous course in machine learning such as CSC321, CSC411, CSC412, STA414, or ECE521 is strongly recommended. However, the only hard requirements are linear algebra, basic multivariate calculus, basics of working with probability, and basic programming skills.

### Where and When

* Spring term, 2018
* Room: TBD
* Instructor: [David Duvenaud](http://www.cs.toronto.edu/~duvenaud)
* Email: <duvenaud@cs.toronto.edu> (put "STA4273" in the subject)
* Office hours: TBD in Room 384 Pratt
* Teaching assistants: TBD

## What is discete latent structure?
Loosely, this refers to generative models with discrete latent variables.


## Why discrete latent struture?

 * **Interpretability** 
 * **Computational efficency** - Making models fully differentiable sometimes requires us to sum over all possiblities to compute gradients, for instance in soft attention models.  Making hard choices about which computation to perform breaks differentiability, but is faster and requires less memory.
 * **Model search**
 * **Communication**
 * **Matches reality** (alphabets)


## Course Structure

After the first two lectures, each week a different student, or pair of students, will present on an aspect of these methods, using a couple of papers as reference.  I'll provide guidance about the content of these presentations.

In-class discussion will center around:

 * Understanding the strengths and weaknesses of these methods.
 * Understanding the relationships between these methods, and with previous approaches.
 * Extensions or applications of these methods.
 * Experiments that might better illuminate their properties.

The hope is that these discussions will lead to actual research papers, or resources that will help others understand these approaches.

Grades will be based on:

  * Class presentations - 20%
  * Project proposal - 20% - Due Oct 14th
  * Project presentation - 20% - Nov 18th and 25th
  * Project report and code - 40% - Dec 10th

### Project
Students can work on projects individually,in pairs, or even in triplets. The grade will depend on the ideas, how well you present them in the report, how clearly you position your work relative to existing literature, how illuminating your experiments are, and well-supported your conclusions are.

Each group of students will write a short (around 2 pages) research project proposal, which ideally will be structured similarly to a standard paper.
It should include a description of a minimum viable project, some nice-to-haves if time allows, and a short review of related work.
You don't have to do what your project proposal says - the point of the proposal is mainly to have _a_ plan and to make it easy for me to give you feedback.

Towards the end of the course everyone will present their project in a short, roughly 5 minute, presentation.

At the end of the class you'll hand in a project report (around 4 to 8 pages), ideally in the format of a machine learning conference paper such as [NIPS](https://nips.cc/Conferences/2016/PaperInformation/StyleFiles).

[Project report grading rubric](project-report-guidelines.html)



## Schedule


- **Structured encoder/decoders** [Slides](slides/structured-encoders-decoders.pdf)

	We have complete freedom in how we compute q(x | z).  There is also currently a lot of exploration going on of different types of generative models, p(x, z).
 
   - [Importance-Weighted Autoencoders](http://arxiv.org/abs/1509.00519) - The recognition network can return multiple weighted samples.
   - [Auxiliary Deep Generative Models](https://arxiv.org/pdf/1602.05473.pdf) - The model can be augmented with extra random variables that are then integrated out.

    
-  **Structured latent variables** [3D Latent rep Slides](slides/unsupervised-3d.pdf) [AIR Slides](slides/attend-infer-repeat.pdf) [SVAE Slides](slides/svae-slides.pdf)
	
	At first, variational autoencoders had only vector-valued latent variables z, in which the different dimensions had no special meaning.  People are starting to explore ways to put more meaningful structure on the latent description of data.

    - [Unsupervised Learning of 3D Structure from Images](http://arxiv.org/abs/1607.00662) - The latent variables can specify a 3D shape, letting us take advantage of existing renderers.
    - [Attend, Infer, Repeat: Fast Scene Understanding with Generative Models](http://arxiv.org/abs/1603.08575) - The latent variables can be a list or set of vectors.

   - [Composing graphical models with neural networks for structured representations and fast inference](http://arxiv.org/abs/1603.06277) - the prior on latent variables can be any tractable graphical model, and we can use this inference as part of the recognition step.


- **Dealing with non-differentiability**
    
    - [Gradient Estimation Using Stochastic Computation Graphs](https://arxiv.org/abs/1506.05254) Latent variables can be discrete, but this makes gradient estimation harder.  Also see the original [REINFORCE](http://incompleteideas.net/sutton/williams-92.pdf) paper.


- **Discrete Optimization Strategies**

   - Variatoinal Optimization

- **Latent-variable language models**

	- [Breaking Sticks and Ambiguities with Adaptive Skip-gram](http://arxiv.org/abs/1502.07257) - word2vec with multiple meanings for each word.
    - [Program Synthesis for Character-Level Language Modeling](http://openreview.net/pdf?id=ry_sjFqgx)
  - Part 1: Building open-ended languages of models [slides](slides/roger-part-1.pdf)
  - Part 2: Evaluating generative models [slides](slides/roger-part-2.pdf)

- **Program Induction**
   - [Sampling for Bayesian Program Learning](http://web.mit.edu/ellisk/www/programSample.pdf)


- **Projects due**
  
  
