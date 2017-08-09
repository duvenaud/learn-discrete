# STA414 / STA2104 Winter 2017
## Statistical Methods for Machine Learning and Data Mining

<img src="https://raw.githubusercontent.com/jamesrobertlloyd/gpss-research/master/logo.png" width="500">

This course introduces machine learning to students with a statistical background.  Besides teaching standard methods such as logistic and ridge regression, kernel density estimation, and random forests, this course course will try to offer a broader view of model-building and optimization using probabilistic building blocks.

### What you will learn:

 * Standard statistical learning algorithms, when to use them, and their limitations.
 * The main elements of probabilistic models (distributions, expectations, latent variables, neural networks) and how to combine them.
 * Standard computational tools (Monte Carlo, Stochastic optimization, regularization, automatic differentiation). 

### Instructors:

* [David Duvenaud](http://www.cs.toronto.edu/~duvenaud), Office: 384 Pratt
   - Email: <duvenaud@cs.toronto.edu> (put "STA414" in the subject)
   - Lectures: Mondays 2-5pm, EM 119
   - Office hours: Mondays 11-12 noon in Pratt Building, Room 384

* [Mark Ebden](http://www.mebden.com/), Office: SS6026C and PT371
   - Email: mark [dot] ebden [at] utoronto [dot] ca
   - Lectures: Tuesdays 7-10 pm, SS1071
   - Office Hours:  Thursdays 3-4 pm in SS6026C, and after each lecture just outside the classroom itself
 
* The two instructors won't stick strictly to lecturing in their own sections. For example, on 16/17 January David Duvenaud will teach both sections (0101 and 0501), and in future sometimes Mark Ebden will teach both. This will occur regularly.


### Teaching Assistants:

* Amanjit Kainth
* Chris Cremer
* Luhui Gan
* Yang Guan Jian Guo (Tommy)

[Syllabus and Course Information](syllabus.pdf)

[Piazza](https://piazza.com/class/ivcpw2h2fq775m)

## Tentative Schedule

* **January 9 and 10:** Introduction.
   - [Intro + Background slides](lectures/lec1.pdf)
   - [Basic supervised learning and probability slides](lectures/lec1-part2-edited.pdf)
   - [Background quiz](lectures/skill-quiz.pdf)
   
   Readings: [Chapter 2 of David Mackay's textbook](http://www.inference.phy.cam.ac.uk/mackay/itprnn/ps/22.40.pdf)

* **January 16 and 17:** 
   - [The Exponential Family and beyond; Maximum Likelihood](lectures/Lecture2.pdf)
   - [Optimization](lectures/optimization.pdf)
   
   Readings:
   
   - [Chapter 3 of David Mackay's textbook](http://www.inference.phy.cam.ac.uk/mackay/itprnn/ps/47.59.pdf)
   - [Animations of different optimization algorithms](http://www.denizyuret.com/2015/03/alec-radfords-animations-for.html)
   
   Example code:
   
   - [Logistic regression autograd example](https://github.com/HIPS/autograd/blob/master/examples/logistic_regression.py)
   - [Neural net regression example](https://github.com/HIPS/autograd/blob/master/examples/neural_net_regression.py)
   - [Mixture of Gaussians example](https://github.com/HIPS/autograd/blob/master/examples/gmm.py)

* **January 23 and 24:** [Linear basis function models, decision theory, classification](lectures/Lecture3.pdf)

* **January 30 and 31:** [Bayesian inference, and kNN](lectures/Lecture4.pdf)

* **February 5:** [Assignment 1](assignments/HW1.pdf) due.

* **February 6 and 7:** [Classification](lectures/Lecture5.pdf)
   - [Classifier neural network demo](http://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4,2&seed=0.15656&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false)

* **February 13:** Midterm exam for both sections. (No class on Feb 14.) [Grade distribution](midtermMarksSTA414.png)

* **February 17 to 26:** Reading week

* **February 27 and 28:** [Mixture models](lectures/Lecture6-mixtures.pdf)

* **March 6 and 7:** [Continuous Latent variable models, and neural networks](lectures/Lecture7.pdf) Note: PPCA is now bonus material.

* **March 12:**  [Assignment 2](assignments/assignment2.pdf) due.
   
   - Helper code in [R](assignments/loadMNIST.R)
   - Helper code in [Python](assignments/data.py)
   - [Binary Logistic regression example](https://github.com/HIPS/autograd/blob/master/examples/logistic_regression.py)
   - Some Python and Numpy resources, copied from [Roger Grosse's neural networks course](http://www.cs.toronto.edu/~rgrosse/courses/csc321_2017/):
     - [Anaconda](https://store.continuum.io/cshop/anaconda/) provides an installer for Python and Numpy for Windows, Linux, and Mac.
     - [Numpy tutorial](http://www.engr.ucsb.edu/~shell/che210d/numpy.pdf)
     - [Learn X in Y minutes](http://learnxinyminutes.com/docs/python/) can get you up to speed in Python if you already know other languages.
     - [Lectures 2, 3, 4, and 6 for MIT's intro programming course](https://courses.edx.org/courses/MITx/6.00.1_4x/3T2014/courseware/Week_0/) Can help you get started if you don't have much programming background.
     
* **March 13 and 14:** [Sampling and Monte Carlo methods](lectures/lecture8-sampling.pdf)

    - Readings: [Chapter 29 of David Mackay's textbook](http://www.inference.phy.cam.ac.uk/itprnn/book.pdf)
    - Demos: [Interactive MCMC demos](https://chi-feng.github.io/mcmc-demo/)

* **March 20 and 21:** [Graphical models, and modelling sequential data](lectures/Lecture9_2017.pdf)

* **March 27 and 28:** [Stochastic Variational Inference and Variational autoencoders](lectures/09-svi.pdf)

  - Optional Reading: [Variational Inference: A Review for Statisticians](https://arxiv.org/pdf/1601.00670.pdf)

* **April 1:** Assignment 3 due at 1 pm. Questions are [here](assignments/assignment3.pdf)

* **April 3 and 4:** [Gaussian processes](lectures/Lecture11_GPs_2.pdf)

* **April 21:** Exam. Some warm-up problems are [here](assignments/practiceQs.pdf), and the first page of the exam will be posted soon.

