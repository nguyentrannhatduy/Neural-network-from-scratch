# Building neural network from scratch 

This is just some kind of self-learning project, I want to have a deeply understanding about neural network, and how it works.
Calculus and linear algebra all of them hit me so hard, but it was also very amusing to apply those into a real thing rather than an excersise. on paper.
I was following this guide: https://blog.stackademic.com/learn-to-build-a-neural-network-from-scratch-yes-really-cac4ca457efc

## Description

I build a simple network with 2 hidden layer + 1 input layer + 1 output layer. I used sigmoid function for both hidden layer and output layer(it's not ideal).

This is model to predict cardiovascular disease.
Input: X[weight, height](pound and inches btw), y[int] (it contains right answer for corresponding weight, height in X).

Output: The same matrix structure as X that matrix has predictions.

## Requirement

Python 3.14.7 (The version I used to make this network)
numpy 2.5.3

## Room for improvement

As you can see using only sigmoid function for all the layers is not a good idea, from what I research it costs so much because of exponential calculation, and has vanishing gradient, exploding gradient problems. Probably, I can replace the activation function for hidden layers to ReLU.

So what if I want my output is more than just 0 or 1. That when softmax comes to play, by putting this function at the output layer instead sigmoid and replace loss function from binary cross-entropy to categorical cross-entropy. 

Actually, I will save it for my school project. 
School projects's idea: Imagine an in-game dialouge will show up when players want to communicate with npc, they can type anything. The model will predict the intention of players, and return the corresponding game feature.
Repo's link of that school project: (not now)
