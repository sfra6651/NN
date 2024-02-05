Simple Feed Forawrd Neuaral Net.
The goal is to be able to classify the MNIST dataset which compromises of a few thousant 28*28 pixle images of hand drawn digits and the corrosponding labels.

Current testing with stochastic gradient decent the network appears to be collapsing. Need to test some more configurations and figure out the cause.
I believe it is the configuration rather an incorrect implementation.

Crucial Tasks still to do:
  - add in biases
  - batch proccessing
  - support for different activation functions at different layers

Nice to have, time dependant:
  - parrallize batching and matrix operation, optimize matrix operations in general
  - visualizations for loss and error at various layers
  - BIG refactor to clean things up, particulary layer storage.
