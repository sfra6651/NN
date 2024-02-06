Simple Feed Forawrd Neuaral Net.
The goal is to be able to classify the MNIST dataset which compromises of a few thousant 28*28 pixle images of hand drawn digits and the corrosponding labels.

Current testing with stochastic gradient decent the network appears to be collapsing. Need to test some more configurations and figure out the cause.
I believe it is the configuration rather an incorrect implementation. - cant figure out the issue. it should be working, clearly missing something.

Crucial Tasks still to do:
  - add in biases /done
  - batch proccessing /adding in basline as part of refactor
  - support for different activation functions at different layers /re-prioritising to nice to have
  - support for different layer depths /refactor will support this

Nice to have, time dependant:
  - parrallize batching and matrix operation, optimize matrix operations in general
  - visualizations for loss and error at various layers
  - BIG refactor to clean things up, particulary layer storage.
