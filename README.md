Simple Feed Forawrd Neuaral Net.
The goal is to be able to classify the MNIST dataset which compromises of a few thousant 28*28 pixle images of hand drawn digits and the corrosponding labels.

Current testing with stochastic gradient decent the network appears to be collapsing. I decided to write an implementation 
in python as a sanity check and that works, meaning its an implementation issue. If I get time over the weekend Ill try 
finish it up.

Main Tasks still to do:
  - try a static configuration in c++
  - implement batching to the python version
  - final refactor to clean up.

Nice to have, time dependant:
  - parrallize batching and matrix operation, optimize matrix operations in general
  - visualizations for loss and error at various layers
  - support for different activation functions at different layers
