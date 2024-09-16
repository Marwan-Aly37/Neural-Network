# Neural-Network

## Perceptors:
* They take multiple inputs but only have one output
* Their output is always 1 or 0
* Can be Used as a NAND gate and since the NAND gate universal it can be used to represent other logic gates as well
* Each input is multiplies by weight
* Weight is a constant that represents the amount of importance the input has on the decision making
* Input is represented by the sum of the dot product between inputs and weights
* If the sum is bigger than the threshold Perceptor puts out a 1
* If the sum is smaller than the threshold Perceptor puts out a 0
* Equation: (sum(w.x)<>threshold) or sum(w.x)+d<>0 (Mostlu used)(d = -threshold)
* The problem with the perceptor is that its very difficult to implement into a learning module because since it only puts out 1 or 0 it is incapable of correcting itself through small changes(Correcting itself for one output ruins another output)
* The solution for this is to use Segmoid neurons

## Segmoid Neurons:
* Segmoid neurons are similar to perceptors but with one key difference
* Their outputs are not limtied to 1 and 0 
* They are also capable of displaying any number between 1 and 0 as output
* This makes them capable of correcting themselves through slight changes that don't effect other outputs
* Equation: 1/(1+e**(-z))
* z = sum(w.x)+d (x is the input)

## Sturcture of Network:
* Network consists of three main layers
* Input layer: Which takes the inputs
* Output layer: Which prints out the outputs
* Hidden layer: Which handels all the decision making
* Hidden layer can consist of more than one layer
* The more layers the hidden layer has the more capable the system will become to handle complex decision making
