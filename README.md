## Higher Moment ADAM Estimations

This project is an adaptation on the original Vanilla ADAM optimization algorithm, by using different metrics of estimation for the normalization of the gradient. We expand on existing literature of Higher Moment ADAM, and apply it to a residual neural network.

We apply this new Optimizer on the CIFAR 10 data set for the 10 set classification. 

In traditional ADAM, we have the following update equation 

<img src="https://github.com/Joshuashou/Higher-Order-ADAM-Optimization/blob/master/Equations/First_Order_Moving_Average.png" width="500" height="80">

<img src="https://github.com/Joshuashou/Higher-Order-ADAM-Optimization/blob/master/Equations/Second_Order_Moving_Average.png" width="500" height="80">

We then employ the following update equation:

<img src="https://github.com/Joshuashou/Higher-Order-ADAM-Optimization/blob/master/Equations/Update_Equations.png" width="500" height="300">


Essentially, we want to use the higher moment estimate of the gradient to control the update sizing based on the variance of our gradients estimates. 

In higher order ADAM, we want to see if we can have better results of variance estimation by using higher moment estimations.

<img src="https://github.com/Joshuashou/Higher-Order-ADAM-Optimization/blob/master/Equations/Nth_Order_Moving_Average.png" width="500" height="80">

<img src="https://github.com/Joshuashou/Higher-Order-ADAM-Optimization/blob/master/Equations/Nth_Order_Update.png" width="500" height="100">


## RESNET Results 

We use an 18 layer RESNET with 7x7 Kernel, Stride 2, and 3x3 max pooling layer, along with skip connections and batch normalizations. Training done with NVIDIA GPU through colab. 

Firstly, we have that any odd moments of estimations lead to divergence of training. Our training loss goes to infinity, and our accuracy goes to 10%, which is esentially random guessing. This is due to the skewness of odd distributions potentially having values that make it lower than the gradient, or even negative.



Sample results for 4th moment Training and Testing Results with Higher Order ADAM, Vanilla ADAM, and ADAMAX.

<p>
  <img width="400" alt="Train Accuracies" src="https://github.com/Joshuashou/Higher-Order-ADAM-Optimization/blob/master/Simulation_Results/Train_Accuracies.png" style="display: inline-block;">
  <img width="400" alt="Test Accuracies" src="https://github.com/Joshuashou/Higher-Order-ADAM-Optimization/blob/master/Simulation_Results/Test_Accuracies.png" style="display: inline-block;">
</p>




## File Descriptions

main.py - Runs experiments for the RESNET with various Optimizers. 

Adamax.py - Contains code for ADAMAX Optimizer estimations. 

Vanilla_Adam.py defines the original ADAM optimizer.

Vanilla_Adam.py is our edited files with various levels of degree estimations of gradients. 

Graphs.ipynb explores accuracy and loss results through epochs.


Resnet.py constructs the residual neural neural network with different bottleneck and block layers. 


Sources:

Adam: A Method for Stochastic Optimization, 22 Dec 14, Diederik P. Kingma, Jimmy Ba https://arxiv.org/abs/1412.6980

https://arxiv.org/pdf/1712.01815.pdf

