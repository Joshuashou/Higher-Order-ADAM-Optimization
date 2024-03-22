File Descriptions

main.py - Runs experiments for the RESNET with various Optimizers. 

Adamax.py - Contains code for ADAMAX Optimizer estimations. 

Vanilla_Adam.py defines the original ADAM optimizer.

Vanilla_Adam.py is our edited files with various levels of degree estimations of gradients. 

Higher_Order_Adam.ipynb was initial notebook adapted from my Deep Learning Project, which I am refactoring and editing to make the python project be organized better and more clearly.

Resnet.py constructs the residual neural neural network with different bottleneck and block layers. 


## RESNET Result. 

Resnet 18 block. 

Training and Testing Results with Higher Order ADAM, Vanilla ADAM, and ADAMAX
<p>
  <img width="400" alt="Train Accuracies" src="https://github.com/Joshuashou/Higher-Order-ADAM-Optimization/blob/master/Simulation_Results/Train_Accuracies.png" style="display: inline-block;">
  <img width="400" alt="Test Accuracies" src="https://github.com/Joshuashou/Higher-Order-ADAM-Optimization/blob/master/Simulation_Results/Test_Accuracies.png" style="display: inline-block;">
</p>


Sources:

Adam: A Method for Stochastic Optimization, 22 Dec 14, Diederik P. Kingma, Jimmy Ba https://arxiv.org/abs/1412.6980

https://arxiv.org/pdf/1712.01815.pdf

