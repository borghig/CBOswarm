# swarmCBO
MATLAB implementation of Consensus Based Optimization algorithms. 

The repository has been created to host MATLAB code _swarmCBO_ for a flexible implementation of Consensus-Based Optimization (CBO) algorithms. The script uses the same syntax as the popular _particleswarm_, implementation of Particle Swarm Optimization (PSO), from MATLAB's Global Optimization Toolbox, allowing both algorithms to be used interchangeably.
The algorithm corresponds to an asymptotic preserving discretization of the Stochastic Differential PSO dynamics which allows to recover both the classical PSO and the first-order CBO algorithms by accurately changing the parameters.
The repository includes illustrative examples where _swarmCBO_ is compared against _particleswarm_ on benchmark optimization problems and on the training of a high-dimensional logistic regression model. 

To use the algorithm _swarmCBO_, the script should be included in the _globaloptim_ MATLAB directory.

-------
_Project:_
_"Realizzazione di un pacchetto software per metodi di tipo consensus based optimization (CBO) e applicazioni all’apprendimento automatico"
Bando di selezione n.1/2024, Dip. di Matematica e Informatica, Università di Ferrara_

