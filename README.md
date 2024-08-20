# swarmCBO
MATLAB implementation of Consensus Based Optimization algorithms 

The repository has been created to host MATLAB code \texttt{swarmCBO} for a flexible implementation of Consensus-Based Optimization (CBO) algorithms. The script uses the same syntax as the popular  \texttt{particleswarm}, implementation of Particle Swarm Optimization (PSO), from MATLAB's Global Optimization Toolbox, allowing both algorithms to be used interchangeably.
The algorithm corresponds to an asymptotic preserving discretization of the Stochastic Differential PSO dynamics which allows to recover both the classical PSO and the first-order CBO algorithms by accurately changing the parameters.
The repository includes illustrative examples where  \texttt{swarmCBO} is compared against \texttt{particleswarm} on benchmark optimization problems and on the training of a high-dimensional logistic regression model. 
