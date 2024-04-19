This file is based on masther thesis titled "Optimization Based Control Of Cascaded Electrical Circuits", It primarily contains three 
control method, they are Flatness Based Control(FBC), Optimization Based Control & Model Predictive Control(MPC).

Link to thesis report: https://forschung.rwu.de/sites/forschung/files/2024-01/aman_dubey_master_thesis.pdf

File Description: 

1. FBC_Nvariation.jl: This file has implementation for flatness Based Control, You need to vary the parameter N which indicates 
   number of cascaded Electrical circuits. 

2. OBC_Higherorder_circuit_optimization.jl: It contains implementation of optimization based control with various optimization
   algorithm running concurrently one after the other, firtly the global optimization then local optimization. You can use the
   other optimization techniques as well which are available on Julia.
   Eg of Optimization technique:- Particleswarm, Polyopt(), Gradient Descent(), Quadratic approximation.
   
3. MPC_Optimization.jl: This file contains implementation of  Model Predictive Control(MPC), it contains parameter such as
   control horizon, prediction horizon and N(order of circuit). You can also use different optimization stratergy to check it'S
   affect on MPC. This file also contains some code for drawing graph using cairoMakie.

There are some more files which contains other research related implementation, if user want it please ask directly.
        
