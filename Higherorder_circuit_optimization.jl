using LinearAlgebra 
using OrdinaryDiffEq, DifferentialEquations 
using Plots

#=Input Function U, Which needs to be tailored in order to get the desirable output=#
function input_parametric(x)
    return (1 + tanh(x))/2
end  

#= Parameterizing the input signal with two parameter variation =#
function input_tanh(t,T,p)

    p_1 = p[1] # here we need to check whether p1 satisfies  p1>q, how can we do that ????
    p_2 = p[2]
    q = 1e-2;
    p_3 = -p_2 / atanh(q/(p_1)-1) # 
    # create one function   

    return p_1*(input_parametric(p_2*((t/T)-(1/p_3)))) # Function with three different parameters, which needs to be optimized 
end

function input_tanh_new(t,T,p)
    p_1= p[1]
    p_2= p[2]
    q = 1e-2;
    p_3= (-1/p_2)*atanh(((2*q)/p_1)-1)

    return p_1*(input_parametric(p_2*((t/T)-p_3)))
end

#=
tin= 0.5:0.01:10
R = zeros(length(tin))
  # creating input signal 

for (l,t) in enumerate(tin) #input signal creation 
    R[l] = tanh(t)
end

plot(tin,R)

for (l,t) in enumerate(tin)
    print("  l value is=", l, "\n  t value is =",t)
end

plot(tin,R)

plot(tspan, input_tanh_new(tspan,T,param))
=#
N = 20; # Set the number of cascaded circuit you want 
M = diagm(ones(Int, N)) # matrix with all diagonal element as 1, square matrix with size N by N 
D = diagm(ones(Int, N)) # matrix with all diagonal element as 1, square matrix with size N by N 
S = diagm(ones(Int, N)) # matrix with all diagonal element as 1, square matrix with size N by N 
for i = 1 : N, j = i+1 : N
    M[i,j] = 1  # above the diagonal element value 1 
    D[i,j] = 1  # above the diagonal element value 1
end

for i = 1 : N-1
    S[i+1,i] = -1
end

R = 1
L = 1
C = 1

A = zeros(2N,2N) # 2N by 2N matrix 
A[1:N, N+1:2N] = diagm(ones(Int,N)) # the top-right side of the matrix is Identity matrix 
A[N+1:2N, 1:N] = (-inv(M)*S) / (L*C)  # lower left side we have S~ matrix  
A[N+1:2N, N+1:2N] = (-inv(M)*D) * (R/L) # lower right side we have D~ matrix 


#Checking for Stiffness in the equation 
eiv = eigvals(A) # Eigen values of A 
scatter(eiv) # to check how they are spreaded over 
stiffness_id = maximum(abs.(eiv))/minimum(abs.(eiv)) # This tells you that how much the system is stiffed. 
maximum(abs.(eiv))
minimum(abs.(eiv))

G = zeros(N)
G[1] = 1
# B = vcat(zeros(N), G)
B = vcat(zeros(N), 1/(L*C), zeros(N-1)) # first N element zero then 1/LC, after that N-1 values are zero 

# C = vcat(zeros(N-1),1,zeros(N))'


# My ode function representing the cascaded RLC circuit in state space 
function my_ode(dx, x, p, t)
    u_in = input_tanh_new(t,T,p) # sin(t)

    dx[1:N] = x[N+1:2N]
    dx[N+1:2N] = (-inv(M)*S) / (L*C) * x[1:N] + (-inv(M)*D) * (R/L) * x[N+1:2N] + G/(L*C) * u_in
    # dx .= A*x + B*u
end


# Solve the equations 
  
const x0 = zeros(2N) 
const T = 5000.0; 
tspan = (0.0, T)
param = [  1.0,  1296.7500000000002] # rand(4)
alg = Tsit5()
prob = ODEProblem(my_ode, x0, tspan,param); # 
sol = solve(prob, Tsit5(), saveat=0.1)


plot(sol)
plot(sol.t,sol[1:N,:]')
plot(sol.t,sol[N,:])


function loss(param) # defining the loss function here 
    sol = solve(prob, Tsit5(), p = param) # what is p here :- p is parameter array  
    err = sol[N,:] .- 1  # considering reference to be 1 
    loss_total = sum(abs2, err)/length(err) # module2 operator on original output- Expected output sol[2] represents X2 
    return loss_total#, sol  # return loss value along with solution 
end

function cb(pars1, loss1)
    #callback = function (pars1, loss1) # why this function is required ??
       display(loss1)
           # Tell Optimization.solve to not halt the optimization. If return true, then
       # optimization stops.
       return false
    
end

#callback function is required to tell the sytem when to stop
function cb_new(pars1, loss1)
    #callback = function (pars1, loss1) # why this function is required ??
       display(loss1)
           # Tell Optimization.solve to not halt the optimization. If return true, then
       # optimization stops.
        if loss1 < 0.5 #50 percent  
            return true
        else
            return false
        end
end

#callback function with loss dropping below 95 percent 
function cb_drop95percent(pars1,loss1)
    #callback = function (pars1, loss1) # why this function is required ??
        display(loss1)
        # Tell Optimization.solve to not halt the optimization. If return true, then
        # optimization stops.
     if loss1 < 0.05
         return true
     else
         return false
     end
end     

function cb_drop99percent(pars1,loss1)
    #callback = function (pars1, loss1) # why this function is required ??
        display(loss1)
        # Tell Optimization.solve to not halt the optimization. If return true, then
        # optimization stops.
     if loss1 < 0.01
         return true
     else
         return false
     end
end  
## Libraries required ##
using Optimization, OptimizationPolyalgorithms, SciMLSensitivity, Zygote, OptimizationOptimJL,LineSearches, OptimizationNLopt  


adtype = Optimization.AutoFiniteDiff()  # adtype allows us to choose the type of Automatic differentiation we use, More research is needed in this area so as to come up with an answer on why we choose a particluad adtype 

optf = Optimization.OptimizationFunction((x, param) -> loss(x),adtype) # x is variable , param is initial value of the parameters

optprob = Optimization.OptimizationProblem(optf, param,lb=[0.1,0.0] , ub=[100.0,50.0] ) #Final problem defination 

opt_sol_Para_temp= Optimization.solve(optprob, Optim.ParticleSwarm(lower = optprob.lb, upper = optprob.ub, n_particles = 200), callback = cb_new, maxiters =4 )
# opt_sol_Para_temp is initial optimized parameters obtained, they would be fed to local optimization technique in order to reach optimized solution 
param_new=[ 1.0188661066056204,  61.593282512921746 ]

new_adtype = Optimization.AutoFiniteDiff()

optf_new = Optimization.OptimizationFunction((x, param_new ) -> loss(x),new_adtype) #now using obtained parameters as input to the system 

#optprob_new = Optimization.OptimizationProblem(optf_new, param_new,lb=[0.1,0.0] , ub=[100.0,10.0] )

optprob_new = Optimization.OptimizationProblem(optf_new, param_new )
#new_optparameters= Optimization.solve(optprob_new, NLopt.LD_LBFGS(), callback=cb_drop95percent ,maxiters = 5);

#new_optparameters= Optimization.solve(optprob_new, Optim.ParticleSwarm(lower = optprob_new.lb, upper = optprob_new.ub, n_particles = 200), callback = cb, maxiters = 50)

new_optparameters= Optimization.solve(optprob_new, PolyOpt(), callback = cb, maxiters = 4)

#par_2= [new_optparameters[1], new_optparameters[2]]
par_2= [1.0188661066056204, 61.593282512921746]
optf_new_2= Optimization.OptimizationFunction((x, par_2 ) -> loss(x),new_adtype)

optprob_new_2 = Optimization.OptimizationProblem(optf_new_2, par_2)

new_optparameters_2= Optimization.solve(optprob_new_2, Optim.GradientDescent(), callback=cb_drop95percent ,maxiters = 4);


par_3= [1.0,   1300.0]
#par_3= [new_optparameters_2[1], new_optparameters_2[2]]
optf_new_3= Optimization.OptimizationFunction((x, par_3 ) -> loss(x),new_adtype)

optprob_new_3 = Optimization.OptimizationProblem(optf_new_3, par_3,lb=[0.1,0.0] , ub=[100.0,1300.0])

new_optparameters_3= Optimization.solve(optprob_new_3,NLopt.LN_BOBYQA(), callback=cb_drop99percent ,maxiters = 50);


#par_4=[new_optparameters_3[1], new_optparameters_3[2]]

par_4=[ 1.0, 1300.0]

optf_new_4= Optimization.OptimizationFunction((x, par_4 ) -> loss(x),new_adtype)

optprob_new_4 = Optimization.OptimizationProblem(optf_new_4, par_4,lb=[0.1,0.0] , ub=[100.0,1300.0])
new_optparameters_4= Optimization.solve(optprob_new_4,NLopt.LN_NELDERMEAD(), callback=cb_drop99percent ,maxiters = 50);

# NLopt.LD_CCSAQ(), LN_AUGLAG_EQ():  not converging when we provide the value close to the solution 
# NLopt.LN_NELDERMEAD() and NLopt.LN_BOBYQA() , Both of them comverge pretty well in case of initial value being close to the actual solution 
# But with same inital value   NLopt.LN_BOBYQA() outperforms other methods
#  
using Plots
#t=0:.1:5000
t = range(1.0, 5000, length=5000)


function tanh_new(t)
    p_1=1.0162703379007416
    p_2=21.17498782264494
    q = 1e-2;
    p_3 = -p_2 / atanh(q/(p_1)-1) # 
    # create one function   

    return (p_1/2)*(1+tanh(p_2*((t/5000)-p_3)))
end

plot(t, tanh_new.(t))