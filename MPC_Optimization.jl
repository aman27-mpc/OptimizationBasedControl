using LinearAlgebra 
using OrdinaryDiffEq, DifferentialEquations 
using Plots

#=Input Function U, Which needs to be tailored in order to get the desirable output=#
function input_parametric(x)
    return (1 + tanh(x))/2
end  

function normal_tanh(t)
    return tanh(t)
end  

#= Parameterizing the input signal with two parameter variation =#
function input_tanh(t,T,p)

    p_1 = p[1] # here we need to check whether p1 satisfies  p1>q, how can we do that ????
    q = 1e-2;
    p_3 = -10 / atanh(q/(p_1)-1) # p_2 is fixed to 10 
    # create one function   

    return p_1*(input_parametric(10*((t/T)-(1/p_3)))) # Function with three different parameters, which needs to be optimized 
end

function input_tanh_new(t,T,p)
    p_1= p[1]
    
    q = 1e-2;
    p_3= (-1/10)*atanh(((2*q)/p_1)-1) # p_2 is fixed to 10 

    return p_1*(input_parametric(10*((t/T)-p_3)))
end

function input_tanh_tdfix(t,T,p) # here the p_3 value is fixed based on the delay time
    p_1= p[1]
    p_2= p[2]
    
    p_3= p[3]  

    return p_1*(input_parametric((p_2*(t/T))-p_3))
    #return p_1*(input_parametric((p_2*(t/T))-5))
end

function mpc_signal(t,T,contr_horizon,param)
    p_1=param[1]
    s=0.0
    
    for(i,c) in enumerate(contr_horizon)
        s += input_tanh_tdfix(t-c*i,T,param)
    end
    
    return s
end    

function mpc_signal_new(t,T,contr_horizon,param)
    p_1=param
    s=0.0
    
    for(i,c) in enumerate(contr_horizon)
        s += input_tanh_tdfix(t,T,param)
    end
    
    return s
end 


N = 8; # Set the number of cascaded circuit you want 
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
B = vcat(zeros(N), 1/(L*C), zeros(N-1)) # first N element zero then 1/LC, after that N-1 values are zero 


Nmpc=5
dt_mpc= 0.1 # sampling time 
contr_horizon = 1
pred_horizon = 5
init_value = rand(contr_horizon)


function my_ode(dx, x, p, t)
    u_in = mpc_signal(t,T,contr_horizon,p) # sin(t)

    dx[1:N] = x[N+1:2N]
    dx[N+1:2N] = (-inv(M)*S) / (L*C) * x[1:N] + (-inv(M)*D) * (R/L) * x[N+1:2N] + G/(L*C) * u_in
    # dx .= A*x + B*u
end

function my_ode_new(dx, x, p, t)
    u_in = mpc_signal_new(t,T,contr_horizon,p) # sin(t)

    dx[1:N] = x[N+1:2N]
    dx[N+1:2N] = (-inv(M)*S) / (L*C) * x[1:N] + (-inv(M)*D) * (R/L) * x[N+1:2N] + G/(L*C) * u_in
    # dx .= A*x + B*u
end

const x0 = zeros(2N) # initial value
param=[1.0,1.0,5.0]



T=10

prob_mpc = ODEProblem(my_ode_new, x0, (0, 600),param)
sol = solve(prob_mpc, alg, p = param, saveat=dt_mpc)

reference = zeros(length(sol.t))       

for (l,t) in enumerate(sol.t) #input signal creation 
    reference[l] = input_tanh_tdfix(t,10,[1.0,1.0,5.0])
end  


using Plots 
#plot(sol.t,sol[N,:])
#plot(sol.t,reference)
plot(sol.t,[sol[N,:],reference])

function loss_tanh(par) # defining the loss function here 
    sol = solve(prob_mpc, alg, p = par, saveat=dt_mpc) # what is p here :- p is parameter array  
    if sol.retcode != ReturnCode.Success
        return Inf
    end 
    st=10*sol.t[begin]
    st=st+1
    en=10*sol.t[end]
    #println("start = ", st,"End =", en)
    err = sol[N,2:end] - reference[Int(st):Int(en)]  # considering reference to be 1 
    loss_total = sum(abs2, err) # module2 operator on original output- Expected output sol[2] represents X2 
    return loss_total#, sol  # return loss value along with solution 
end

function cb_new(pars1, loss1)
    #callback = function (pars1, loss1) # why this function is required ??
       display(loss1)
           # Tell Optimization.solve to not halt the optimization. If return true, then
       # optimization stops.
        if loss1 < 0.1 #50 percent  
            return true
        else
            return false
        end
end


using Optimization, OptimizationOptimJL, SciMLSensitivity, ForwardDiff ,OptimizationNOMAD
using OptimizationPolyalgorithms,OptimizationNLopt,Sundials  



## the prediction horizon would be like ten times the control horizon ###


t_ch= 2
t_ph= 10
Nmpc=5
#T=t_ph

Nsteps=150
tspan=(0,50) # this is our prediction horizon interval
#alg = Tsit5()
alg = Vern9()
param=[1.0,1.0,5.0]
#adtype = Optimization.AutoZygote()
adtype = Optimization.AutoForwardDiff()
next_param=copy(param)
optf = Optimization.OptimizationFunction((x, next_param ) -> loss_tanh(x),adtype)

sol_ar=zeros(Nsteps,(t_ch*10)+1)
solArray = Array{ODESolution}(undef,Nsteps)
paramArray = zeros(length(next_param), Nsteps+1);

prob_mpc = ODEProblem(my_ode_new, x0, (0,t_ph),param)


#sol = solve(prob_mpc, alg, p = param, saveat=dt_mpc)
#optprob = Optimization.OptimizationProblem(optf, next_param); # define thr problem 
#new_optparameter = Optimization.solve(optprob, PolyOpt(), maxiters = 100);

for i =1:Nsteps
    #optprob = Optimization.OptimizationProblem(optf, next_param,lb=[0.0,0.0,0.0],ub=[100.0,300.0,100.0]); # define the problem 
    optprob = Optimization.OptimizationProblem(optf, next_param); 
    
    new_optparameter = Optimization.solve(optprob,PolyOpt(),callback=cb_new, maxiters = 30); # Now optimize the problem with the param 
    # Now we have opt_res as the new opt parameter 
    #sol = solve(prob_mpc, alg, p = 1.0, saveat=dt_mpc)
    #println("start = ", 10*sol.t[begin] +1,"End =", 10*sol.t[end])
    println(new_optparameter)
    paramArray[:,i+1] = new_optparameter.u 
    next_param= new_optparameter.u 

    #prob1 = remake(prob_mpc; tspan=(0+(i-1), t_ch+(i-1)))
    prob1 = remake(prob_mpc; tspan=(0+t_ch*(i-1), t_ch*i))

    #solArray[i] = solve(prob1,alg, p=[new_optparameter.u[1]], saveat=0.1*t_ch)
    solution = solve(prob1,alg, p=[new_optparameter.u[1],new_optparameter.u[2],new_optparameter.u[3]],alg_hints=[:stiff], saveat=0.1)
    sol_ar[i,:]=solution[N,:]
    println(sol_ar[i,:])

    #prob_mpc = remake(prob_mpc; u0= (solArray[i])[:,end],tspan =(0+i,t_ph+i))
    #prob_mpc = remake(prob_mpc; u0=solution[:,end] ,tspan =(0+i,t_ph+i))
    prob_mpc = remake(prob_mpc; u0=solution[:,end] ,tspan =(0+i*t_ch,t_ph+(i*t_ch)))
end    


# now what we have obtained after optimization ??

########rough work#########
#######start#######
prob_new = ODEProblem(my_ode_new, x0, (450,460),param)

sol_new = solve(prob_new, alg, p = param[1], saveat=dt_mpc)
plot(sol_new)

prob_mpc = ODEProblem(my_ode_new, x0, (0,t_ch),param)
for i=1:100
    solution = solve(prob_mpc,alg, p = param[1], saveat=0.1*t_ch)
    sol_ar[i,:]=solution[N,:]
    prob_mpc = remake(prob_mpc; u0=solution[:,end] ,tspan =(0+i,t_ch+i))
end


#######end#######


# plot the solarray 
trajectory = zeros(length(sol_ar),1)



k=0
for i=1:Nsteps
    for j=1:((t_ch*10)+1)
        k=k+1
        trajectory[k]=sol_ar[i,j]
    end    
end


#tgrid=0.1:0.1:Nsteps*51
tgrid=0.1:0.1:(length(sol_ar)/10)

plot(tgrid,trajectory)
plot(tgrid,reference[1:length(sol_ar)])
plot(tgrid,[trajectory,reference[1:length(sol_ar)]])


path="C:\\Users\\student\\Documents\\RESULTS\\MPC\\Control_Horizon\\ConHorPlot"

using CairoMakie


S_t=vec(trajectory)
V=vec(reference[1:length(sol_ar)])
S_l=vec(sol[N,1:length(sol_ar)])
tim=vec(sol.t[1:length(sol_ar)])

# Storing the data 
using DelimitedFiles

open("MPC_Output_N_8_ContH_5.txt", "w") do io
    writedlm(io, ["time" "Original op" "Reference" "Optimized op MPC"],';')
    writedlm(io, [tim S_l V S_t],';')
    #writedlm(io, [paramArray[1,201] paramArray[2,201] paramArray[3,201] loss_tanh(New_Parameters_GD)],';')
end;

#Store the parameters and the loss value, it might be helpful to draw the graphs from it.

parameter_1=zeros(Nsteps) #length(paramArray)/3.0 =201
parameter_2=zeros(Nsteps)
parameter_3=zeros(Nsteps)
ind= ones(Nsteps)
loss_val= zeros(Nsteps)

for i=1:Nsteps
    parameter_1[i]=paramArray[1,i+1]
    parameter_2[i]=paramArray[2,i+1]
    parameter_3[i]=paramArray[3,i+1]
    loss_val[i] = loss_tanh(paramArray[:,i+1])
    ind[i] = i
end


open("MPC_Param_Loss_ContH_5.txt", "w") do io
    writedlm(io, ["time" "p_1" "p_2" "p_3" "loss_value"],';')
    writedlm(io, [ind parameter_1 parameter_2 parameter_3 loss_val],';')
end;


f = Figure()
Axis(f[1, 1],titlealign=:center,title="MPC for M=5 ",xlabel="Time",ylabel="Amplitude")

#=
for i=1:2

    lines!(tgrid,S,color=:blue,label="Output")
    lines!(tgrid,reference[1:length(sol_ar)],color=:red,label="Reference" )
    
end
=#
lines!(tgrid,S_l,color=:blue,label="Output without MPC")
lines!(tgrid,reference[1:length(sol_ar)],color=:red,label="Reference" )
lines!(tgrid,S_t,color=:green,label="Output with MPC")
axislegend()
current_figure()

#title!("Line Plot Example", position = :center)
#lines(tgrid,S,color=:blue,label="Output")
#title!("Line Plot Example", position = :center)
#lines!(tgrid,reference[1:length(sol_ar)],color=:red,label="Reference" )

f


path="C:\\Users\\student\\Documents\\RESULTS\\MPC\\Control_Horizon\\ConHorPlot"
path2file = path * "\\MPC with M=5.pdf"

save(path2file,f)




