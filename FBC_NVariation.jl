
using LinearAlgebra

steep_id =0.943119912147522 ; 
N = 8;
Tf = 100;
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

R1 = 1 
L1 = 1
C1 = 1 

# Now calculating the values of A,B and C, For state space representation 
A = zeros(2N,2N) # 2N by 2N matrix 
A[1:N, N+1:2N] = diagm(ones(Int,N)) # the top-right side of the matrix is Identity matrix 
A[N+1:2N, 1:N] = (-inv(M)*S) / (L1*C1)  # lower left side we have S~ matrix  
A[N+1:2N, N+1:2N] = (-inv(M)*D) * (R1/L1) # lower right side we have D~ matrix 

B = vcat(zeros(N), 1/(L1*C1), zeros(N-1)) # first N element zero then 1/LC, after that N-1 values are zero 

C = vcat(zeros(N-1),1,zeros(N))'

ngrid = 1:2N
T = vcat(map(n-> C*A^(n-1), ngrid)...)

#=
    Eulerian number / Eulerian triangle
    see also: https://en.wikipedia.org/wiki/Eulerian_number
=#
function euler_dir(n::Int64, k::Int64)
    igrid = 0 : 1 : k
    a_int(i) = (-1)^(i) * big(binomial(n+1,i))*(big(k+1-i))^n
    return mapreduce( i -> a_int(i), +, igrid)
end


#=
Compute n-th order derivative of tanh(a*t)
- time t
- constant parameter a
- order of differentiation n
=#
function der_tanh(t::Float64,a::Float64,n::Int64)

    c1 = 2^(n+1) * exp(big(2t)) / (1 + exp(big(2t)))^(n+1)

    kgrid = 0:1:n-1
    c2 = mapreduce(k-> (-1)^k * euler_dir(n,k)*exp(big(2*k*t)),+,kgrid)

    return a^n * c1*c2
end 

function ref(t)
    
    rmat = zeros(2N+1)
    rmat[1] = (tanh(steep_id*(t-Tf/2))+1)/2 # input function y 
    for i = 1:2N
        rmat[i+1]= der_tanh(t-Tf/2,steep_id,i)/2 # derivatives of y 
    end    

    

    return rmat 
end

tgrid = 0: 0.1 : Tf
using Plots
ref_data = hcat(ref.(tgrid)...)
plot(tgrid,ref_data')



# plotting via cairomakie
Der_1= vec(ref_data[1,:])
Der_2= vec(ref_data[2,:])
Der_3= vec(ref_data[3,:])
f = Figure()
Axis(f[1, 1],titlealign=:center,title="Tanh and it's derivatives")
lines(tgrid,Der_1,color=:Red,linewidth = 1,label="tanh")
lines!(tgrid,Der_2,color=:blue,linewidth = 1,label="First derivative")
lines!(tgrid,Der_3,color=:Yellow,linewidth = 1,label="Second Derivative")
axislegend()
current_figure()
using CairoMakie
path_tanh="C:\\Users\\student\\Documents\\RESULTS\\Optimization based control\\tanh_function"
pathoffile = path_tanh * "\\tanh  .pdf"

save(pathoffile,current_figure())

#

# Tanh and it's parameter variations p1,p2 and p3
function tanh_parameter(t,p_1,p_2,p_3)
    return (p_1/2)*(1+tanh(p_2*((t))-(p_3)))
end    

tp=0:0.1:100

Ar=zeros(length(tp))
Br=zeros(length(tp))

for (i,j) in enumerate(tp)
    Ar[i]=tanh_parameter(j,1,1,10)
end

for (i,j) in enumerate(tp)
    Br[i]=tanh_parameter(j,5,1,10)
end

Par_1= vec(Ar)
Par_2= vec(Br)

f_p = Figure()
Axis(f_p[1, 1],titlealign=:center,title="Parameter P_1 variation")
lines(tgrid,Par_1,color=:Red,linewidth = 1,label="p_1 = 1")
lines!(tgrid,Par_2,color=:blue,linewidth = 1,label="p_1 = 5")

axislegend()
current_figure()

path_tanh="C:\\Users\\student\\Documents\\RESULTS\\Optimization based control\\tanh_function"
pathoffile = path_tanh * "\\Varying P_1 1 to 5.pdf"
save(pathoffile,current_figure())


#



path="C:\\Users\\student\\Documents\\output_optimization_openloopsystem"



pd=plot(tgrid,ref_data[1,:]+ref_data[2,:]+ref_data[3,:]+ref_data[4,:]+ref_data[5,:]+ref_data[6,:]+ref_data[7,:]+ref_data[8,:])
    # savefig(joinpath(path,"a=" * string(sc)*"N="*string(N)*".jpg"))
savefig(pd, joinpath(path, string("tanh plus it's derivative upto 7 ",".pdf")))  


plot(tgrid,ref_data[1,:]+ref_data[2,:]+ref_data[3,:]+ref_data[4,:]+ref_data[5,:]+ref_data[6,:]+ref_data[7,:]+ref_data[8,:])
plot(tgrid,ref_data[1,:]+ref_data[2,:])

#section begin
# Implementing tanh and adding it's derivative 

ty=0:0.1:100
S=zeros(length(ty))
for (i,j) in enumerate(ty)
    S[i]=(1+tanh((j)-10))/2
end
plot(ty, S)

#Now it's derivative 

K=zeros(length(ty))
for (i,j) in enumerate(ty)
    K[i]=0.5*der_tanh(j-10,steep_id,1)
end

L=S+K
plot(ty, L)

# Implementation End 
#section end 


function myinput(t)

    c1 = (C*A^(2*N-1)*B)[1]
    v1 = inv(c1) * hcat(-C*A^(2*N) * inv(T), 1)
    v2 = ref(t)[1:2N+1]

    return (v1*v2)[1] # this is our U 
end


function myode(dx, x, p,t) # Here we need to solve the function with provided input(via flatness based control)

    u = myinput(t)
    dx .= A*x + B*u  
end

# Now lets solve the equation by using obtained u as input 
using OrdinaryDiffEq


x0 = zeros(2*N)# initial value 
tspan = (0.0, Tf)
alg = Tsit5()  # algorithm 

prob = ODEProblem(myode, x0, tspan)
sol = solve(prob, alg, saveat=0.1)

using Plots
plot(sol)                     #All solution plot
plot(sol.t, sol[N,:])         #Obtained/Desired output plot   
plot(sol.t, myinput.(sol.t))  


R = zeros(length(sol.t))      # creating input signal 

for (l,t) in enumerate(sol.t) #input signal creation 
    R[l] = (tanh(steep_id*(t-Tf/2))+1)/2
end

plot(sol.t,[sol[N,:],R])

using CairoMakie
path="C:\\Users\\student\\Documents\\RESULTS\\FBC\\N=6_vari_a"



S=vec(sol[N,:])
V=vec(R)




f = Figure()
Axis(f[1, 1],titlealign=:center,title="FBC for N = 3",xlabel="time",ylabel="Amplitude")
#=
for i=1:2

    lines(sol.t,S,color=:blue,label="Output")
    lines(sol.t,V,color=:red,label="Reference" )
    
end
=#

lines(sol.t,S,color=:blue, label="Output")
lines!(sol.t,V,color=:red,label="Reference" )
axislegend()
current_figure()




#f

path2file = path * "\\FBC with N=6 a=0.7  .pdf"

save(path2file,current_figure())


#Below is the method to zoom a particular area in a graph using cairomakie 
sim_t = sol.t
S=vec(sol[N,:])
V=vec(R)

using CairoMakie
begin
  fig1 = Figure(fontsize=20)
  ax1 = Axis(fig1[1, 1], xlabel ="time", ylabel = "Amplitude", ylabelsize = 24,
      xlabelsize = 24, xgridstyle = :dash, ygridstyle = :dash, 
      xtickalign = 1., xticksize = 10, 
      xminorgridvisible = true, xminorticksvisible = true, xminortickalign = 1,
      yminorgridvisible = true, yminorticksvisible = true, yminortickalign = 1,
      ytickalign = 1, yticksize = 10, xlabelpadding = 0)
  
  ax1.xticks = 0 : 10 : sim_t[end];    
  #ax1.yticks = -1 : 0.2 : 1;
  lines!(sim_t, S; linewidth = 2, label = "Output")
  lines!(sim_t, R; linewidth = 2, label = "Reference")
  axislegend(; position = :lt, bgcolor = (:grey90, 0.1));

    ax2 = Axis(fig1, bbox=BBox(600, 750, 380, 508), ylabelsize = 24)
  ax2.xticks = 60 : 1 : 63;
  #ax2.yticks = -0.01 : 0.002 : 0.01;
  lines!(ax2, sim_t[600:630],S[600:630]; linewidth = 1, color=Makie.wong_colors()[1])
  lines!(ax2, sim_t[600:630],R[600:630]; linewidth = 1, color=Makie.wong_colors()[2])
  CairoMakie.translate!(ax2.scene, 0, 0, 10);
  
  fig1
  #save("results/figures/"*"sin_cos.pdf", fig1, pt_per_unit = 1)    
end

path="C:\\Users\\student\\Documents\\RESULTS\\FBC\\a=1_vari_N"
fig1
path2file = path * "\\zoom FBC with N=15 a=1.0  .pdf"

save(path2file,fig1)


