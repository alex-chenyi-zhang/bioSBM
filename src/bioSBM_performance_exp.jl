using BenchmarkTools
using Random, Distributions, StatsBase, LinearAlgebra, DelimitedFiles, Optim, LineSearches, Flux
using JLD2  ## this is to store trained flux model to be loaded for future usage
Random.seed!(1)
n_iter = 20
n_runs = 2

covariate_file_names = 
covariate_file_names = "data/GM12878_input_files/X_files_GM12878.txt"
map_file_names = "data/GM12878_input_files/Y_files_GM12878.txt"

K = 10
R = 0.1


io = open(covariate_file_names, "r")
covariate_files = readdlm(io, String)
close(io)

io = open(map_file_names, "r")
contact_map_files = readdlm(io, String)
close(io)

n_regions = length(contact_map_files)
println(n_regions)

Ns = ones(Int, n_regions)
Ncum = 0

N_starts = ones(Int, n_regions)
N_ends = ones(Int, n_regions)
P = 0

for i_region in 1:n_regions
    global Ncum
    global P
    # println(contact_map_files[i_region], "\t" , covariate_files[i_region])
    io_x = open(covariate_files[i_region],"r")
    X_i = readdlm(io_x, Float64; header=true)[1]#[1:end-1,:]
    close(io_x)

    Ns[i_region] = length(X_i[1,:])
    N_starts[i_region] = Ncum + 1
    N_ends[i_region] = Ncum + Ns[i_region]
    Ncum += Ns[i_region]
    P = length(X_i[:,1])
end
println("Ns = ", Ns)
# println("Nsstart = ", N_starts)
# println("Nends = ", N_ends)
# println("Ntot: ", Ncum)

X = zeros(P, Ncum)
Y = [zeros(Ns[i_region], Ns[i_region]) for i_region in 1:n_regions]
#X[:,1:Ns[1]] .= X_i
#Y[1] .= Y_i
for i_region in 1:n_regions
    # println(contact_map_files[i_region], "\t" , covariate_files[i_region])
    io_x = open(covariate_files[i_region],"r")
    X_i = readdlm(io_x, Float64; header=true)[1]#[1:end-1,:]
    close(io_x)

    io_x = open(contact_map_files[i_region],"r")
    Y_i = readdlm(io_x, Float64)
    close(io_x)
    X[:,N_starts[i_region]:N_ends[i_region]] .= X_i
    Y[i_region] .= Y_i
end
println(Ns, "\t", P)


ϕ = [ones(Ns[i_region],Ns[i_region],K) for i_region in 1:n_regions]
for i_region in 1:n_regions
    for i in 1:Ns[i_region]
        for j in 1:Ns[i_region]
            ϕ[i_region][i,j,:] = rand(Dirichlet(K,0.5))
        end
    end
end

λ = randn(K, Ncum)
ν = [zeros(K,K,Ns[i_region]) for i_region in 1:n_regions]
for i_region in n_regions
    for i in 1:Ns[i_region]
        ν[i_region][:,:,i] = rand(Wishart(K,Matrix(.5I,K, K)))
    end
end

Σ = Matrix(0.7I, K, K)

like_var = zeros(K,K)
for k in 1:K
    for g  in 1:k
        like_var[k,g] = rand(InverseGamma(1,1))
        like_var[g,k] = like_var[k,g]
    end
end

B = zeros(K,K)
for k in 1:K
    B[k,k] = randn()*0.2+1.0
    for g in 1:k-1
        B[k,g] = randn()*0.2
        B[g,k] = B[k,g]
    end
end

# here we define the flux model that maps X into θ
#Γ   = Chain(Dense(P, 64, relu), Dense(64, 64, relu), Dense(64, 32, relu), Dense(32, K))
Γ   = Dense(P => K)
Γ = f64(Γ)
println("Linear model!!!")
ps  = Flux.params(Γ)
#opt = ADAM(0.01) # the value in brackts is the learnin rate for the optmizer

#Γ = zeros(K,P)
#for k in 1:K
#    Γ[k,:] .= randn(P)* sqrt(σ_2[k])
#end

#X = Float32.(X)

μ = zeros(K,Ncum)
#μ = Γ * X;
#μ = Γ(Float32.(X));
μ = Γ(X);
for i_region in 1:n_regions
    for i in 1:Ns[i_region]
        ϕ[i_region][i,i,:] .= 0
    end
end

μ = Γ(X);


inv_Σ = inv(Σ)

i_region = 1 

# In-place softmax function
function softmax!(x)
    max_x = maximum(x)
    @inbounds for i in eachindex(x)
        x[i] = exp(x[i] - max_x)
    end
    sum_x = sum(x)
    @inbounds for i in eachindex(x)
        x[i] /= sum_x
    end
end


function Estep_multinomial_gauss_opt!(ϕ::Array{Float64, 3}, λ, B::Array{Float64, 2},
    Y::Array{Float64, 2}, N::Int, K::Int, like_var::Array{Float64, 2})
    
    # Precompute reusable quantities
    inv_like_var = 0.5 ./ like_var;
    log_like_var = 0.5 .* log.(like_var);
    
    for i in 1:N
# #    for i in sample(1:N, div(N,4), replace=false)
        for j in 1:N
#     #    for j in sample(1:N, div(N,4), replace=false)
            if i != j
                for k in 1:K
                    logPi = λ[k,i]
#                     #ϕ[i, j, k] = λ[k,i]
                    for g in 1:K
                        logPi += -ϕ[j,i,g] *( ((Y[i,j] - B[k,g])^2) * inv_like_var[k,g]  + log_like_var[k,g])
                        #ϕ[i,j,k] += -ϕ[j,i,g] *( ((Y[i,j] - B[k,g])^2) * inv_like_var[k,g]  + log_like_var[k,g])
                    end
                    ϕ[i,j,k] = logPi
                end
                ϕ[i,j,:] .= softmax(ϕ[i,j,:])
                #softmax!(view(ϕ, i, j, :))
            end
        end
    end
end


@time Estep_multinomial_gauss_opt!(ϕ[i_region], @view(λ[:,N_starts[i_region]:N_ends[i_region]]), B, Y[i_region], Ns[i_region], K, like_var)

#@time Estep_multinomial_gauss!(ϕ[i_region], @view(λ[:,N_starts[i_region]:N_ends[i_region]]), B, Y[i_region], Ns[i_region], K, like_var)



function Mstep_blockmodel_gauss_multi!(ϕ::Vector{Array{Float64, 3}}, B::Array{Float64, 2}, like_var::Array{Float64, 2},
    Y::Vector{Array{Float64, 2}}, Ns::Array{Int,1}, K::Int, n_regions::Int)
    lv = 0.
    learn_r = 0.1
    cum_den = 0.
    for k in 1:K
        for g in 1:k
            num_gauss = 0.
            num = 0.
            den = 0.
            for i_region in 1:n_regions
                @inbounds for j in 1:Ns[i_region]
                    @inbounds for i in 1:Ns[i_region]
                # for j in sample(1:Ns[i_region], div(Ns[i_region], 4))
                #     for i in sample(1:Ns[i_region], div(Ns[i_region], 4))
                        phi_prod = ϕ[i_region][i,j,k]*ϕ[i_region][j,i,g]
                        num += phi_prod*Y[i_region][i,j]
                        den += phi_prod
                        num_gauss += phi_prod * (Y[i_region][i,j] - B[k,g])^2
                        #lv  += phi_prod * (Y[i,j] - B[k,g])^2
                    end
                end
            end
            B[k,g] =  (1-learn_r)*B[k,g] + learn_r*num/(den)
            #cum_den += den
            like_var[k,g] =  (1-learn_r)*like_var[k,g] + learn_r*num_gauss/(den)
            B[g,k] = B[k,g]
            like_var[g,k] = like_var[k,g]
        end
    end

    #like_var[1] = lv/cum_den

end


@btime Mstep_blockmodel_gauss_multi!(ϕ, B, like_var, Y, Ns, K, n_regions)


