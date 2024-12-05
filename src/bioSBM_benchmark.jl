include("bioSBM_fns.jl")
using BenchmarkTools

n_iter = 20
n_runs = 2
covariate_file_names = "data/GM12878_input_files/X_files_GM12878_chr1_100k.txt"
map_file_names = "data/GM12878_input_files/Y_files_GM12878_chr1_100k.txt"

K = 12
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



#Σ = rand(Wishart(K,Matrix(.5I,K, K)))
#σ_2 = rand(InverseGamma(1,1), K)
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
#println(μ)
###### end of initialization


μ = Γ(X);


inv_Σ = inv(Σ)

i_region = 1 

# @time Estep_logitNorm!(ϕ[i_region], @view(λ[:,N_starts[i_region]:N_ends[i_region]]), ν[i_region], inv_Σ, Float64.(μ[:,N_starts[i_region]:N_ends[i_region]]), Ns[i_region], K)
# @time Estep_multinomial_gauss!(ϕ[i_region], @view(λ[:,N_starts[i_region]:N_ends[i_region]]), B, Y[i_region], Ns[i_region], K, like_var)
# @time Mstep_blockmodel_gauss_multi!(ϕ, B, like_var, Y, Ns, K, n_regions)
# @time ELBO_gauss(ϕ[i_region], λ[:,N_starts[i_region]:N_ends[i_region]], ν[i_region], Σ, B, μ[:,N_starts[i_region]:N_ends[i_region]],  Y[i_region], K, Ns[i_region], like_var)

# println("\n")

# @btime Estep_logitNorm!(ϕ[i_region], @view(λ[:,N_starts[i_region]:N_ends[i_region]]), ν[i_region], inv_Σ, Float64.(μ[:,N_starts[i_region]:N_ends[i_region]]), Ns[i_region], K)
# @btime Estep_multinomial_gauss!(ϕ[i_region], @view(λ[:,N_starts[i_region]:N_ends[i_region]]), B, Y[i_region], Ns[i_region], K, like_var)
# @btime Mstep_blockmodel_gauss_multi!(ϕ, B, like_var, Y, Ns, K, n_regions)
# @btime ELBO_gauss(ϕ[i_region], λ[:,N_starts[i_region]:N_ends[i_region]], ν[i_region], Σ, B, μ[:,N_starts[i_region]:N_ends[i_region]],  Y[i_region], K, Ns[i_region], like_var)






println("\n")
println("\n")
@time run_VEM_gauss_NN!(n_iter, ϕ, λ, ν, Σ, B, like_var, μ, Y, X, Γ, ps, K, Ns, P, n_regions, R)
@time run_VEM_gauss_NN!(n_iter, ϕ, λ, ν, Σ, B, like_var, μ, Y, X, Γ, ps, K, Ns, P, n_regions, R)