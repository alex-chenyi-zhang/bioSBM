using Random, Distributions, StatsBase, LinearAlgebra, DelimitedFiles, Optim, LineSearches, Flux
using JLD2  ## this is to store trained flux model to be loaded for future usage
Random.seed!()

################################################################################

# This function computes the approximate ELBO of the model with the gaussian likelihood
function ELBO_gauss(ϕ::Array{Float64, 3}, λ, ν::Array{Float64, 3},
    Σ::Array{Float64, 2}, B::Array{Float64, 2}, μ, Y::Array{Float64, 2}, K::Int, N::Int, like_var::Array{Float64, 2})
    apprELBO = 0.0
    inv_Σ = inv(Σ)
    #inv_Σ = Matrix(1.0I, length(Σ[1,:]), length(Σ[1,:]))
    apprELBO = 0.5 * N * log(det(inv_Σ))
    for i in 1:N
        apprELBO -= 0.5 * (dot(λ[:,i] .- μ[:,i], inv_Σ, λ[:,i] .- μ[:,i]) + tr(inv_Σ*ν[:,:,i])) #first 2 terms
    end

    if isnan(apprELBO)
        println("ERROR 1")
        #break
    end

    for k in 1:K
        for j in 1:N
            for i in 1:N
                if i != j && ϕ[i,j,k] > eps()
                    apprELBO -= ϕ[i,j,k]*log(ϕ[i,j,k]) #last entropic term
                end
            end
        end
    end
    if isnan(apprELBO)
        println("ERROR 2")
        #break
    end

    for j in 1:N
        for i in 1:N
            if i != j
                apprELBO += dot(ϕ[i,j,:],λ[:,i])# vcat(λ[:,i], 1.0)) #third term
            end
        end
    end
    if isnan(apprELBO)
        println("ERROR 3")
        #break
    end

    for i in 1:N
        theta = zeros(K)
        #theta .= exp.(λ[:,i])  #exp.(vcat(λ[:,i], 1.0))
        #theta /= sum(theta)
        theta .= softmax(λ[:,i])
        # second line of the expression above
        apprELBO -= (N-1) * ( (log(sum(exp.(λ[:,i])))) + 0.5*tr((diagm(theta) .- theta * theta')*ν[:,:,i]) )
        #gaussian entropic term
        apprELBO += 0.5*log(det(ν[:,:,i]))
    end
    if isnan(apprELBO)
        println("ERROR 4")
        #break
    end
    #println("Partial ELBO: ", apprELBO)
    #likelihood term
    for i in 1:N
        for j in 1:i-1
            for k in 1:K
                for g in 1:K
                    #logP = -(0.5*(Y[i,j] - B[k,g])^2)/like_var[1] -0.5*log(like_var[1])
                    logP = -(0.5*(Y[i,j] - B[k,g])^2)/like_var[k,g] -0.5*log(like_var[k,g])
                    apprELBO += ϕ[i,j,k]*ϕ[j,i,g]*logP
                end
            end
        end
    end
    #apprELBO += -0.25*N*(N-1)*log(like_var[1])
    if isnan(apprELBO)
        println("ERROR 5")
        #break
    end
    return apprELBO
end


#########################################################################################
#########################################################################################
#########################################################################################

#=
In the E-step due to the non-conjugancy of the logistic normal with the multinomial
we resort to a gaussian approximation of the variational porsterior (Wang, Blei, 2013).
The approximations equals an optimizetion process that we characterize with the followfin functions
=#

function f(η_i::Array{Float64, 1}, ϕ_i::Array{Float64, 1}, inv_Σ::Array{Float64, 2}, μ_i, N::Int)
    #f = 0.5 * dot(η_i .- μ_i, inv_Σ, η_i .- μ_i) - dot(η_i, ϕ_i) +(N-1)*log(sum(exp.(η_i)))
    f = 0.5 * dot(η_i .- μ_i, inv_Σ, η_i .- μ_i) - dot(η_i, ϕ_i) +(N-1)*log(sum(exp.(η_i)))

    #MMM = Matrix(1.0I, length(inv_Σ[1,:]), length(inv_Σ[1,:]))
    #f = 0.1 * 0.5 * dot(η_i .- μ_i, η_i .- μ_i) - dot(η_i, ϕ_i) +(N-1)*log(sum(exp.(η_i)))

    #f = 0.5 * dot(η_i .- μ_i, inv_Σ, η_i .- μ_i) - dot(η_i, ϕ_i)/(N-1) +log(sum(exp.(η_i)))
    return f
end

function gradf!(G, η_i::Array{Float64, 1}, ϕ_i::Array{Float64, 1}, inv_Σ::Array{Float64, 2}, μ_i, N::Int)
    G .= softmax(η_i)*(N-1) .- ϕ_i .+ inv_Σ*(η_i .- μ_i)

    #MMM = Matrix(1.0I, length(inv_Σ[1,:]), length(inv_Σ[1,:]))
    #G .= softmax(η_i)*(N-1) .- ϕ_i .+ 0.1 * (η_i .- μ_i)
    #G .= exp.(η_i)/sum(exp.(η_i))*(N-1) .- ϕ_i .+ inv_Σ*(η_i .- μ_i)
    #G .= exp.(η_i)/sum(exp.(η_i)) .- ϕ_i/(N-1) .+ inv_Σ*(η_i .- μ_i)
end

function hessf!(H, η_i::Array{Float64, 1}, inv_Σ::Array{Float64, 2}, μ_i, N::Int)
    #theta = exp.(η_i)/sum(exp.(η_i))
    theta = softmax(η_i)
    H .=  (N-1)*(diagm(theta) .- theta*theta') .+ inv_Σ

    #MMM = Matrix(1.0I, length(inv_Σ[1,:]), length(inv_Σ[1,:]))
    #H .=  (N-1)*(diagm(theta) .- theta*theta') .+ 0.1 * MMM
    #H .=  (diagm(theta) .- theta*theta') .+ inv_Σ
end


################################################################################
# Function that perform the variational optimization
function Estep_logitNorm!(ϕ::Array{Float64, 3}, λ, ν::Array{Float64, 3},
    inv_Σ::Array{Float64, 2}, μ, N::Int, K::Int)
    G = zeros(K)
    H = zeros(K,K)
    for i in 1:N
        ϕ_i = sum(ϕ[i,:,:],dims=1)[1,:]
        μ_i = μ[:,i]
        res = optimize(η_i -> f(η_i, ϕ_i, inv_Σ, μ_i, N), (G, η_i) -> gradf!(G,η_i, ϕ_i, inv_Σ, μ_i, N), randn(K), BFGS(linesearch = LineSearches.BackTracking(order=2)))#BFGS())
        η_i = Optim.minimizer(res)
        hessf!(H, η_i, inv_Σ, μ_i, N)
        λ[:,i] .= η_i
        ν[:,:,i] .= Hermitian(inv(H))
    end
end


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

function Estep_multinomial_gauss!(ϕ::Array{Float64, 3}, λ, B::Array{Float64, 2},
    Y::Array{Float64, 2}, N::Int, K::Int, like_var::Array{Float64, 2})
    
    # Precompute reusable quantities
    inv_like_var = 0.5 ./ like_var
    log_like_var = 0.5 .* log.(like_var)
    
    # for i in 1:N
   for i in sample(1:N, div(N,4), replace=false)
        # for j in 1:N
       for j in sample(1:N, div(N,4), replace=false)
            if i != j
                for k in 1:K
                    logPi = λ[k,i]
                    for g in 1:K
                        logPi += -ϕ[j,i,g] *( ((Y[i,j] - B[k,g])^2) * inv_like_var[k,g]  + log_like_var[k,g])
                    end
                    ϕ[i,j,k] = logPi
                end
                # ϕ[i,j,:] = softmax(ϕ[i,j,:])
                softmax!(view(ϕ, i, j, :))
            end
        end
    end
end




#########################################################################################
#########################################################################################

function Mstep_blockmodel_gauss_multi!(ϕ::Vector{Array{Float64, 3}}, B::Array{Float64, 2}, like_var::Array{Float64, 2},
    Y::Vector{Array{Float64, 2}}, Ns::Array{Int,1}, K::Int, n_regions::Int)
    lv = 0.
    learn_r = 0.1
    cum_den = 0.
    for k in 1:K
        for g in 1:K
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
        end
    end

    #like_var[1] = lv/cum_den

end

function Mstep_blockmodel_gauss_multi_opt!(ϕ::Vector{Array{Float64, 3}}, B::Array{Float64, 2}, like_var::Array{Float64, 2},
    Y::Vector{Array{Float64, 2}}, Ns::Array{Int,1}, K::Int, n_regions::Int)
    
    learn_r = 0.1  # Learning rate
    
    # Preallocate temporary variables outside loops
    for k in 1:K
        for g in 1:K
            num = 0.0
            den = 0.0
            num_gauss = 0.0
            
            for i_region in 1:n_regions
                ϕ_region = ϕ[i_region]  # Avoid repeated indexing
                Y_region = Y[i_region]
                Ns_region = Ns[i_region]
                
                for j in 1:Ns_region
                    for i in 1:Ns_region
                        phi_prod = ϕ_region[i, j, k] * ϕ_region[j, i, g]
                        
                        # Update accumulators for numerator, denominator, and variance
                        y_val = Y_region[i, j]
                        diff = y_val - B[k, g]
                        num += phi_prod * y_val
                        den += phi_prod
                        num_gauss += phi_prod * diff^2
                    end
                end
            end
            
            # Update parameters using accumulated values
            if den > 0
                B[k, g] = (1 - learn_r) * B[k, g] + learn_r * (num / den)
                like_var[k, g] = (1 - learn_r) * like_var[k, g] + learn_r * (num_gauss / den)
            end
        end
    end
end




#########################################################################################
#########################################################################################
#########################################################################################


function run_VEM_gauss_NN!(n_iterations::Int, ϕ::Vector{Array{Float64, 3}}, λ, ν::Vector{Array{Float64, 3}},
    Σ::Array{Float64, 2}, B::Array{Float64, 2}, like_var::Array{Float64, 2},
    μ, Y::Vector{Array{Float64, 2}}, X::Array{Float64, 2}, Γ, ps, K::Int, Ns::Array{Int,1}, P::Int, n_regions::Int, R::Float64)

    Ncum = 0
    N_s = ones(Int, n_regions)
    N_e = ones(Int, n_regions)
    for i_region in 1:n_regions
        N_s[i_region] = Ncum + 1
        N_e[i_region] = Ncum + Ns[i_region]
        Ncum += Ns[i_region]
    end

    elbows = zeros(n_regions, n_iterations)
    det_Sigma = zeros(n_iterations)
    #det_nu = [zeros(N, n_iterations) for i_reg in 1:n_regions]
    # opt = ADAM(0.01) #the value in the brackets is
    opt = Descent(0.01)
    #################################
    # definition of the loss functional to be used to optimize the flux model
    L(a,b) = (Flux.Losses.kldivergence(softmax(Γ(a)), softmax(b)))

    #################################


    for i_iter in 1:n_iterations
        inv_Σ = inv(Σ)
        for i_region in 1:n_regions
            Estep_logitNorm!(ϕ[i_region], @view(λ[:,N_s[i_region]:N_e[i_region]]), ν[i_region], inv_Σ, Float64.(μ[:,N_s[i_region]:N_e[i_region]]), Ns[i_region], K)
            for m in 1:5
                Estep_multinomial_gauss!(ϕ[i_region], @view(λ[:,N_s[i_region]:N_e[i_region]]), B, Y[i_region], Ns[i_region], K, like_var)
            end
            Estep_logitNorm!(ϕ[i_region], @view(λ[:,N_s[i_region]:N_e[i_region]]), ν[i_region], inv_Σ, Float64.(μ[:,N_s[i_region]:N_e[i_region]]), Ns[i_region], K)

            n_flux = 30
            for i_flux in 1:n_flux
                gs = gradient(()-> L(Float32.(X), Float32.(λ)), ps)
                Flux.Optimise.update!(opt, ps, gs)
            end

            μ = Γ(X);

        end



        RΣ = zeros(K,K)
        for i_region in 1:n_regions
            for i in 1:Ns[i_region]
                #Σ .+= 1/(N*n_regions) * (ν[i_region][:,:,i] .+ (λ[:,i+(i_region-1)*N] .- μ[:,i+(i_region-1)*N])*(λ[:,i+(i_region-1)*N] .- μ[:,i+(i_region-1)*N])')
                RΣ .+= (ν[i_region][:,:,i] .+ (λ[:,i+N_s[i_region]-1] .- μ[:,i+N_s[i_region]-1])*(λ[:,i+N_s[i_region]-1] .- μ[:,i+N_s[i_region]-1])')
            end
        end
        RΣ .= sqrt(8*R*RΣ/(Ncum) + Matrix(1.0I, K, K)) .- Matrix(1.0I, K, K)
        Σ .= Hermitian(RΣ/(4*R))
        #println("\n Σ:  ", det(Σ))
        
        
        Mstep_blockmodel_gauss_multi!(ϕ, B, like_var, Y, Ns, K, n_regions)

        for i_region in 1:n_regions
            # elbows[i_region, i_iter] = -R*Ns[i_region]*tr(Σ) + ELBO_gauss(ϕ[i_region], λ[:,N_s[i_region]:N_e[i_region]], ν[i_region], Σ, B, μ[:,N_s[i_region]:N_e[i_region]],  Y[i_region], K, Ns[i_region], like_var)
            elbows[i_region, i_iter] = 1
            if isnan(elbows[i_region, i_iter])
                break
            end
        end
        det_Sigma[i_iter] = det(Σ)
    end
    return elbows, det_Sigma #, det_nu

end

##########################################################################################