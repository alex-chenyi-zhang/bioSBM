include("bioSBM_fns.jl")

function run_inference_gauss_multi_NN(n_iter::Int, n_runs::Int, covariate_file_names::String, map_file_names::String, K::Int, R::Float64)
    println("STOCHA!  trace norm NN!! yay R = ", R)
    io = open(covariate_file_names, "r")
    covariate_files = readdlm(io, String)
    close(io)

    io = open(map_file_names, "r")
    contact_map_files = readdlm(io, String)
    close(io)

    n_regions = length(contact_map_files)
    println(n_regions)

    println(contact_map_files[1], "\t" , covariate_files[1])
    # read the first files from list
    #=io = open(covariate_files[1],"r")
    X_i = readdlm(io, Float64; header=true)[1]#[1:end-1,:]
    close(io)

    io = open(contact_map_files[1],"r")
    Y_i = readdlm(io, Float64)
    close(io)=#

    #N = length(X_i[1,:])
    Ns = ones(Int, n_regions)
    Ncum = 0

    N_starts = ones(Int, n_regions)
    N_ends = ones(Int, n_regions)
    P = 0

    for i_region in 1:n_regions
        println(contact_map_files[i_region], "\t" , covariate_files[i_region])
        io = open(covariate_files[i_region],"r")
        X_i = readdlm(io, Float64; header=true)[1]#[1:end-1,:]
        close(io)

        Ns[i_region] = length(X_i[1,:])
        N_starts[i_region] = Ncum + 1
        N_ends[i_region] = Ncum + Ns[i_region]
        Ncum += Ns[i_region]
        P = length(X_i[:,1])
    end
    println("Ns = ", Ns)
    println("Nsstart = ", N_starts)
    println("Nends = ", N_ends)
    println("Ntot: ", Ncum)



    # data structures for aggregated covariates and contact maps
    #X = zeros(P,N*n_regions)
    #Y = zeros(N, N*n_regions)
    #X[:,1:N] .= X_i
    #Y[:,1:N] .= Y_i
    X = zeros(P, Ncum)
    Y = [zeros(Ns[i_region], Ns[i_region]) for i_region in 1:n_regions]
    #X[:,1:Ns[1]] .= X_i
    #Y[1] .= Y_i
    for i_region in 1:n_regions
        println(contact_map_files[i_region], "\t" , covariate_files[i_region])
        io = open(covariate_files[i_region],"r")
        X_i = readdlm(io, Float64; header=true)[1]#[1:end-1,:]
        close(io)

        io = open(contact_map_files[i_region],"r")
        Y_i = readdlm(io, Float64)
        close(io)
        X[:,N_starts[i_region]:N_ends[i_region]] .= X_i
        Y[i_region] .= Y_i
    end
    println(Ns, "\t", P)

    for i_run in 1:n_runs
        ##############################    Initialize all parameters

        println("run number: ", i_run)
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
        println("Linear model!!!")
        ps  = Flux.params(Γ)
        #opt = ADAM(0.01) # the value in brackts is the learnin rate for the optmizer

        #Γ = zeros(K,P)
        #for k in 1:K
        #    Γ[k,:] .= randn(P)* sqrt(σ_2[k])
        #end

        μ = zeros(K,Ncum)
        #μ = Γ * X;
        μ = Γ(Float32.(X));
        for i_region in 1:n_regions
            for i in 1:Ns[i_region]
                ϕ[i_region][i,i,:] .= 0
            end
        end
        #println(μ)
        ###### end of initialization

        elbows, det_Sigma = run_VEM_gauss_NN!(n_iter, ϕ, λ, ν, Σ, B, like_var, μ, Y, X, Γ, ps, K, Ns, P, n_regions, R)

        i1 = findfirst('_', covariate_file_names)[1]
        cell_line = covariate_file_names[6:i1-1]
        cov_file_n = covariate_file_names[i1+findfirst(cell_line, covariate_file_names[i1:end])[1]-1:end]

        μ = Γ(X);
        # data_dir = "data/results/linear/SVI_$(cov_file_n[1:end-4])_NN_regularized$(R)/"
        # println(data_dir)

        # if !isdir(data_dir)
        #     mkdir(data_dir)
        # end

        # for i_region in 1:n_regions
        #     chr_ind = findfirst("_chr",covariate_files[i_region])[1]
        #     #=nu_matrix = zeros(N,K*K)
        #     for i in 1:N
        #         nu_matrix[i,:] .= [ν[i_region][:,:,i]...]
        #     end=#
        #     println((covariate_files[i_region][chr_ind:end]))

        #     open("$(data_dir)elbows_$(Ns[i_region])_$(K)$(covariate_files[i_region][chr_ind:end])", "a") do io
        #         writedlm(io, elbows[i_region,:]')
        #     end
        #     # open("$(data_dir)nu_$(Ns[i_region])_$(K)$(covariate_files[i_region][chr_ind:end])", "a") do io
        #     #     writedlm(io, nu_matrix)
        #     # end
        #     open("$(data_dir)lambda_$(Ns[i_region])_$(K)$(covariate_files[i_region][chr_ind:end])", "a") do io
        #         writedlm(io, λ[:, N_starts[i_region]:N_ends[i_region]])
        #     end
        #     open("$(data_dir)mu_$(Ns[i_region])_$(K)$(covariate_files[i_region][chr_ind:end])", "a") do io
        #         writedlm(io, μ[:, N_starts[i_region]:N_ends[i_region]])
        #     end
        #     #open("$(data_dir)det_nu_$(N)_$(K)$(covariate_files[i_region][chr_ind:end])", "a") do io
        #     #    writedlm(io, det_nu[i_region]')
        #     #end
        # end

        # open("$(data_dir)B_$(K).txt", "a") do io
        #     writedlm(io, B)
        # end
        # open("$(data_dir)Sigma_$(K).txt", "a") do io
        #     writedlm(io, Σ)
        # end
        # open("$(data_dir)like_var_$(K).txt", "a") do io
        #     writedlm(io, like_var)
        # end

        # open("$(data_dir)det_Sigma_$(K).txt", "a") do io
        #     writedlm(io, det_Sigma')
        # end

        # model_state = Flux.state(Γ)
        # jldsave("$(data_dir)$(i_run)_Gamma_$(K).jld2"; model_state)

        # #@save "$(data_dir)$(i_run)_Gamma_$(N)_$(K)_chr2_100k.bson" Γ
    end
end
