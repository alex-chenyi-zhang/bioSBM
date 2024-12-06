include("src/bioSBM_main.jl")

n_iter = 1100
n_runs = 10
covariate_file_names = "data/GM12878_input_files/X_files_GM12878_even_100k.txt"
map_file_names = "data/GM12878_input_files/Y_files_GM12878_even_100k.txt"

K = 13
R = 0.1

run_inference_gauss_multi_NN(n_iter, n_runs, covariate_file_names, map_file_names, K, R)