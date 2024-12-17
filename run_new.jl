include("src/bioSBM_main.jl")

max_n_iter = 3000
n_runs = 4
covariate_file_names = "data/GM12878_pq_input_files/X_files_GM12878_pq_even_100k.txt"
map_file_names = "data/GM12878_pq_input_files/Y_files_GM12878_pq_even_100k.txt"

K = 9
R = 0.1

run_inference_gauss_multi_NN(max_n_iter, n_runs, covariate_file_names, map_file_names, K, R)