include("src/bioSBM_main.jl")

R = 0.1
K = 8
iters = 10

dirX = "data/GM12878_input_files/X_files_GM12878.txt"
dirY = "data/GM12878_input_files/Y_files_GM12878.txt"


run_inference_gauss_multi_NN(iters, 2, "$(dirX)", "$(dirY)", K, R)
