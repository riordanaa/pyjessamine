#!/usr/bin/env julia
"""
benchmark_julia_baseline.jl — Run Jessamine.jl natively on benchmark problems
and save results to JSON for comparison with the Python wrapper.

Usage:
    julia --project=. benchmark_julia_baseline.jl
"""

using Logging
global_logger(ConsoleLogger(stderr, Logging.Warn))

using Jessamine
using Random
using Dates
using LinearAlgebra
using Statistics

# ── Benchmark problems ──────────────────────────────────────────────────

function make_polynomial(n_samples=200, seed=42)
    rng = Random.Xoshiro(seed)
    X = randn(rng, n_samples, 2)
    y = X[:, 1].^2 .+ 2 .* X[:, 2]
    return X, y, "polynomial", "y = x1^2 + 2*x2"
end

function make_rational(n_samples=200, seed=42)
    rng = Random.Xoshiro(seed)
    X = randn(rng, n_samples, 2)
    y = X[:, 1] ./ (1.0 .+ X[:, 2].^2)
    return X, y, "rational", "y = x1 / (1 + x2^2)"
end

function make_kepler(n_samples=200, seed=42)
    rng = Random.Xoshiro(seed)
    a = 0.5 .+ 9.5 .* rand(rng, n_samples)
    T = a .^ 1.5
    X = reshape(a, :, 1)
    return X, T, "kepler", "T = a^(3/2)"
end

function make_nguyen7(n_samples=200, seed=42)
    rng = Random.Xoshiro(seed)
    x = 2.0 .* rand(rng, n_samples)
    y = log.(x .+ 1) .+ log.(x.^2 .+ 1)
    X = reshape(x, :, 1)
    return X, y, "nguyen7", "y = log(x+1) + log(x^2+1)"
end

# ── Common hyperparams ──────────────────────────────────────────────────

const BENCHMARK_PARAMS = Dict(
    :max_time => 120,
    :output_size => 6,
    :scratch_size => 6,
    :parameter_size => 2,
    :num_time_steps => 3,
    :max_epochs => 5,
    :num_to_keep => 20,
    :num_to_generate => 40,
    :random_seed => 42,
    :verbosity => 0,
)

# ── Run one benchmark ───────────────────────────────────────────────────

function run_benchmark(make_fn; op_inventory="polynomial")
    X, y, name, ground_truth = make_fn()
    n = size(X, 1)

    # Train/test split (same as Python: first 80% train, last 20% test with seed 42)
    n_train = Int(floor(0.8 * n))
    # Use same split logic: shuffle with seed 42
    rng = Random.Xoshiro(42)
    perm = randperm(rng, n)
    train_idx = perm[1:n_train]
    test_idx = perm[n_train+1:end]

    X_train = X[train_idx, :]
    y_train = y[train_idx]
    X_test = X[test_idx, :]
    y_test = y[test_idx]

    println("\n" * "="^60)
    println("  Problem: $name ($ground_truth)")
    println("  Op inventory: $op_inventory")
    println("  Train: $(length(y_train)), Test: $(length(y_test))")
    println("="^60)

    # Fit
    t0 = time()
    result = jessamine_fit(X_train, y_train;
        max_time = BENCHMARK_PARAMS[:max_time],
        output_size = BENCHMARK_PARAMS[:output_size],
        scratch_size = BENCHMARK_PARAMS[:scratch_size],
        parameter_size = BENCHMARK_PARAMS[:parameter_size],
        num_time_steps = BENCHMARK_PARAMS[:num_time_steps],
        max_epochs = BENCHMARK_PARAMS[:max_epochs],
        op_inventory = op_inventory,
        random_seed = BENCHMARK_PARAMS[:random_seed],
        num_to_keep = BENCHMARK_PARAMS[:num_to_keep],
        num_to_generate = BENCHMARK_PARAMS[:num_to_generate],
        verbosity = BENCHMARK_PARAMS[:verbosity],
    )
    elapsed = time() - t0

    # Predict
    y_pred_train = jessamine_predict(result, X_train)
    y_pred_test = jessamine_predict(result, X_test)

    # R-squared
    ss_res_train = sum((y_train .- y_pred_train).^2)
    ss_tot_train = sum((y_train .- mean(y_train)).^2)
    r2_train = 1.0 - ss_res_train / ss_tot_train

    ss_res_test = sum((y_test .- y_pred_test).^2)
    ss_tot_test = sum((y_test .- mean(y_test)).^2)
    r2_test = 1.0 - ss_res_test / ss_tot_test

    # Symbolic expression
    sym_str = jessamine_symbolic_string(result)
    compl = jessamine_complexity(result)

    println("  Rating:      $(result.rating)")
    println("  R2 (train):  $r2_train")
    println("  R2 (test):   $r2_test")
    println("  Symbolic:    $sym_str")
    println("  Complexity:  $compl")
    println("  Runtime (s): $(round(elapsed, digits=2))")

    # Save predictions to file for Python comparison
    outdir = joinpath(@__DIR__, "python", "results")
    mkpath(outdir)
    outfile = joinpath(outdir, "julia_baseline_$(name).txt")
    open(outfile, "w") do f
        println(f, "problem=$name")
        println(f, "ground_truth=$ground_truth")
        println(f, "op_inventory=$op_inventory")
        println(f, "r2_train=$r2_train")
        println(f, "r2_test=$r2_test")
        println(f, "rating=$(result.rating)")
        println(f, "symbolic=$sym_str")
        println(f, "complexity=$compl")
        println(f, "runtime=$elapsed")
        println(f, "n_train=$(length(y_train))")
        println(f, "n_test=$(length(y_test))")
        # Save all predictions for parity check
        println(f, "predictions_train=$(join(y_pred_train, ","))")
        println(f, "predictions_test=$(join(y_pred_test, ","))")
        # Save train/test indices for reproducibility
        println(f, "train_indices=$(join(train_idx, ","))")
        println(f, "test_indices=$(join(test_idx, ","))")
    end
    println("  Saved to: $outfile")

    return Dict(
        "name" => name,
        "r2_train" => r2_train,
        "r2_test" => r2_test,
        "symbolic" => sym_str,
        "complexity" => compl,
        "runtime" => elapsed,
    )
end

# ── Main ────────────────────────────────────────────────────────────────

function main()
    println("Jessamine.jl Benchmark Suite — Julia Baseline")
    println("Julia version: $(VERSION)")
    println("Timestamp: $(Dates.now())")

    results = []

    # Problem 1: Polynomial (easy)
    push!(results, run_benchmark(make_polynomial; op_inventory="polynomial"))

    # Problem 2: Rational (medium)
    push!(results, run_benchmark(make_rational; op_inventory="rational"))

    # Problem 3: Kepler (medium-hard)
    push!(results, run_benchmark(make_kepler; op_inventory="polynomial"))

    # Problem 4: Nguyen-7 (hard, needs explog)
    try
        push!(results, run_benchmark(make_nguyen7; op_inventory="explog"))
    catch e
        println("\n[!] Nguyen-7 with explog failed: $e")
        println("    Retrying with polynomial inventory...")
        try
            push!(results, run_benchmark(make_nguyen7; op_inventory="polynomial"))
        catch e2
            println("    Nguyen-7 with polynomial also failed: $e2")
        end
    end

    println("\n" * "="^60)
    println("  SUMMARY")
    println("="^60)
    for r in results
        println("  $(r["name"]): R2_test=$(round(r["r2_test"], digits=6)), time=$(round(r["runtime"], digits=1))s")
    end
end

main()
