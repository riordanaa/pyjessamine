# This file has the code for a from-scratch implementation of
# ridge regression with time series.  It was written so we could work on the
# genome and evolution framework before trying to interface with
# MLJ.jl.

export time_series_least_squares_ridge, time_series_least_squares_ridge_grow_and_rate

"""
    time_series_least_squares_ridge(xs, y, lambda, g_spec, genome, parameter; memory_steps = 1)

Compute ridge regression.
Assume `xs` is a vector of state points and `y` is a column of target values.
Apply `run_genome_time_series` using `parameter` as the parameter vector and `xs` as the inputs.
Pass through the given value of `memory_steps`.
Gather a column of 1s and the output columns as a matrix `X`.
The prediction values are `y_hat = X * b`, where `b` is a column of (unknown) coefficients.
Solve for the `b` that minimizes `norm(y - y_hat)^2 + lambda * norm(b)^2`.
If all goes well, return `(norm(y - y_hat), b)`.
Otherwise return `(Inf, nothing)`.
"""
function time_series_least_squares_ridge(
        xs::AbstractArray,
        y::AbstractArray,
        lambda::Number,
        g_spec::GenomeSpec,
        genome::AbstractGenome,
        parameter::AbstractArray;
        memory_steps = 1)
    @assert g_spec.input_size == length(xs[1]) + memory_steps * g_spec.output_size
    num_rows = length(y)
    # local outputs::Vector{Vector{Vector{Float64}}}
    outputs = run_genome_time_series(g_spec, genome, parameter, xs, memory_steps = memory_steps)
    # local last_round::Vector{Vector{Float64}}
    X = stack(outputs, dims=1)
    @assert length(outputs[1]) == size(X, 2)
    @assert num_rows == size(X, 1)
    b = (X' * X + lambda * I) \ (X' * y)
    y_hat = X * b
    r = y - y_hat
    n = norm(r)
    if isnan(n)
        return (Inf, nothing)
    else
        return (n, b)
    end
end

"""
    time_series_least_squares_ridge_grow_and_rate(xs, y, lambda_b, lambda_p, lambda_op, g_spec, genome, memory_steps = 1, p_init = zeros(...))

Solve for the parameter vector `p` that minimizes
`norm(y - y_hat)^2 + lambda_b * norm(b)^2 + lambda_p * norm(p)^2 + lambda_op R`,
where `y_hat` and `b` are found using `time_series_least_squares_ridge`.
`R` is the total number of operands across all instructions in `genome`.

The solver starts with `p_init` for the initial value of `p`.
If `p_init` is `nothing` or not given, a vector of zeros is used.

If all goes well, return an `Agent`, whose genome is `genome`,
whose `parameter` is the best `p`, and whose `extra` is a
`BasicLinearModelResult` with coefficient vector `b`.

Otherwise, return `nothing`.

"""
function time_series_least_squares_ridge_grow_and_rate(
        xs,
        y,
        lambda_b::Number,
        lambda_p::Number,
        lambda_operand::Number,
        g_spec::GenomeSpec,
        genome::AbstractGenome;
        memory_steps = 1,
        p_init::AbstractArray = zeros(g_spec.parameter_size)
)::Union{Agent, Nothing}
    optim_fn = OptimizationFunction(_TSLSRGR_f)
    c = _TSLSRGR_Context(g_spec, genome, memory_steps, lambda_b, xs, y, lambda_p, nothing, nothing)
    optim_prob = OptimizationProblem(optim_fn, p_init, c, sense = MinSense)
    try
        sol = solve(optim_prob, NelderMead())
        if SciMLBase.successful_retcode(sol)
            _TSLSRGR_f(sol.u, c)
            @assert !isnothing(c.b) "Optimization should have succeeded."
            r = sol.objective + lambda_operand * num_operands(genome)
            return Agent(r, genome, sol.u, BasicLinearModelResult(c.b))
        else
            @debug "Solve for optimal p did not succeed: $(sol.retcode)"
            return Agent(infinitely_bad(optim_prob.sense), genome, nothing, nothing)
        end
    catch e
        if isa(e, ArgumentError) || isa(e, SingularException) || isa(e, DomainError)
            @debug "Masking exception" exception=e
            return Agent(infinitely_bad(optim_prob.sense), genome, nothing, nothing)
        end
        rethrow()
    end
end

#=
function time_series_least_squares_ridge_grow_and_rate(
        xs::AbstractArray{<:AbstractArray},
        y::AbstractArray,
        lambda_b::Number,
        lambda_p::Number,
        lambda_operand::Number,
        g_spec::GenomeSpec,
        genome::AbstractGenome,
        p_init::Nothing,
        memory_steps = 1
)::Union{Agent, Nothing}
    return time_series_least_squares_ridge_grow_and_rate(
        xs, y, lambda_b, lambda_p, lambda_operand, g_spec, genome,
        memory_steps = memory_steps)
end
=#

@kwdef mutable struct _TSLSRGR_Context{TXs, Ty}
    g_spec::GenomeSpec
    genome::AbstractGenome
    memory_steps::Int
    lambda_b::Float64
    xs::TXs
    y::Ty
    lambda_p::Float64
    n::Union{Float64, Nothing}
    b::Union{Vector{Float64}, Nothing}
end

function _TSLSRGR_f(u::Vector{Float64}, c::_TSLSRGR_Context{TXs, Ty}) where {TXs, Ty}
    n, b = time_series_least_squares_ridge(c.xs, c.y, c.lambda_b, c.g_spec, c.genome, u,
        memory_steps = c.memory_steps)
    if isnothing(b)
        # Ridge regression failed.
        # This should be some form of infinity.
        return n
    else
        # Ridge regression succeeded.
        c.n = n
        c.b = b
        return n^2 + c.lambda_b * dot(b, b) + c.lambda_p * dot(u, u)
    end
end
