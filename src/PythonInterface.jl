# PythonInterface.jl
# Thin helper providing flat-array entry points for the Python wrapper.
# These functions avoid MLJ Tables so Python only needs to pass matrices and vectors.

export jessamine_fit, jessamine_predict, jessamine_symbolic_string, jessamine_complexity

# Use the predefined inventory constants from Operations.jl
function _get_op_inventory(name::String)
    if name == "polynomial"
        return PolynomialInventory
    elseif name == "rational"
        return RationalFunctionInventory
    elseif name == "explog"
        return ExpLogInventory
    elseif name == "trig"
        return TrigInventory
    elseif name == "hyperbolic"
        return HyperbolicInventory
    else
        @warn "Unknown op_inventory '$name', using PolynomialInventory"
        return PolynomialInventory
    end
end

"""
    jessamine_fit(X::Matrix{Float64}, y::Vector{Float64}; kwargs...)

Fit a Jessamine symbolic regression model on data.
`X` is an n×p matrix (rows=observations, cols=features).
`y` is a length-n target vector.

Returns a NamedTuple containing the fit result that can be passed
to `jessamine_predict`, `jessamine_symbolic_string`, and `jessamine_complexity`.

Keyword arguments:
- `max_time::Int = 300`: Maximum time in seconds
- `output_size::Int = 6`
- `scratch_size::Int = 6`
- `parameter_size::Int = 2`
- `num_time_steps::Int = 3`
- `max_epochs::Int = 10`
- `op_inventory::String = "polynomial"`: One of "polynomial", "rational", "explog", "trig"
- `random_seed::Union{Nothing,Int} = nothing`
- `lambda_model::Float64 = 0.01`
- `lambda_parameter::Float64 = 0.01`
- `lambda_operand::Float64 = 0.01`
- `stop_threshold::Union{Nothing,Float64} = 0.001`
- `num_to_keep::Int = 20`
- `num_to_generate::Int = 40`
- `simplifier::Bool = true`
- `verbosity::Int = 0`
"""
function jessamine_fit(
        X_in::AbstractMatrix{<:Real},
        y_in::AbstractVector{<:Real};
        max_time::Int = 300,
        output_size::Int = 6,
        scratch_size::Int = 6,
        parameter_size::Int = 2,
        num_time_steps::Int = 3,
        max_epochs::Int = 10,
        op_inventory::String = "polynomial",
        random_seed::Union{Nothing, Int} = nothing,
        lambda_model::Float64 = 0.01,
        lambda_parameter::Float64 = 0.01,
        lambda_operand::Float64 = 0.01,
        stop_threshold::Union{Nothing, Float64} = 0.001,
        num_to_keep::Int = 20,
        num_to_generate::Int = 40,
        simplifier::Bool = true,
        verbosity::Int = 0
)
    # Convert to native Julia arrays to avoid PyArray dispatch issues
    X = Matrix{Float64}(X_in)
    y = Vector{Float64}(y_in)
    n_features = size(X, 2)

    # Build operation inventory
    ops = _get_op_inventory(op_inventory)

    # Build neighborhoods with user's settings
    neighborhoods = map(default_neighborhoods) do es
        EpochSpec(
            p_mutate_op = es.p_mutate_op,
            p_mutate_index = es.p_mutate_index,
            p_duplicate_index = es.p_duplicate_index,
            p_delete_index = es.p_delete_index,
            p_duplicate_instruction = es.p_duplicate_instruction,
            p_delete_instruction = es.p_delete_instruction,
            p_hop_instruction = es.p_hop_instruction,
            op_inventory = ops,
            op_probabilities = nothing,
            num_to_keep = num_to_keep,
            num_to_generate = num_to_generate,
            p_take_better = es.p_take_better,
            p_take_very_best = es.p_take_very_best,
            max_generations = es.max_generations,
            stop_on_innovation = es.stop_on_innovation
        )
    end

    # Set up RNG
    rng = isnothing(random_seed) ? Random.default_rng() : Random.Xoshiro(random_seed)

    # Build the simplifier
    simp = simplifier ? default_simplifier : nothing
    if !isnothing(simp)
        simp = EpochSpec(
            p_mutate_op = 0.0,
            p_mutate_index = 0.0,
            p_duplicate_index = 0.0,
            p_delete_index = 0.1,
            p_duplicate_instruction = 0.0,
            p_delete_instruction = 0.1,
            op_inventory = ops,
            max_generations = 100
        )
    end

    # Compute deadline
    deadline = Dates.now() + Dates.Second(max_time)

    # Create the JessamineDeterministic model
    jm = JessamineDeterministic(
        rng = rng,
        lambda_model = lambda_model,
        lambda_parameter = lambda_parameter,
        lambda_operand = lambda_operand,
        output_size = output_size,
        scratch_size = scratch_size,
        parameter_size = parameter_size,
        input_size = n_features,
        num_time_steps = num_time_steps,
        neighborhoods = neighborhoods,
        max_epochs = max_epochs,
        simplifier = simp,
        stop_threshold = stop_threshold,
        stop_deadline = deadline,
        logging_generation_modulus = 10
    )

    # Convert X columns to vector-of-vectors for the MLJ Table interface
    xs = [X[:, j] for j in 1:n_features]
    col_names = [Symbol("x$j") for j in 1:n_features]
    X_table = NamedTuple{Tuple(col_names)}(Tuple(xs))

    # Run the fit
    fit_result, cache, report = MLJModelInterface.fit(jm, verbosity, X_table, y)

    return (
        g_spec = fit_result.g_spec,
        best_agent = fit_result.best_agent,
        rating = report.rating,
    )
end

"""
    jessamine_predict(fit_result, X::Matrix{Float64}) -> Vector{Float64}

Predict target values for new data using a fitted Jessamine model.
"""
function jessamine_predict(fit_result, X_in::AbstractMatrix{<:Real})
    X = Matrix{Float64}(X_in)
    n_features = size(X, 2)
    xs = [X[:, j] for j in 1:n_features]
    col_names = [Symbol("x$j") for j in 1:n_features]
    X_table = NamedTuple{Tuple(col_names)}(Tuple(xs))
    return model_predict(fit_result.g_spec, fit_result.best_agent, X_table)
end

"""
    jessamine_symbolic_string(fit_result) -> String

Return a string representation of the fitted symbolic expression.
Uses Symbolics.jl (not SymPy) to avoid PyCall conflicts.
Variable naming: x1, x2, ... for inputs.
"""
function jessamine_symbolic_string(fit_result)
    try
        result = model_symbolic_output(fit_result.g_spec, fit_result.best_agent)
        if haskey(result, :y_num)
            return string(result.y_num)
        else
            # Fallback: return raw genome symbolic output
            z_strs = [string(z) for z in result.z_num]
            return join(z_strs, ", ")
        end
    catch e
        # Last resort fallback
        try
            result = model_basic_symbolic_output(fit_result.g_spec, fit_result.best_agent)
            if haskey(result, :y_num)
                return string(result.y_num)
            else
                z_strs = [string(z) for z in result.z_num]
                return join(z_strs, ", ")
            end
        catch e2
            return "ERROR: Could not extract symbolic expression: $e2"
        end
    end
end

"""
    jessamine_complexity(fit_result) -> Int

Return the complexity of the fitted model as the total number of operands
across all instructions in the genome.
"""
function jessamine_complexity(fit_result)
    return num_operands(fit_result.best_agent.genome) + num_instructions(fit_result.best_agent.genome)
end
