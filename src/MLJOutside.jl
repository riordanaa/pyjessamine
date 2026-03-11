# Make language server happy
if false
    using Base: nothing_sentinel
    using MLJ
    using MLJModelInterface
    using Optimization
    using Random
    using Tables
    include("Evolution.jl")
    include("GenomeCore.jl")
    include("Operations.jl")
    include("MachineLayer.jl")
    include("MLJInside.jl")
    include("Mutation.jl")
    include("SymbolicForm.jl")
end

export EpochSpec, JessamineModel
export area_above_curve

# This is kind of a mess because in MLJ, the input size
# is not really available until the call to machine().
# Which means that the GenomeSpec can't be part of the model.
# Which means that EvolutionSpec can't be part of the model.
# Hence the following massive config system.
#
# This is why there are overloads for convert() for spec types.
# convert(::Type{SpecType}, x) constructs the spec object using
# fields of x with the same name as a field in the SpecType.
# That way we can call convert(MutationSpec, my_epoc_spec) for
# example.

"""
struct EpochSpec

    EpochSpec is a struct that contains the parameters for building `MutationSpec`, `SelectionSpec`, and `EvolutionSpec` objects.
    It's needed to build the `JessamineModel` object.
"""
@kwdef mutable struct EpochSpec
    # MutationSpec
    p_mutate_op::Float64 = 0.1
    p_mutate_index::Float64 = 0.1
    p_duplicate_index::Float64 = 0.01
    p_delete_index::Float64 = 0.01
    p_duplicate_instruction::Float64 = 0.001
    p_delete_instruction::Float64 = 0.001
    p_hop_instruction::Float64 = 0.0
    op_inventory::Vector{AbstractGeneOp} = [Add(), Subtract(), Multiply()]
    op_probabilities::Union{Vector{Float64}, Nothing} = nothing

    # SelectionSpec
    num_to_keep::Int = 20
    num_to_generate::Int = 40
    p_take_better::Float64 = 0.8
    p_take_very_best::Float64 = 0.25

    # EvolutionSpec
    max_generations::Int = 10
    stop_on_innovation::Bool = false
end

function EpochSpec(
        m_spec::MutationSpec,
        s_spec::SelectionSpec,
        max_generations::Integer,
        stop_on_innovation::Bool = false)
    return EpochSpec(
        m_spec.p_mutate_op,
        m_spec.p_mutate_index,
        m_spec.p_duplicate_index,
        m_spec.p_delete_index,
        m_spec.p_duplicate_instruction,
        m_spec.p_delete_instruction,
        m_spec.p_hop_instruction,
        m_spec.op_inventory,
        m_spec.op_probabilities,
        s_spec.num_to_keep,
        s_spec.num_to_generate,
        s_spec.p_take_better,
        s_spec.p_take_very_best,
        max_generations,
        stop_on_innovation
    )
end

# Set up default neighborhood structure

e_spec_1 = EpochSpec()
e_spec_2 = EpochSpec(
    p_mutate_op = 0.2,
    p_mutate_index = 0.2,
    p_duplicate_index = 0.02,
    p_delete_index = 0.02,
    p_duplicate_instruction = 0.002,
    p_delete_instruction = 0.002,
    max_generations = 40,
    stop_on_innovation = true)
e_spec_3 = EpochSpec(
    p_mutate_op = 0.2,
    p_mutate_index = 0.2,
    p_duplicate_index = 0.02,
    p_delete_index = 0.02,
    p_duplicate_instruction = 0.002,
    p_delete_instruction = 0.002,
    num_to_keep = 10,
    num_to_generate = 50,
    p_take_better = 0.6,
    p_take_very_best = 0.0,
    max_generations = 40,
    stop_on_innovation = true)
e_spec_4 = EpochSpec(
    p_mutate_op = 0.3,
    p_mutate_index = 0.3,
    p_duplicate_index = 0.03,
    p_delete_index = 0.03,
    p_duplicate_instruction = 0.003,
    p_delete_instruction = 0.003,
    num_to_keep = 10,
    num_to_generate = 50,
    p_take_better = 0.6,
    p_take_very_best = 0.0,
    max_generations = 40,
    stop_on_innovation = true)
e_spec_5 = EpochSpec(
    p_mutate_op = 0.3,
    p_mutate_index = 0.3,
    p_duplicate_index = 0.03,
    p_delete_index = 0.03,
    p_duplicate_instruction = 0.003,
    p_delete_instruction = 0.003,
    num_to_keep = 10,
    num_to_generate = 50,
    p_take_better = 0.5,
    p_take_very_best = 0.0,
    max_generations = 40,
    stop_on_innovation = true)

const default_neighborhoods = [
    e_spec_1, e_spec_2, e_spec_3, e_spec_4, e_spec_5
]

const default_simplifier = EpochSpec(
    p_mutate_op = 0.0,
    p_mutate_index = 0.0,
    p_duplicate_index = 0.0,
    p_delete_index = 0.1,
    p_duplicate_instruction = 0.0,
    p_delete_instruction = 0.1,
    max_generations = 100
)

# I really need something like
# struct JModel{Super} <: Super   model::Super ... end
# so that JModel{Deterministic} <: Deterministic.
# However, this is not possible in Julia's type system.
# The syntax struct Thing{T} <: T ... end
# results in Thing{T} <: (T where T),
# or equivalently Thing{T} <: Any.
# Thing{T} <: T evaluates to false.

# So we resort to macros.

# Complication: @mlj_model only works on a
# non-parametric struct.

# Complication: There can't be a field called `model` inside the
# @mlj_model because of some goofiness with how it's implemented.

# Complication: @mlj_model doesn't work inside another
# macro because the clean! method comes out with a hygenized
# type in its definition.  That is, the model field in the struct
# conflicts with a local variable called model in the generated clean!
# method.  So we have to use @kwdef instead of @mlj_model.

# Complication: @kwdef barfs if I try to put a type parameter here.
# As in `@kwdef mutable struct $(struct_name){T}` just won't work.

macro declareJessamineModel(model_type, default_model, default_performance)
    struct_name = Symbol("Jessamine$(model_type)")
    return quote
        export $struct_name
        @kwdef mutable struct $struct_name <: $model_type
            model::$model_type = $default_model

            rng::AbstractRNG = Random.default_rng()
            init_arity_dist::Distribution = DiscreteNonParametric(
                [1, 2, 3], [0.25, 0.5, 0.25])

            # *MachineModelSpec
            lambda_model::Float64 = 0.01
            lambda_parameter::Float64 = 0.01
            lambda_operand::Float64 = 0.01
            performance::Any = $default_performance

            # GenomeSpec
            output_size::Int = 6
            scratch_size::Int = 6
            parameter_size::Int = 2
            input_size::Int = 0
            num_time_steps::Int = 3

            neighborhoods::AbstractVector{EpochSpec} = default_neighborhoods
            max_epochs::Int = 10
            simplifier::Union{Nothing, EpochSpec} = default_simplifier
            sense::Any = MinSense
            stop_threshold::Union{Nothing, Number} = 0.001
            stop_channel::Union{Nothing, Channel} = nothing
            stop_deadline::Union{Nothing, DateTime} = nothing

            logging_generation_modulus::Int = 10
        end
    end
end

"""
    area_above_curve(y_hat, y_ref)

Return the area *above* the ROC curve for the predictions
`y_hat` and actual reference values `y_ref`.
This is 1 minus the area under the curve.
The vector `y_hat` should consist of Bernoulli distributions,
as returned by a `LogisticClassifier`, for example.
The vector `y` should consist of values from those distributions.
"""
function area_above_curve(y_hat, y_ref)
    return 1.0 - MLJ.area_under_curve(y_hat, y_ref)
end

@declareJessamineModel Deterministic RidgeRegressor(lambda = 0.01) MLJ.l2
@declareJessamineModel Probabilistic LogisticClassifier() area_above_curve
"""
A union of `JessamineDeterministic` and `JessamineProbabilistic`
"""

const JessamineModel = Union{JessamineDeterministic, JessamineProbabilistic}

"""
    supports_weights(::Type{JessamineModel})

Return `true`.  A Jessamine model supports weight vectors for
training data *if* the inner model does.  Since MLJ determines
this at the type level, it has to return `true` here to allow a
weight vector to be passed in.  The `fit` implementation throws
an exception if a non-`nothing` weight vector is passed in and
the inner model does not allow a weight vector.
"""
function MLJ.supports_weights(::Type{JessamineDeterministic})
    return true
end

"""
    supports_weights(::Type{JessamineModel})

Return `true`.  A Jessamine model supports weight vectors for
training data *if* the inner model does.  Since MLJ determines
this at the type level, it has to return `true` here to allow a
weight vector to be passed in.  The `fit` implementation throws
an exception if a non-`nothing` weight vector is passed in and
the inner model does not allow a weight vector.
"""
function MLJ.supports_weights(::Type{JessamineProbabilistic})
    return true
end

"""
    machine_spec_type(t_MLJ_model::Type)::Type{<:AbstractMachineSpec}

Return the Jessamine type (subtype of `AbstractMachineSpec`)
corresponding to the given MLJ model type.
The default is to return `BasicModelMachineSpec`
"""
function machine_spec_type(::Type)
    return BasicModelMachineSpec
end

function machine_spec_type(x)
    return machine_spec_type(typeof(x))
end

"""
    machine_spec_type(::Type{RidgeRegressor})

Return `LinearModelMachineSpec`
"""
function machine_spec_type(::Type{RidgeRegressor})
    return LinearModelMachineSpec
end

"""
    machine_spec_type(::Type{LassoRegressor})

Return `LinearModelMachineSpec`
"""
function machine_spec_type(::Type{LassoRegressor})
    return LinearModelMachineSpec
end

"""
    machine_spec_type(::Type{LogisticClassifier})

Return `LinearModelMachineSpec`
"""
function machine_spec_type(::Type{LogisticClassifier})
    return LinearModelMachineSpec
end

function build_specs!(
        jm::JessamineModel,
        X,
        y,
        w = nothing)
    xs = Tables.columns(X)
    jm.input_size = length(xs)
    g_spec = convert(GenomeSpec, jm)
    mn_spec = convert(machine_spec_type(jm.model), jm)
    function grow_and_rate(rng, g_spec, genome)
        return machine_grow_and_rate(
            xs, y, g_spec, genome, mn_spec,
            DefaultSolverSpec(), w)
    end
    neighborhoods = map(jm.neighborhoods) do epoch_spec
        m_spec = convert(MutationSpec, epoch_spec)
        s_spec = convert(SelectionSpec, epoch_spec)
        e_spec = EvolutionSpec(
            g_spec,
            m_spec,
            s_spec,
            grow_and_rate,
            epoch_spec.max_generations,
            epoch_spec.stop_on_innovation)
        return e_spec
    end
    if isnothing(jm.simplifier)
        simp_spec = nothing
    else
        simp_spec = EvolutionSpec(
            g_spec,
            convert(MutationSpec, jm.simplifier),
            convert(SelectionSpec, jm.simplifier),
            grow_and_rate,
            jm.simplifier.max_generations
        )
    end
    return (g_spec = g_spec,
        mn_spec = mn_spec,
        neighborhoods = neighborhoods,
        simp_spec = simp_spec)
end

function MLJModelInterface.fit(
        jm::JessamineModel,
        verbosity::Int,
        X,
        y,
        w = nothing)
    # Check whether weights are supported by the inner model:
    if !isnothing(w)
        @assert MLJ.supports_weights(typeof(jm.model)) "Inner model does not support a weight vector; use `w = nothing`"
    end
    specs = build_specs!(jm, X, y, w)
    init_neighborhood = specs.neighborhoods[1]
    @info "Building initial population"
    pop_init = random_initial_population(
        jm.rng,
        init_neighborhood,
        jm.init_arity_dist,
        sense = jm.sense)
    return MLJModelInterface.update(
        jm, verbosity, nothing,
        (specs = specs, pop = pop_init),
        X, y, w)
end

"""
    do_simplification(::AbstractPopulationCondition)

Return whether to perform the simplification epoch.
Utility function.
"""
do_simplification(::AbstractPopulationCondition) = true
do_simplification(::ReachedDeadline) = false
do_simplification(::ReceivedStopMessage) = false


function MLJModelInterface.update(
        jm::JessamineModel,
        verbosity::Int,
        old_fit_result,
        old_cache,
        X,
        y,
        w = nothing)
    # This asssumes X and y have already been baked into
    # old_cache.specs
    specs = old_cache.specs
    pop_init = old_cache.pop
    if verbosity > 0
        @info "Begin VNS loop"
    end
    pop_next = vns_evolution_loop(
        jm.rng,
        specs.neighborhoods,
        pop_init;
        generation_mod = jm.logging_generation_modulus,
        verbosity = verbosity,
        max_epochs = jm.max_epochs,
        stop_threshold = jm.stop_threshold,
        stop_channel = jm.stop_channel,
        stop_deadline = jm.stop_deadline)
    if !isnothing(jm.simplifier)
        if do_simplification(pop_next.condition)
            if verbosity > 0
                @info "Begin simplification epoch"
            end
            pre_simp_condition = pop_next.condition
            pop_next = evolution_loop(
                jm.rng,
                specs.simp_spec,
                pop_next;
                verbosity = verbosity,
                generation_mod = jm.logging_generation_modulus,
                sense = jm.sense,
                stop_threshold = jm.stop_threshold,
                stop_channel = jm.stop_channel,
                stop_deadline = jm.stop_deadline
            )
            if pop_next.condition == InProgress()
                pop_next = Population(pop_next, pre_simp_condition)
            end
        elseif verbosity > 0
            @info "Skipping simplification: $(describe_condition(pop_next.condition))"
        end
    end
    if verbosity > 0
        @info "Evolution ended"
    end
    best_agent = pop_next.agents[1]
    best_rating = best_agent.rating
    symbolic_result = run_genome_symbolic(specs.g_spec, best_agent.genome)
    fit_result = (
        g_spec = specs.g_spec,
        best_agent = best_agent,
        inner_fitted_params = MLJ.fitted_params(best_agent.extra.m))
    report = (
        rating = best_rating,
        symbolic_result = symbolic_result)
    # Including a non-nothing cache object here gives MLJ.fit! a
    # way to continue if called again by calling update() I
    # think...
    cache = (specs = specs, pop = pop_next)
    return (fit_result, cache, report)
end

function MLJ.fitted_params(jm::JessamineModel, fit_result)
    return fit_result
end

function MLJModelInterface.predict(jm::JessamineModel, fit_result, X)
    return model_predict(fit_result.g_spec, fit_result.best_agent, X)
end


"""
    model_basic_symbolic_output(fit_result::NamedTuple; kw_args...)

Call `model_basic_symbolic_output` with the `g_spec::GenomeSpec`
and `best_agent::Agent` found in `fit_result`.
Return the result.
"""
function model_basic_symbolic_output(fit_result::NamedTuple; kw_args...)
    return model_basic_symbolic_output(
        fit_result.g_spec,
        fit_result.best_agent;
        kw_args...)
end

"""
    model_symbolic_output(fit_result::NamedTuple; kw_args...)

Call `model_symbolic_output` with the `g_spec::GenomeSpec`
and `best_agent::Agent` found in `fit_result`.
Return the result.
"""
function model_symbolic_output(fit_result::NamedTuple; kw_args...)
    return model_symbolic_output(
        fit_result.g_spec,
        fit_result.best_agent;
        kw_args...)
end


"""
    model_basic_sympy_output(fit_result::NamedTuple; kw_args...)

Call `model_basic_sympy_output` with the `g_spec::GenomeSpec`
and `best_agent::Agent` found in `fit_result`.
Return the result.
"""
function model_basic_sympy_output(fit_result::NamedTuple; kw_args...)
    return model_basic_sympy_output(
        fit_result.g_spec,
        fit_result.best_agent;
        kw_args...)
end

"""
    model_sympy_output(fit_result::NamedTuple; kw_args...)

Call `model_sympy_output` with the `g_spec::GenomeSpec`
and `best_agent::Agent` found in `fit_result`.
Return the result.
"""
function model_sympy_output(fit_result::NamedTuple; kw_args...)
    return model_sympy_output(
        fit_result.g_spec,
        fit_result.best_agent;
        kw_args...)
end


"""
    show_symbolic(fit_result::NamedTuple; kw_args...)

Call `show_symbolic` with the `g_spec::GenomeSpec`
and `best_agent.genome::AbstractGenome` found in `fit_result`.
Return the result.
"""
function show_symbolic(fit_result::NamedTuple; kw_args...)
    return show_symbolic(
        fit_result.g_spec,
        fit_result.best_agent.genome;
        kw_args...)
end

"""
    show_sympy(fit_result::NamedTuple; kw_args...)

Call `show_sympy` with the `g_spec::GenomeSpec`
and `best_agent.genome::AbstractGenome` found in `fit_result`.
Return the result.
"""
function show_sympy(fit_result::NamedTuple; kw_args...)
    return show_sympy(
        fit_result.g_spec,
        fit_result.best_agent.genome;
        kw_args...)
end
