# Make language server happy
if false
    using Distributions
    include("GenomeCore.jl")
    include("Operations.jl")
    include("Recombination.jl")
    include("Mutation.jl")
end

export SelectionSpec, SelectionDist, Population
export Agent
export AbstractPopulationCondition, InitialPopulation, InProgress
export DiscoveredInnovation, ReachedStopThreshold, ReachedDeadline
export ReachedMaxGenerations, ReachedMaxEpochs
export generation_size
export random_genome, random_initial_population, next_generation
export EvolutionSpec, evolution_loop
export vns_evolution_loop
export infinitely_bad

"""
    infinitely_bad(sense::SciMLBase.ObjSense)

Return a value that is considered almost infinitely bad in the
given optimization sense.

This returns a finite number of the largest possible magnitude
because infinite float values`Inf` and `-Inf` can't be compared
to any other float value using `<`.
"""
function infinitely_bad(sense::ObjSense; t=Float64)
    if sense == MinSense
        return prevfloat(typemax(t))
    else
        return nextfloat(typemin(t))
    end
end

"""
Parameters for tournament selection.
"""
@kwdef struct SelectionSpec
    "Keep this many high-rated agents from the current generation."
    num_to_keep::Int

    "Create this many new agents to add to the next generation."
    num_to_generate::Int

    """When choosing parents, pick two agents and random, and use
    the one with the highest rating with this probability.
    Otherwise, use the other one."""
    p_take_better::Float64 = 0.6

    """When choosing parents, pick the top-rated individual
    with this probability first, otherwise proceed to tournament
    selection."""
    p_take_very_best::Float64 = 0.0
end

function Base.convert(::Type{SelectionSpec}, s_spec::SelectionSpec)
    return s_spec
end

function Base.convert(::Type{SelectionSpec}, x)
    return SelectionSpec(
        x.num_to_keep,
        x.num_to_generate,
        x.p_take_better,
        x.p_take_very_best)
end

"""
    generation_size(s_spec::SelectionSpec)

Return the size of a generation.
"""
function generation_size(s_spec::SelectionSpec)
    return s_spec.num_to_keep + s_spec.num_to_generate
end

"""
Distribution object for tournament selection.
"""
struct SelectionDist
    spec::SelectionSpec
    d_take_better::Bernoulli
    d_take_very_best::Bernoulli
end

function SelectionDist(s_spec)
    return SelectionDist(
        convert(SelectionSpec, s_spec),
        Bernoulli(s_spec.p_take_better),
        Bernoulli(s_spec.p_take_very_best))
end

"""An agent has a rating, a genome, a parameter vector, and an
extra bit of data that depends on how rating is done."""
struct Agent{R, G, VPar, T}
    rating::R
    genome::G
    parameter::VPar
    extra::T
end

function compile(g_spec, agent)
    return Agent(agent.rating, compile(g_spec, agent.genome), agent.parameter, agent.extra)
end

function short_show(io::IO, a::Agent)
    println(io, "rating = $(a.rating)")
    short_show(io, a.genome)
    println(io, "parameter = $(a.parameter)")
    println(io, "extra = $(a.extra)")
end

"""
    isless(x::Agent, y::Agent)

Compare the `rating` field of the two agents.
This definition allows a population to be sorted.
"""
Base.isless(x::Agent, y::Agent) = isless(x.rating, y.rating)

"""Base type of conditions for a population.  These indicate
whether the population is a work in progress, or why its
evolution was stopped."""
abstract type AbstractPopulationCondition end

struct InitialPopulation <: AbstractPopulationCondition end

struct InProgress <: AbstractPopulationCondition end

struct DiscoveredInnovation <: AbstractPopulationCondition end

struct ReachedStopThreshold <: AbstractPopulationCondition end

struct ReachedMaxGenerations <: AbstractPopulationCondition end

struct ReachedDeadline <: AbstractPopulationCondition end

struct ReachedMaxEpochs <: AbstractPopulationCondition end

struct ReceivedStopMessage <: AbstractPopulationCondition end

"""
    describe_condition(::AbstractPopulationCondition)

Return a short string describing the population condition.
"""
describe_condition(::AbstractPopulationCondition) = "unknown"
describe_condition(::InitialPopulation) = "initial population"
describe_condition(::InProgress) = "in progress"
describe_condition(::ReachedStopThreshold) = "reached stop threshold"
describe_condition(::ReachedMaxGenerations) = "reached max generations"
describe_condition(::ReachedDeadline) = "reached deadline"
describe_condition(::ReachedMaxEpochs) = "reached max epochs"
describe_condition(::ReceivedStopMessage) = "received stop message"

"""
    vns_keep_going(::AbstractPopulationCondition)

Return whether VNS should continue to another epoch.
Utility function.
"""
vns_keep_going(::AbstractPopulationCondition) = true
vns_keep_going(::ReachedDeadline) = false
vns_keep_going(::ReachedMaxEpochs) = false
vns_keep_going(::ReceivedStopMessage) = false
vns_keep_going(::ReachedStopThreshold) = false

"""
A population is a single generation of living `Agent`s.
"""
struct Population
    agents::Vector{Agent}
    condition::AbstractPopulationCondition
end

function short_show(io::IO, p::Population)
    count = 0
    for j in eachindex(p.agents)
        a = p.agents[j]
        println(io, "slot $j:")
        short_show(io, a)
        println(io, "")
        count += 1
        if count >= 10
            println(io, "...")
            break
        end
    end
end

"""
    random_genome(rng::AbstractRNG, g_spec::GenomeSpec, m_spec::MutationSpec, arity_dist::Distribution; domain_safe=false)

Produce a random genome.  It will contain an instruction block
for each mutable slot in the state array as specified by
`g_spec`. Each block will contain one random instruction.  The
operator is picked uniformly at random from
`m_spec.op_inventory`.  The number of operands is drawn from
`arity_dist`.  The operands are drawn uniformly from the set of
possible indices.
If `domain_safe` is `true`, only operators for which `is_domain_safe(op)` is `true` are
used.
"""
function random_genome(rng::AbstractRNG, g_spec::GenomeSpec,
        m_dist::MutationDist, arity_dist::Distribution;
        domain_safe = false)
    index_max = workspace_size(g_spec)
    index_dist = DiscreteUniform(1, index_max)
    num_instruction_blocks = g_spec.output_size + g_spec.scratch_size
    instruction_blocks = Vector(undef, num_instruction_blocks)
    if domain_safe
        d_op = filter_domain_safe_ops(m_dist)
    else
        d_op = m_dist.d_op
    end
    for j in 1:num_instruction_blocks
        op = m_dist.op_inventory[rand(rng, d_op)]
        num_operands = rand(rng, arity_dist)
        operands = rand(rng, index_dist, num_operands)
        instruction_blocks[j] = [Instruction(op, operands)]
    end
    g = Genome(instruction_blocks)
    # cg = compile(g_spec, g)
    return g
end

function filter_domain_safe_ops(m_dist)
    safe_op_ixs = Int[]
    safe_dist = Float64[]
    pv = probs(m_dist.d_op)
    for (i, op) in enumerate(m_dist.op_inventory)
        if is_domain_safe(op)
            push!(safe_op_ixs, i)
            push!(safe_dist, pv[i])
        end
    end
    @assert !isempty(safe_op_ixs)
    safe_dist = safe_dist / sum(safe_dist)  # Normalize probabilities
    return DiscreteNonParametric(safe_op_ixs, safe_dist)
end

"""
Parameters needed to run `random_initial_population` and `evolution_loop`.
"""
@kwdef struct EvolutionSpec
    g_spec::GenomeSpec
    m_dist::MutationDist
    s_dist::SelectionDist
    grow_and_rate::Any
    max_generations::Union{Nothing, Int} = nothing
    stop_on_innovation::Bool = false
end

function EvolutionSpec(
        g_spec::GenomeSpec,
        m_spec::MutationSpec,
        s_spec::SelectionSpec,
        grow_and_rate,
        max_generations::Int,
        stop_on_innovation::Bool = false
)
    src_index_max = workspace_size(g_spec)
    m_dist = MutationDist(m_spec, src_index_max)
    s_dist = SelectionDist(s_spec)
    return EvolutionSpec(
        g_spec,
        m_dist,
        s_dist,
        grow_and_rate,
        max_generations,
        stop_on_innovation
    )
end

function is_better(reference, other, sense)
    if sense == MinSense
        return other < reference
    else
        return other > reference
    end
end

"""
    random_initial_population(
        rng::AbstractRNG,
        g_spec::GenomeSpec,
        m_dist::MutationDist,
        arity_dist::Distribution,
        s_spec::SelectionSpec,
        grow_and_rate;
        domain_safe = false,
        valid_agent_max_attempts = 100,
        sense = MinSense)::Population

Make a random initial population.  The number of genomes is
specified by adding the number of new genomes per generation and
the number of keepers given in `s_spec`.  Other parameters are
passed to `random_genome` to produce random genomes.
The `grow_and_rate` function is called with `rng, g_spec, genome`.

Since growing an agent from a genome may fail, the function
will try to produce a valid agent up to `valid_agent_max_attempts`
times.  After that many failures, it will give up and throw an
error.

If `domain_safe` is `true`, only operators for which `is_domain_safe(op)` 
is `true` are used in the genomes.

The `sense` parameter specifies whether selection should aim to
minimize (`MinSense`) or maximize (`MaxSense`) the rating of the agents.
"""
function random_initial_population(
        rng::AbstractRNG,
        g_spec::GenomeSpec,
        m_dist::MutationDist,
        arity_dist::Distribution,
        s_spec::SelectionSpec,
        grow_and_rate;
        domain_safe = false,
        valid_agent_max_attempts = 100,
        sense = MinSense)::Population
    pop_size = s_spec.num_to_keep + s_spec.num_to_generate
    function make_agent()
        attempt_count = 0
        agent = nothing
        while isnothing(agent) || !isfinite(agent.rating)
            if attempt_count > 0 && mod(attempt_count, 10) == 0
                @debug "random_initial_population: Failed to produce a valid agent after $(attempt_count) attempts; trying again"
            end
            if attempt_count >= valid_agent_max_attempts
                @warn "random_initial_population: Failed to produce a valid agent after $(attempt_count) attempts; giving up"
                return agent
            end
            genome = random_genome(
                rng, g_spec, m_dist, arity_dist; domain_safe = domain_safe)
            try
                agent = grow_and_rate(rng, g_spec, genome)
            catch e
                if isa(e, Union{ArgumentError,SingularException,DomainError})
                    @debug "random_initial_population: attempt to create agent failed with numerical exception, trying again" exception=e
                else
                    rethrow()
                end
            end
            attempt_count += 1
        end
        return agent
    end
    agents = []
    while length(agents) < pop_size
        remaining = pop_size - length(agents)
        more_agents = fetch.([Threads.@spawn make_agent() for i in 1:remaining])
        if isempty(more_agents)
            error("random_initial_population: Failed to produce enough valid agents; giving up")
        end
        append!(agents, more_agents)
    end
    rev = sense == Optimization.MaxSense
    sort!(agents, rev = rev)
    return Population(agents, InitialPopulation())
end

"""
    random_initial_population(
        rng::AbstractRNG,
        e_spec::EvolutionSpec,
        arity_dist::Distribution;
        domain_safe = false,
        sense = MinSense)

Initialize a random initial population from an `e_spec`.
"""
function random_initial_population(
        rng::AbstractRNG,
        e_spec::EvolutionSpec,
        arity_dist::Distribution;
        domain_safe = false,
        sense = MinSense)::Population
    return random_initial_population(
        rng,
        e_spec.g_spec,
        e_spec.m_dist,
        arity_dist,
        e_spec.s_dist.spec,
        e_spec.grow_and_rate;
        domain_safe = domain_safe,
        sense = sense)
end

"""
    pick_parent(rng::AbstractRNG, s_dist::SelectionDist, pop::Population)::Agent

Using tournament selection, pick a parent from the population.
"""
function pick_parent(rng::AbstractRNG, s_dist::SelectionDist, pop::Population)::Agent
    if s_dist.spec.p_take_very_best > 0 && rand(rng, s_dist.d_take_very_best)
        return pop.agents[1]
    else
        cs = rand(rng, pop.agents, 2)
        if rand(rng, s_dist.d_take_better)
            return maximum(cs)
        else
            return minimum(cs)
        end
    end
end

"""
    new_genome(rng::AbstractRNG, s_dist::SelectionDist, m_dist::MutationDist, pop::Population)

Produce a new offspring genome.
Pick two parents using tournament selection.
Recombine their genomes, and apply mutations.
"""
function new_genome(
        rng::AbstractRNG,
        s_dist::SelectionDist,
        m_dist::MutationDist,
        pop::Population
)::Genome
    a1 = pick_parent(rng, s_dist, pop)
    a2 = pick_parent(rng, s_dist, pop)
    gr = recombine(rng, a1.genome, a2.genome)
    # @assert check_genome(gr) "gr"
    grm = mutate(rng, m_dist, gr)
    # @assert check_genome(grm) "grm"
    return grm
end

"""
    next_generation(
        rng::AbstractRNG,
        g_spec::GenomeSpec,
        s_dist::SelectionDist,
        m_dist::MutationDist,
        pop::Population,
        grow_and_rate;
        valid_agent_max_attempts = 100,
        sense = MinSense)

Produce the next generation of a population by selection and
mutation.  The offsprings' genomes are produced by `new_genome`,
and fed to `grow_and_rate(rng, g_spec, genome)`, which should "grow" each
organism and return the rating and extra data as an `Agent`,
which is inserted into the population.  The `sense` parameter
specifies whether selection should minimize or maximize the
rating.

Since growing an agent from a genome may fail, the function
will try to produce a valid agent up to `valid_agent_max_attempts`
times.  After that many failures, it will give up and throw an
error.

The `sense` parameter specifies whether selection should aim to
minimize (`MinSense`) or maximize (`MaxSense`) the rating of the agents.

"""
function next_generation(
        rng::AbstractRNG,
        g_spec::GenomeSpec,
        s_dist::SelectionDist,
        m_dist::MutationDist,
        pop::Population,
        grow_and_rate;
        valid_agent_max_attempts = 100,
        sense = MinSense)::Population
    s_spec = s_dist.spec
    function make_agent()
        attempt_count = 0
        agent = nothing
        while isnothing(agent) || !isfinite(agent.rating)
            if attempt_count > 0 && mod(attempt_count, 10) == 0
                @debug "next_generation: Failed to produce a valid agent after $(attempt_count) attempts; trying again"
            end
            if attempt_count >= valid_agent_max_attempts
                @warn "next_generation: Failed to produce a valid agent after $(attempt_count) attempts; giving up"
                return agent
            end
            ng = new_genome(rng, s_dist, m_dist, pop)
            # cg = compile(g_spec, g)
            try
                agent = grow_and_rate(rng, g_spec, ng)
            catch e
                @debug "$next_generation: grow_and_rate failed with exception, trying again" exception=e
            end
            attempt_count += 1
        end
        return agent
    end
    new_agents = fetch.([Threads.@spawn make_agent() for i in 1:s_spec.num_to_generate])
    to_keep = min(s_spec.num_to_keep, length(pop.agents))
    keepers = pop.agents[1:to_keep]
    next_rated_genomes = vcat(new_agents, keepers)
    rev = sense == Optimization.MaxSense
    sort!(next_rated_genomes, rev = rev)
    return Population(next_rated_genomes, InProgress())
end

"""
    evolution_loop(
        rng::AbstractRNG,
        g_spec::GenomeSpec,
        m_dist::MutationDist,
        s_dist::SelectionDist,
        grow_and_rate,
        pop_init::Population;
        kw_args...)

Advance `pop_init` by `num_generations` and return the new `Population`.

The various spec records and `grow_and_rate` function are passed
to `next_generation()`.

Keyword arguments:

* `sense = MinSense`:
Should be `MinSense` or `MaxSense` to
indicate whether the search should minimize or maximize the
rating.

* `max_generations = nothing`:
If `max_generations` is an integer rather than nothing,
stop after at most that many generations.

* `stop_on_innovation = false`:
If set to `true`, the process stops as soon as
an agent with a better rating than the previous best is
discovered.

* `verbosity = 1`:
Setting `verbosity` to `0` disables all `@info` messages.
Progress is reported as an `@info` log message when an agent with
a better rating than the previous best is discovered.

* `generation_mod = 10`:
Progress is reported as an `@info` log message when the
generation number is a multiple of `generation_mod`.  If this
number is not > 0, this progress report is disabled.

* `stop_threshold = nothing`:
If `stop_threshold` is a number rather than `nothing`, the
loop stops as soon as an agent with a rating better than
`stop_threshold` is discovered.

* `stop_channel = nothing`:
If `stop_channel` is a `Channel` rather than nothing,
the loop checks for a message on `stop_channel` before
calling `next_generation`.
If a message is found, the loop stops.

* `stop_deadline = nothing`:
If `stop_deadline` is a `DateTime` rather than nothing,
the loop checks whether `now() > stop_deadline` before
calling `next_generation`, and if so, the loop stops.

* `valid_agent_max_attempts = 100`:
Passed to `next_generation()`.

"""

function evolution_loop(
        rng::AbstractRNG,
        g_spec::GenomeSpec,
        m_dist::MutationDist,
        s_dist::SelectionDist,
        grow_and_rate,
        pop_init::Population;
        max_generations::Union{Nothing, Integer},
        generation_mod = 10,
        verbosity = 1,
        valid_agent_max_attempts = 100,
        sense = MinSense,
        stop_threshold::Union{Nothing, Number} = nothing,
        stop_on_innovation = false,
        stop_channel::Union{Nothing, Channel} = nothing,
        stop_deadline::Union{Nothing, DateTime} = nothing
)
    pop_next = pop_init
    best_in_gen = pop_next.agents[1]
    best_rating = best_in_gen.rating
    condition = InProgress()
    t = 0
    while true
        if !isnothing(max_generations) && t >= max_generations
            condition = ReachedMaxGenerations()
            break
        end
        if !isnothing(stop_threshold) && is_better(stop_threshold, best_rating, sense)
            condition = ReachedStopThreshold()
            break
        end
        if !isnothing(stop_channel) && isready(stop_channel) && take!(stop_channel)
            condition = ReceivedStopMessage()
            break
        end
        if !isnothing(stop_deadline) && now() > stop_deadline
            condition = ReachedDeadline()
            break
        end
        best_in_gen = pop_next.agents[1]
        if is_better(best_rating, best_in_gen.rating, sense)
            best_rating = best_in_gen.rating
            condition = DiscoveredInnovation()
            if verbosity > 0
                @info "Generation $t, new best = $best_rating"
            end
            if stop_on_innovation
                break
            end
        end
        t += 1
        pop_next = next_generation(
            rng, g_spec, s_dist, m_dist, pop_next, grow_and_rate,
            valid_agent_max_attempts = valid_agent_max_attempts)
        if verbosity > 0 && mod(t, generation_mod) == 0
            @info "Generation $t, best = $best_rating"
        end
    end
    if verbosity > 0
        @info "Stopping: $(describe_condition(condition))"
    end
    return Population(pop_next.agents, condition)
end

"""
    evolution_loop(
    rng::AbstractRNG,
    e_spec::EvolutionSpec
    pop_init::Population;
    kw_args...
    )

Advance `pop_init` and return the new `Population`.

Calls `evolution_loop()` with the specifications unpacked from `e_spec`.

Keyword arguments are passed to the next implementation of
`evolution_loop()`.
"""

function evolution_loop(
        rng::AbstractRNG,
        e_spec::EvolutionSpec,
        pop_init::Population;
        kw_args...
)
    return evolution_loop(
        rng,
        e_spec.g_spec,
        e_spec.m_dist,
        e_spec.s_dist,
        e_spec.grow_and_rate,
        pop_init;
        max_generations = e_spec.max_generations,
        stop_on_innovation = e_spec.stop_on_innovation,
        kw_args...)
end

"""
    vns_evolution_loop(
    rng::AbstractRNG,
    neighborhoods::AbstractArray{EvolutionSpec},
    pop_init::Population;
    kw_args...
    )

Perform variable neighborhood search.

Evolution is performed using `neighborhoods[1]` using
`evolution_loop`.  That's the first epoch.  If a new best rating
is achieved, continue with that neighborhood in the second epoch.
Otherwise, switch to using `neighborhoods[2]` etc.  Once a new
best rating is achieved, return to `neighborhoods[1]`.

Fields from the neighborhoods and relevant keyword arguments are
passed as keyword arguments to `evolution_loop`.

Other keyword arguments:

* `verbosity = 1`:
Set `verbosity` to `0` to disable `@info` messages.
`@info` log messages are
produced after each epoch completes.

* `max_epochs = nothing`:
If `max_epochs` is an integer rather than `nothing`, the loop ends
after at most that many epochs.

* `stop_threshold = nothing`:
If `stop_threshold` is a number rather than `nothing`, the
loop stops as soon as an agent with a rating better than
`stop_threshold` is discovered.

* `stop_channel = nothing`:
If `stop_channel` is a `Channel` rather than nothing,
the loop checks for a message on `stop_channel` before
calling `next_generation`.
If a message is found, the loop stops.

* `stop_deadline = nothing`:
If `stop_deadline` is a `DateTime` rather than nothing,
the loop checks whether `now() > stop_deadline` before
calling `next_generation`, and if so, the loop stops.

* `valid_agent_max_attempts = 100`:
Passed to `next_generation()`.
"""

function vns_evolution_loop(
        rng::AbstractRNG,
        neighborhoods::AbstractArray{EvolutionSpec},
        pop_init::Population;
        max_epochs::Union{Nothing, Integer} = nothing,
        generation_mod = 10,
        verbosity = 1,
        valid_agent_max_attempts = 100,
        sense = MinSense,
        stop_threshold::Union{Nothing, Number} = nothing,
        stop_channel::Union{Nothing, Channel} = nothing,
        stop_deadline::Union{Nothing, DateTime} = nothing
)
    pop_next = pop_init
    neighborhood_index = 1
    condition = InProgress()
    e = 0
    best_in_gen = pop_next.agents[1]
    best_rating = best_in_gen.rating
    while vns_keep_going(condition)
        if !isnothing(max_epochs) && e >= max_epochs
            condition = ReachedMaxEpochs()
            break
        end
        e += 1
        pop_next = evolution_loop(
            rng,
            neighborhoods[neighborhood_index],
            pop_next;
            generation_mod = generation_mod,
            verbosity = verbosity,
            valid_agent_max_attempts = valid_agent_max_attempts,
            sense = sense,
            stop_threshold = stop_threshold,
            stop_channel = stop_channel,
            stop_deadline = stop_deadline
        )
        best_in_gen = pop_next.agents[1]
        condition = pop_next.condition
        if (condition == DiscoveredInnovation()
            ||
            is_better(best_rating, best_in_gen.rating, sense))
            best_rating = best_in_gen.rating
            # Return to neighborhood 1
            neighborhood_index = 1
            if verbosity > 0
                @info "Epoch $e: Discovered innovation, returning to first neighborhood"
            end
        elseif neighborhood_index < length(neighborhoods)
            # Advance to the next neighborhood
            neighborhood_index += 1
            if verbosity > 0
                @info "Epoch $e: Completed with no innovation, advancing to neighborhood $neighborhood_index"
            end
        else
            if verbosity > 0
                @info "Epoch $e: Completed with no innovation, continuing in neighborhood $neighborhood_index"
            end
        end
    end
    if verbosity > 0
        @info "Stopping VNS: $(describe_condition(condition))"
    end
    return Population(pop_next.agents, condition)
end
