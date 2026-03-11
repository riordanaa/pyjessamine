# Make language server happy
if false
    include("Operations.jl")
end

export MutationSpec, MutationDist
export mutate

"""
A collection of parameters specifying the probabilities of various mutations.
"""
@kwdef struct MutationSpec
    "Probability that an operation changes"
    p_mutate_op::Float64 = 0.1

    "Probability that an index changes"
    p_mutate_index::Float64 = 0.1

    "Probability that an index is duplicated"
    p_duplicate_index::Float64 = 0.01

    "Probability that an index is deleted"
    p_delete_index::Float64 = 0.01

    "Probability that an instruction is duplicated"
    p_duplicate_instruction::Float64 = 0.001

    "Probability that an instruction is deleted"
    p_delete_instruction::Float64 = 0.001

    "Probability that an instruction hops to another block"
    p_hop_instruction::Float64 = 0.0

    "Which operators are available"
    op_inventory::Vector{AbstractGeneOp} = [Add(), Subtract(), Multiply()]

    "Probability of choosing each operator; `nothing` means choose uniformly"
    op_probabilities::Union{Vector{Float64}, Nothing} = nothing
end

function Base.convert(::Type{MutationSpec}, m_spec::MutationSpec)
    return m_spec
end

function Base.convert(::Type{MutationSpec}, x)
    return MutationSpec(
        x.p_mutate_op,
        x.p_mutate_index,
        x.p_duplicate_index,
        x.p_delete_index,
        x.p_duplicate_instruction,
        x.p_delete_instruction,
        x.p_hop_instruction,
        x.op_inventory,
        x.op_probabilities)
end

"""
A collection of distribution objects built from a `MutationSpec` and used to produce random numbers determining mutations.
"""
@kwdef struct MutationDist
    d_mutate_op::Bernoulli
    d_mutate_index::Bernoulli
    d_new_src_index::DiscreteUniform
    d_duplicate_index::Bernoulli
    d_delete_index::Bernoulli
    d_duplicate_instruction::Bernoulli
    d_delete_instruction::Bernoulli
    d_hop_instruction::Bernoulli
    d_op::DiscreteNonParametric
    op_inventory::Vector{AbstractGeneOp}
end

"""
    MutationDist(m_spec::MutationSpec, src_index_max::Int)

Use `m_spec` to build a `MutationDist`.
Specify that an operand index must be <= `src_index_max`.
"""
function MutationDist(m_spec::MutationSpec, src_index_max::Int)
    if isnothing(m_spec.op_probabilities)
        p = 1.0 / length(m_spec.op_inventory)
        op_probabilities = fill(p, length(m_spec.op_inventory))
    else
        op_probabilities = m_spec.op_probabilities
    end
    # DiscreteNonParametric requires the values of possibilities to be
    # numerical, so I have to set this up to produce a random index
    # into the op_inventory.
    return MutationDist(
        Bernoulli(m_spec.p_mutate_op),
        Bernoulli(m_spec.p_mutate_index),
        DiscreteUniform(1, src_index_max),
        Bernoulli(m_spec.p_duplicate_index),
        Bernoulli(m_spec.p_delete_index),
        Bernoulli(m_spec.p_duplicate_instruction),
        Bernoulli(m_spec.p_delete_instruction),
        Bernoulli(m_spec.p_hop_instruction),
        DiscreteNonParametric(1:length(op_probabilities), op_probabilities),
        m_spec.op_inventory
    )
end

"""
    mutate(rng::AbstractRNG, m_dist::MutationDist, x)

Randomly change `x`, using random numbers supplied by RNG and
the distributions specified by `m_dist`.
"""
function mutate end

function mutate(rng::AbstractRNG, m_dist::MutationDist, j::Int)
    if rand(rng, m_dist.d_mutate_index)
        return rand(rng, m_dist.d_new_src_index)
    else
        return j
    end
end

function mutate(rng::AbstractRNG, m_dist::MutationDist, op::AbstractGeneOp)
    if rand(rng, m_dist.d_mutate_op)
        k = rand(rng, m_dist.d_op)
        m_dist.op_inventory[k]
    else
        op
    end
end

function mutate(rng::AbstractRNG, m_dist::MutationDist, inst::Instruction)
    op = mutate(rng, m_dist, inst.op)
    if isempty(inst.operand_ixs)
        return Instruction(op, inst.operand_ixs)
    else
        rdd = RandomDuplicateDelete(
            rng,
            m_dist.d_duplicate_index,
            m_dist.d_delete_index,
            inst.operand_ixs)
        new_operand_ixs = mutate(rng, m_dist, collect(rdd))
        return Instruction(op, new_operand_ixs)
    end
end

function mutate(rng::AbstractRNG, m_dist::MutationDist, v::AbstractArray)
    return map(x -> mutate(rng, m_dist, x), v)
end

function mutate(rng::AbstractRNG, m_dist::MutationDist, g::Genome)
    instruction_blocks = map(g.instruction_blocks) do block
        if isempty(block)
            return []
        else
            new_block = []
            rdd = RandomDuplicateDelete(
                rng,
                m_dist.d_duplicate_instruction,
                m_dist.d_delete_instruction,
                block)
            new_block = collect(rdd)
            # if isempty(new_block)
            #     new_block = [rand(rng, block)]
            # end
            # @assert !isempty(new_block)
            return mutate(rng, m_dist, new_block)
        end
    end
    # Hop mutations are tricker
    for b in eachindex(instruction_blocks)
        block = instruction_blocks[b]
        adjustment = 0
        for j in eachindex(block)
            if rand(rng, m_dist.d_hop_instruction)
                dest = rand(rng, instruction_blocks)
                k = j - adjustment
                instr = block[k]
                push!(dest, instr)
                deleteat!(block, k)
                adjustment += 1
            end
        end
    end
    return Genome(instruction_blocks)
end
