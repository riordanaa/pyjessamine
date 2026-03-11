export RandomDuplicateDelete

"""
Wrap an iterator `source` to produce an iterator that will
randomly include some elements yielded by the source twice in a
row, and some not at all.
"""
@kwdef struct RandomDuplicateDelete{Source}
    rng::AbstractRNG

    "Distribution for whether to duplicate an element"
    d_dup::Bernoulli

    "Distribution for whether to delete an element"
    d_del::Bernoulli

    "Source iterator"
    source::Source
end

"""
State structure corresponding to ongoing iteration of `RandomDuplicateDelete`
"""
@kwdef struct RDDState{Etype}
    next_val::Etype
    source_state::Any
end

rdd_state(::Nothing) = nothing
rdd_state((next_val, source_state)) = RDDState(next_val, source_state)

Base.eltype(::Type{RandomDuplicateDelete{Source}}) where {Source} = eltype(Source)

Base.IteratorSize(::RandomDuplicateDelete) = Base.SizeUnknown()

function Base.iterate(rdd::RandomDuplicateDelete)
    return Base.iterate(rdd, rdd_state(Base.iterate(rdd.source)))
end

function Base.iterate(rdd::RandomDuplicateDelete, ::Nothing)
    return nothing
end

function Base.iterate(rdd::RandomDuplicateDelete, rdds::RDDState)
    # If a random trial says duplicate,
    # return next_val and leave the iteration state unchanged.
    if rand(rdd.rng, rdd.d_dup)
        return (rdds.next_val, rdds)
    end
    # If a random trial says delete,
    # skip next_val and advance the source iteration
    if rand(rdd.rng, rdd.d_del)
        return Base.iterate(rdd, rdd_state(Base.iterate(rdd.source, rdds.source_state)))
    end
    # Otherwise, just emit next_val and avance the source iteration
    return (rdds.next_val, rdd_state(Base.iterate(rdd.source, rdds.source_state)))
end
