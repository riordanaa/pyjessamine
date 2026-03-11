"""
    separate_columns(X)

Convert a `Tables.jl`-compliant object `X` into a vector of column vectors.
If `X` is already a vector, return it unchanged.
"""

function separate_columns(x_table)
    @assert Tables.istable(x_table)
    x_cols = Tables.columns(x_table)
    xs = collect(x_cols[col] for col in Tables.columnnames(x_cols))
    return xs
end

function separate_columns(xs::AbstractVector{<:AbstractVector})
    return xs
end

function separate_columns(xs::AbstractVector{<:Number})
    return xs
end
