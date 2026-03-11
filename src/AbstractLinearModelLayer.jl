# Make language server happy
if false
    using Symbolics
    using SymPy
    include("SymPyForm.jl")
end

export AbstractLinearModelResult
export BasicLinearModelResult, coefficients, intercept

export extend_if_singleton

"""
Abstract base type for the results of fitting linear models.
"""
abstract type AbstractLinearModelResult <: AbstractModelResult end

"""
A `BasicLinearModelResult` has a vector of coefficients
plust a bias or intercept.

"""
@kwdef struct BasicLinearModelResult <: AbstractLinearModelResult
    coefficients::Vector{Float64}
    intercept::Float64
end

BasicLinearModelResult(b) = BasicLinearModelResult(b, 0.0)

function coefficients(r::BasicLinearModelResult)
    return r.coefficients
end

function intercept(r::BasicLinearModelResult)
    return r.intercept
end

function model_basic_symbolic_output(
        g_spec::GenomeSpec,
        genome::AbstractGenome,
        parameter::AbstractArray,
        mr::AbstractLinearModelResult
)
    p, x, z = run_genome_symbolic(g_spec, genome)
    # This includes `b[1] =` $b_0$ as an intercept or bias term.
    # Which throws off the numbering.
    b = Symbolics.variables(:b, 0:(g_spec.output_size))
    y_sym = dot(z, b[2:end]) + b[1]
    p_subs = Dict(p[j] => parameter[j] for j in eachindex(p))
    b_num = coefficients(mr)
    b_subs = Dict(b[1 + j] => b_num[j] for j in eachindex(b_num))
    # This is really $b_0$:
    b_subs[b[1]] = intercept(mr)
    y_num = Symbolics.substitute(y_sym, merge(p_subs, b_subs))
    return (p = p, x = x, z = z, b = b, p_subs = p_subs, b_subs = b_subs, y_sym = y_sym,
        y_num = y_num)
end

function model_symbolic_output(
        g_spec::GenomeSpec,
        genome::AbstractGenome,
        parameter::AbstractArray,
        mr::AbstractLinearModelResult
)
    p, x, z = run_genome_symbolic(g_spec, genome)
    # This includes `b[1] =` $b_0$ as an intercept or bias term.
    # Which throws off the numbering.
    b = Symbolics.variables(:b, 0:(g_spec.output_size))
    y_sym = dot(z, b[2:end]) + b[1]
    used_vars = Set(v.name for v in Symbolics.get_variables(y_sym))
    # To handle rational functions that have things like 1/(x/0),
    # replace Inf with W and do a limit as W -> Inf.
    # First, grind through and make sure we have a unique symbol.
    j = 0
    local W
    while true
        W = Symbolics.variable(:W, j)
        if !(W in used_vars)
            break
        end
        j += 1
    end
    y_W = Symbolics.substitute(y_sym, Dict([Inf => W]))
    y_lim = Symbolics.limit(y_W.val, W.val, Inf)
    y_simp = Symbolics.simplify(y_lim)
    p_subs = Dict(p[j] => parameter[j] for j in eachindex(p))
    b_num = coefficients(mr)
    b_subs = Dict(b[1 + j] => b_num[j] for j in eachindex(b_num))
    # This is really $b_0$:
    b_subs[b[1]] = intercept(mr)
    y_sub = Symbolics.substitute(y_simp, merge(p_subs, b_subs))
    y_num = Symbolics.simplify(y_sub)
    return (p = p, x = x, z = z, b = b, p_subs = p_subs, b_subs = b_subs, y_sym = y_sym,
        y_lim = y_lim, y_simp = y_simp, y_sub = y_sub, y_num = y_num)
end

function model_basic_sympy_output(
        g_spec::GenomeSpec,
        genome::AbstractGenome,
        parameter::AbstractArray,
        mr::AbstractLinearModelResult;
        assumptions = Dict(:extended_real => true)
)
    p, x, z = run_genome_sympy(g_spec, genome; assumptions = assumptions)
    # This includes `b[1] =` $b_0$ as an intercept or bias term.
    # Which throws off the numbering.
    b = [symbols("b$j"; assumptions...) for j in 0:(g_spec.output_size)]
    # This has to be dot(b[..], z) because SymPy computes dot products
    # as u dot v = ajoint(u) v, and you end up with stray complex conjugates
    # all over everything.
    y_sym = dot(b[2:end], z) + b[1]
    p_subs = Dict(p[j] => parameter[j] for j in eachindex(p))
    b_num = coefficients(mr)
    b_subs = Dict(b[1 + j] => b_num[j] for j in eachindex(b_num))
    # This is really $b_0$:
    b_subs[b[1]] = intercept(mr)
    y_num = y_sym(merge(p_subs, b_subs))
    return (p = p, x = x, z = z, b = b, p_subs = p_subs, b_subs = b_subs, y_sym = y_sym,
        y_num = y_num)
end

function model_sympy_output(
        g_spec::GenomeSpec,
        genome::AbstractGenome,
        parameter::AbstractArray,
        mr::AbstractLinearModelResult;
        assumptions = Dict(:extended_real => true)
)
    p, x, z = run_genome_sympy(g_spec, genome; assumptions = assumptions)
    # This includes `b[1] =` $b_0$ as an intercept or bias term.
    # Which throws off the numbering.
    b = [symbols("b$j"; assumptions...) for j in 0:(g_spec.output_size)]
    # This has to be dot(b[..], z) because SymPy computes dot products
    # as u dot v = ajoint(u) v, and you end up with stray complex conjugates
    # all over everything.
    y_sym = dot(b[2:end], z) + b[1]
    used_vars = y_sym.free_symbols
    # To handle rational functions that have things like 1/(x/0),
    # replace Inf with W and do a limit as W -> Inf.
    # First, grind through and make sure we have a unique symbol.
    j = 0
    local W
    while true
        W = symbols("W$j"; assumptions...)
        if !(W in used_vars)
            break
        end
        j += 1
    end
    # Replace all infinities with W.  The complex infinited zoo
    # is not really correct but this is the best we can do at
    # this point.
    y_W = y_sym.subs(sympy.oo, W).subs(-sympy.oo, -W).subs(sympy.zoo, W)
    y_lim = y_W.limit(W, sympy.oo)
    y_simp = SymPy.simplify(y_lim)
    p_subs = Dict(p[j] => parameter[j] for j in eachindex(p))
    b_num = coefficients(mr)
    b_subs = Dict(b[1 + j] => b_num[j] for j in eachindex(b_num))
    # This is really $b_0$:
    b_subs[b[1]] = intercept(mr)
    y_sub = y_simp(merge(p_subs, b_subs))
    y_num = SymPy.simplify(y_sub)
    return (p = p, x = x, z = z, b = b, p_subs = p_subs, b_subs = b_subs, y_sym = y_sym,
        y_lim = y_lim, y_simp = y_simp, y_sub = y_sub, y_num = y_num)
end

"""
    model_predict(mr::AbstractLinearModelResult, xs::AbstractArray{<:AbstractArray}; kw_args...)

Stack the columns `xs` into a matrix `X` and call `model_predict`.
The `kw_args` are splatted in.
"""
function model_predict(
        mr::AbstractLinearModelResult, xs::AbstractVector{<:AbstractVector}; kw_args...)
    X = stack(xs)
    return model_predict(mr, X; kw_args...)
end

function model_predict(
        mr::AbstractLinearModelResult, xs::NamedTuple; kw_args...)
    return model_predict(mr, fieldvalues(xs); kw_args...)
end

function model_predict(
        mr::AbstractLinearModelResult, xs::Tuple; kw_args...)
    return model_predict(mr, collect(xs); kw_args...)
end


"""
    model_predict(lmr::AbstractLinearModelResult, X::AbstractMatrix)

Return `X * coefficients(lmr) + intercept(lmr)`
"""
function model_predict(lmr::AbstractLinearModelResult, X::AbstractMatrix)
    return X * coefficients(lmr) .+ intercept(lmr)
end

"""
    model_predict(lmr::AbstractLinearModelResult, x::AbstractArray; kw_args...)

Return `dot(x, coefficients(lmr)) + intercept(lmr)`.
"""

function model_predict(
        lmr::AbstractLinearModelResult, x::AbstractArray;
        kw_args...
)
    # This is a bit of a hack to get the right type.
    # We need to convert the array to a matrix.
    return dot(x, coefficients(lmr)) + intercept(lmr)
end

"""
    extend_if_singleton(v::AbstractArray, m::Int)

If `v` is a singleton, as in `v = [x]`,
return `[x, x, ...]`, that is, a vector of `m` copies of `x`.
Otherwise, assert that `length(v) == m` and return `v`.
So the result is always a vector of length `m`.
"""
function extend_if_singleton(v::AbstractArray, m::Int)
    if length(v) == 1
        return fill(v[1], m)
    else
        @assert length(v)==m "m = $m, v = $v"
        return v
    end
end

function extend_if_singleton(v::AbstractArray, shape::Tuple{Int})
    (m,) = shape
    return extend_if_singleton(v, m)
end
