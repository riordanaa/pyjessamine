# !!! TODO Look into NaNMath.jl which returns NaN instead of throwing an error for things
# like sqrt(negative)

# For vectorization, sometimes we need a blank vector of the appropriate size.
# I'm taking the size and type from workspace[1].
# This assumes the outputs are first in the workspace,
# and that they are expanded to full vectors.  Otherwise,
# we have to go through all the indices and figure out
# how big of a vector to make.

# Make language server happy
if false
    # include("GenomeCore.jl")
end

export splat_or_default
export AbstractUnaryOp, AbstractMultiOp
export Add, Multiply, Subtract
export UnaryComposition
export Reciprocal, ReciprocalMultiply, ReciprocalAdd, ReciprocalSubtract
export Power, WholePower
export FzAnd, FzOr, FzNand, FzNor
export SoftMax, SoftMin
export Maximum, Minimum
export Sign, SignAdd, SignSubtract
export Sigmoid, SigmoidAdd, SigmoidSubtract
export define_unary_op

abstract type AbstractUnaryOp end

abstract type AbstractMultiOp <: AbstractGeneOp end

"""
    is_domain_safe(op::AbstractMultiOp)

Return `true` if the operation has no domain restrictions.
Return `false` if the operation has domain restrictions, such as division, in which the denominator cannot be zero.
The default is to return `false`.
"""

function is_domain_safe(::AbstractMultiOp)
    return false
end

"""
    is_domain_safe(op::AbstractUnaryOp)

Return `true` if the operation has no domain restrictions.
Return `false` if the operation has domain restrictions, such as division, in which the denominator cannot be zero.
The default is to return `false`.
"""

function is_domain_safe(::AbstractUnaryOp)
    return false
end


"""
    splat_or_default(op, def, workspace, indices)

Return the result of applying `op` to the `operands = workspace[indices]`,
with `op([]) = def` and `op([x1...]) = op(x1, op(x2, ...))`.
The `Base.splat` function doesn't have a way to deal with the first case.
The operation should be flat (commutative and associative).
"""

#    return reduce(op, operands, def)


function splat_or_default(op, def, workspace::AbstractVector, indices::AbstractVector{<:Integer})
    if isempty(indices)
        return def
    elseif length(indices) == 1
        return workspace[indices[1]]
    else
        return reduce(op, workspace[indices])
    end
end

"Add operands."
struct Add <: AbstractMultiOp end

short_show(io::IO, ::Add) = print(io, "add")

is_domain_safe(::Add) = true

"""
    op_eval(op::Add, workspace, indices)

Evaluate the `Add` operation on the `workspace` at the given `indices`.
This function adds the elements in `workspace` at the specified `indices`.
"""

op_eval(::Add, workspace, indices) = splat_or_default(.+, 0.0, workspace, indices)

function op_eval(::Add, workspace::AbstractVector{V}, indices::AbstractVector{<:Integer}) where {V<:AbstractVector}
    n = length(indices)
    if n == 0
        return [zero(eltype(V))]
    elseif n == 1
        return workspace[indices[1]]
    else
        dest = similar(workspace[1])
        dest .= workspace[indices[1]]
        for k in indices[2:end]
            dest .+= workspace[k]
        end
        return dest
    end
end

#get(a::AbstractArray, index) = a[index]
#get(x::Number, index::Integer) = x

function op_eval_add_into!(dest::AbstractVector, ::Add, workspace::AbstractVector, indices::AbstractVector{<:Integer})
    for k in indices
        dest .+= workspace[k]
    end
end

function to_expr(::Add, cs, operands)
    if isempty(operands)
        return :(0)
    elseif length(operands) == 1
        field, j = operands[1]
        return :($cs.$field[$j])
    else
        return Expr(:call, :.+, (:($cs.$field[$j]) for (field, j) in operands)...)
    end
end

"Subtract LISP-style: `0 + x[1] - x[2] - x[3]...`"
struct Subtract <: AbstractMultiOp end

short_show(io::IO, ::Subtract) = print(io, "sub")

is_domain_safe(::Subtract) = true

"""
    op_eval(op::Subtract, workspace, indices)

Evaluate the `Subtract` operation on the `workspace` at the given `indices`.
This function returns 0 if the list of indices is empty,
and otherwise computes the result of `workspace[indices[1]] - workspace[indices[2]] - ...`.
"""

function op_eval(::Subtract, workspace, indices)
    n = length(indices)
    if n == 0
        return 0
    elseif n == 1
        return workspace[indices[1]]
    else
        return workspace[indices[1]] .- op_eval(Add(), workspace, indices[2:end])
    end
end

function op_eval(::Subtract, workspace::AbstractVector{V}, indices::AbstractVector{<:Integer}) where {V<:AbstractVector}
    n = length(indices)
    if n == 0
        return [zero(eltype(V))]
    elseif n == 1
        return workspace[indices[1]]
    else
        dest = similar(workspace[1])
        dest .= workspace[indices[1]]
        for k in indices[2:end]
            dest .-= workspace[k]
        end
        return dest
    end
end

function op_eval_add_into!(dest::AbstractVector, ::Subtract, workspace::AbstractVector, indices::AbstractVector{<:Integer})
    if !isempty(indices)
        dest .+= workspace[indices[1]]
        for k in indices[2:end]
            dest .-= workspace[k]
        end
    end
end

function to_expr(::Subtract, cs, operands)
    if isempty(operands)
        return :(0)
    elseif length(operands) == 1
        field, j = operands[1]
        return :($cs.$field[$j])
    else
        field1, j1 = operands[1]
        return Expr(:call, :.-, :($cs.$field1[$j1]),
            Expr(:call, :.+, (:($cs.$field[$j]) for (field, j) in operands[2:end])...))
    end
end

"Multiply operands."
struct Multiply <: AbstractMultiOp end

short_show(io::IO, ::Multiply) = print(io, "mul")

is_domain_safe(::Multiply) = true

"""
    op_eval(op::Multiply, workspace, indices)

Evaluate the `Multiply` operation on the `workspace` at the given `indices`.
This function multiplies the elements in `workspace` at the specified `indices`.
"""
op_eval(::Multiply, workspace, indices) = splat_or_default(.*, 1.0, workspace, indices)

function op_eval(::Multiply, workspace::AbstractVector{V}, indices::AbstractVector{<:Integer}) where {V<:AbstractVector}
    n = length(indices)
    if n == 0
        return [one(eltype(V))]
    elseif n == 1
        return workspace[indices[1]]
    else
        dest = similar(workspace[1])
        dest .= workspace[indices[1]]
        for k in indices[2:end]
            dest .*= workspace[k]
        end
        return dest
    end
end


function to_expr(::Multiply, cs, operands)
    if isempty(operands)
        return :(1)
    elseif length(operands) == 1
        field, j = operands[1]
        return :($cs.$field[$j])
    else
        return Expr(:call, :.*, (:($cs.$field[$j]) for (field, j) in operands)...)
    end
end

"Do a multi-arity operation and apply a unary operation"
@kwdef struct UnaryComposition{Un <: AbstractUnaryOp, Multi <: AbstractMultiOp} <:
              AbstractMultiOp
    unary::Un = Un()
    multi::Multi = Multi()
end

function is_domain_safe(c::UnaryComposition)
    return is_domain_safe(c.unary)
end

function short_show(io::IO, c::UnaryComposition)
    short_show(io, c.unary)
    print(io, "@")
    short_show(io, c.multi)
end

function op_eval(c::UnaryComposition, workspace, indices)
    return un_op_eval(c.unary, op_eval(c.multi, workspace, indices))
end

function to_expr(c::UnaryComposition, cs, operands)
    inner_expr = to_expr(c.multi, cs, operands)
    outer_expr = to_expr(c.unary, inner_expr)
    return outer_expr
end

struct Reciprocal <: AbstractUnaryOp end
short_show(io::IO, ::Reciprocal) = print(io, "rcp")
un_op_eval(::Reciprocal, t) = 1 ./ t
to_expr(::Reciprocal, expr) = :(1 ./ $expr)

"Multiply operands and return the reciprocal."
const ReciprocalMultiply = UnaryComposition{Reciprocal, Multiply}

"Add operands and return the reciprocal."
const ReciprocalAdd = UnaryComposition{Reciprocal, Add}

"Subtract operands and return the reciprocal."
const ReciprocalSubtract = UnaryComposition{Reciprocal, Subtract}

struct Power <: AbstractUnaryOp
    exponent::Float64
end

short_show(io::IO, p::Power) = print(io, "pow(", p.exponent, ")")
is_domain_safe(::Power) = false
un_op_eval(p::Power, t) = t .^ p.exponent
to_expr(p::Power, expr) = :($expr .^ $(p.exponent))

struct WholePower <: AbstractUnaryOp
    exponent::Int
    function WholePower(exponent::Int)
        if exponent < 0
            throw(ArgumentError("WholePower exponent must be non-negative, given $exponent"))
        end
        new(exponent)
    end
end
short_show(io::IO, p::WholePower) = print(io, "wpow(", p.exponent, ")")
is_domain_safe(::WholePower) = true
un_op_eval(p::WholePower, t) = t .^ p.exponent
to_expr(p::WholePower, expr) = :($expr .^ $(p.exponent))

"Return fuzzy AND of the operands"
struct FzAnd <: AbstractMultiOp end

short_show(io::IO, ::FzAnd) = print(io, "fzAnd")

is_domain_safe(::FzAnd) = true

"""
    op_eval(op::FzAnd, workspace, indices)

Evaluate the fuzzy AND operation on the `workspace` at the given `indices`.
"""
op_eval(::FzAnd, workspace, indices) = splat_or_default(.*, 1.0, workspace, indices)

function to_expr(::FzAnd, cs, operands)
    if isempty(operands)
        return :1.0
    elseif length(operands) == 1
        field, j = operands[1]
        return :($cs.$field[$j])
    else
        return Expr(:call, :.*, (:($cs.$field[$j]) for (field, j) in operands)...)
    end
end

"Return fuzzy OR of the operands."
struct FzOr <: AbstractMultiOp end

short_show(io::IO, ::FzOr) = print(io, "fzOr")

is_domain_safe(::FzOr) = true

"""
    op_eval(op::FzOr, workspace, indices)

Evaluate the fuzzy OR operation on the `workspace` at the given `indices`.
"""
function op_eval(::FzOr, workspace, indices)
    return 1.0 .- op_eval(FzNor(), workspace, indices)
end

function to_expr(::FzOr, cs, operands)
    return Expr(:call, :.-, 1.0, to_expr(FzNor(), cs, operands))
end

"Return fuzzy NAND of the operands."
struct FzNand <: AbstractMultiOp end

short_show(io::IO, ::FzNand) = print(io, "fzNand")

is_domain_safe(::FzNand) = true

"""
    op_eval(op::FzNand, workspace, indices)

Evaluate the fuzzy NAND operation on the `workspace` at the given `indices`.
"""
function op_eval(::FzNand, workspace, indices)
    return 1.0 .- op_eval(FzAnd(), workspace, indices)
end

function to_expr(::FzNand, cs, operands)
    return Expr(:call, :.-, 1.0, to_expr(FzAnd(), cs, operands))
end

"Return fuzzy NOR of the operands."
struct FzNor <: AbstractMultiOp end

short_show(io::IO, ::FzNor) = print(io, "fzNor")

is_domain_safe(::FzNor) = true

"""
    op_eval(op::FzNor, workspace, indices)

    Evaluate the fuzzy NOR operation on the `workspace` at the given `indices`.
"""
function op_eval(::FzNor, workspace, indices)
    return splat_or_default((acc, x) -> acc .* (1.0 .- x), 1.0, workspace, indices)
end

function to_expr(::FzNor, cs, operands)
    if isempty(operands)
        return :1.0
    elseif length(operands) == 1
        field, j = operands[1]
        return :(1.0 .- $cs.$field[$j])
    else
        return Expr(:call, :.*, (:(1.0 .- $cs.$field[$j]) for (field, j) in operands)...)
    end
end

"Return the sign of the operand"
struct Sign <: AbstractUnaryOp end
short_show(io::IO, ::Sign) = print(io, "sign")
is_domain_safe(::Sign) = true
un_op_eval(::Sign, t) = sign.(t)
to_expr(::Sign, expr) = :(sign.($expr))

"Return the sign of the sum of the operands"
const SignAdd = UnaryComposition{Sign, Add}

"Return the sign of the difference of the operands"
const SignSubtract = UnaryComposition{Sign, Subtract}

"Apply the exponential sigmoid to the operand"
struct Sigmoid <: AbstractUnaryOp end
short_show(io::IO, ::Sigmoid) = print(io, "sigmoid")
is_domain_safe(::Sigmoid) = true
sigmoid(t) = 1.0 / (1.0 + exp(-t))
# un_op_eval(::Sigmoid, t) = 1 ./ (1 .+ exp.(-t))
un_op_eval(::Sigmoid, t) = sigmoid.(t)
to_expr(::Sigmoid, expr) = :(1.0 ./ (1.0 .+ exp.(-$expr)))

"Apply the exponential sigmoid to the sum of the operands"
const SigmoidAdd = UnaryComposition{Sigmoid, Add}

"Apply the exponential sigmoid to the difference of the operands"
const SigmoidSubtract = UnaryComposition{Sigmoid, Subtract}

# TODO The SoftMax and SoftMin will allocate a lot of temporary
# arrays as written.  Re-write them to the form softmax.(...)
# somehow.

"Apply the soft-max function to the operands"
struct SoftMax <: AbstractMultiOp end
short_show(io::IO, ::SoftMax) = print(io, "softmax")
is_domain_safe(::SoftMax) = true

function op_eval(::SoftMax, workspace, indices::AbstractVector{<:Integer})
    if isempty(indices)
        return Inf
    elseif length(indices) == 1
        return workspace[indices[1]]
    else
        return log.(mapreduce(v->exp.(v), (.+), (workspace[indices])))
    end
end

function to_expr(::SoftMax, cs, operands)
    if isempty(operands)
        return :(-Inf)
    elseif length(operands) == 1
        field, j = operands[1]
        return :($cs.$field[$j])
    else
        return quote
            log.(mapreduce(v->exp.(v), (.+), ($((:($cs.$field[$j]) for (field, j) in operands)...))))
        end
    end
end

"Apply the soft-min function to the operands"
struct SoftMin <: AbstractMultiOp end
short_show(io::IO, ::SoftMin) = print(io, "softmin")
is_domain_safe(::SoftMin) = true

function op_eval(::SoftMin, workspace, indices::AbstractVector{<:Integer})
    if isempty(indices)
        return Inf
    elseif length(indices) == 1
        return workspace[indices[1]]
    else
        return -log.(mapreduce(v->exp.(-v), (.+), (workspace[indices])))
    end
end

function to_expr(::SoftMin, cs, operands)
    if isempty(operands)
        return :(Inf)
    elseif length(operands) == 1
        field, j = operands[1]
        return :($cs.$field[$j])
    else
        return quote
            -log.(mapreduce(v->exp.(-v), (.+), ($((:($cs.$field[$j]) for (field, j) in operands)...))))
        end
    end
end


"Return the maximum of the operands."
struct Maximum <: AbstractMultiOp end

short_show(io::IO, ::Maximum) = print(io, "max")

is_domain_safe(::Maximum) = true

function op_eval(::Maximum, workspace, indices::AbstractVector{<:Integer})
    if isempty(indices)
        return -Inf
    elseif length(indices) == 1
        return workspace[indices[1]]
    else
        # This can be sped up...
        return max.(workspace[indices]...)
    end
end

function to_expr(::Maximum, cs, operands)
    if isempty(operands)
        return :(-Inf)
    elseif length(operands) == 1
        field, j = operands[1]
        return :($cs.$field[$j])
    else
        return quote
            max.($((:($cs.$field[$j]) for (field, j) in operands)...))
        end
    end
end

"Return the minimum of the operands."
struct Minimum <: AbstractMultiOp end

short_show(io::IO, ::Minimum) = print(io, "min")

is_domain_safe(::Minimum) = true

function op_eval(::Minimum, workspace, indices::AbstractVector{<:Integer})
    if isempty(indices)
        return -Inf
    elseif length(indices) == 1
        return workspace[indices[1]]
    else
        # This can be sped up...
        return min.(workspace[indices]...)
    end
end

function to_expr(::Minimum, cs, operands)
    if isempty(operands)
        return :(Inf)
    elseif length(operands) == 1
        field, j = operands[1]
        return :($cs.$field[$j])
    else
        return quote
            min.($((:($cs.$field[$j]) for (field, j) in operands)...))
        end
    end
end

macro define_unary_op(struct_name, function_name)
    return quote
        export $struct_name
        export $(Symbol(string(struct_name) * "Add"))
        export $(Symbol(string(struct_name) * "Subtract"))
        export $(Symbol(string(struct_name) * "Multiply"))
        @doc "Return "*string($function_name)*" of the operand"
        struct $struct_name <: AbstractUnaryOp end
        global short_show
        short_show(io::IO, ::($struct_name)) = print(io, string($function_name))
        global un_op_eval
        un_op_eval(::($struct_name), t) = ($function_name).(t)
        global to_expr
        to_expr(::($struct_name), expr) = :($($function_name).($expr))
        @doc "Return "*string($function_name)*" applied to the sum of the operands"
        const $(Symbol(string(struct_name) * "Add")) = UnaryComposition{$struct_name, Add}
        @doc "Return "*string($function_name)*" applied to the result of subtraction of the operands"
        const $(Symbol(string(struct_name) * "Subtract")) = UnaryComposition{
            $struct_name, Subtract}
        @doc "Return "*string($function_name)*" applied to the product of the operands"
        const $(Symbol(string(struct_name) * "Multiply")) = UnaryComposition{
            $struct_name, Multiply}
    end
end

@define_unary_op Sqrt sqrt
@define_unary_op Exp exp
is_domain_safe(::Exp) = true
@define_unary_op Log log

@define_unary_op Sin sin
is_domain_safe(::Sin) = true
@define_unary_op Cos cos
is_domain_safe(::Cos) = true
@define_unary_op Tan tan
@define_unary_op Cot cot
@define_unary_op Sec sec
@define_unary_op Csc csc

@define_unary_op ASin asin
@define_unary_op ACos acos
@define_unary_op ATan atan
is_domain_safe(::ATan) = true
@define_unary_op ACot acot
is_domain_safe(::ACot) = true
@define_unary_op ASec asec
@define_unary_op ACsc acsc

@define_unary_op Sinh sinh
is_domain_safe(::Sinh) = true
@define_unary_op Cosh cosh
is_domain_safe(::Cosh) = true
@define_unary_op Tanh tanh
is_domain_safe(::Tanh) = true
@define_unary_op Coth coth
@define_unary_op Sech sech
is_domain_safe(::Sech) = true
@define_unary_op Csch csch
@define_unary_op ASinh asinh
is_domain_safe(::ASinh) = true
@define_unary_op ACosh acosh
@define_unary_op ATanh atanh
@define_unary_op ACoth acoth
@define_unary_op ASech asech
@define_unary_op ACsch acsch

export PolynomialInventory
const PolynomialInventory = [Add(), Subtract(), Multiply()]

export RationalFunctionInventory
const RationalFunctionInventory = vcat(
    PolynomialInventory, [ReciprocalAdd(), ReciprocalSubtract(), ReciprocalMultiply()])

export ExpLogInventory
const ExpLogInventory = vcat(RationalFunctionInventory,
    reshape(
        [UnaryComposition{un_op, bin_op}()
         for un_op in [Sqrt, Exp, Log], bin_op in [Add, Subtract, Multiply]],
        :))

export TrigInventory
const TrigInventory = vcat(ExpLogInventory,
    reshape(
        [UnaryComposition{un_op, bin_op}()
         for un_op in [Sin, Cos, Tan, Cot, Sec, Csc, ASin, ACos, ATan, ACot, ASec, ACsc],
            bin_op in [Add, Subtract, Multiply]],
        :))

export HyperbolicInventory
const HyperbolicInventory = vcat(ExpLogInventory,
        reshape(
            [UnaryComposition{un_op, bin_op}()
             for un_op in [Sinh, Cosh, Tanh, Coth, Sech, Csch, ASinh, ACosh, ATanh, ACoth, ASech, ACsch],
                bin_op in [Add, Subtract, Multiply]],
            :))
