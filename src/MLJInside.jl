# For using MLJ machinery applied to the output of a genome
export BasicModelMachineSpec
export LinearModelMachineSpec
export sum_sq_coefs

using MLJLinearModels

@kwdef struct BasicModelMachineSpec <: AbstractMachineSpec
    model::Model
    lambda_parameter::Float64
    lambda_operand::Float64
    performance::Any # callable
end

function Base.convert(::Type{BasicModelMachineSpec}, mn_spec::BasicModelMachineSpec)
    return mn_spec
end

function Base.convert(::Type{BasicModelMachineSpec}, x)
    return BasicModelMachineSpec(
        x.model,
        x.lambda_parameter,
        x.lambda_operand,
        x.performance)
end

"""
Wrap an MLJ `Model` that uses a linear combination of input
columns to produce its predictions.
"""
@kwdef struct LinearModelMachineSpec <: AbstractMachineSpec
    model::Model
    lambda_model::Float64
    lambda_parameter::Float64
    lambda_operand::Float64
    performance::Any # callable
end

"""
    LinearModelMachineSpec(model::Model, lambda_parameter, lambda_operand)

Construct a `LinearModelMachineSpec` using `model.lambda` for `lambda_model`.
"""
function LinearModelMachineSpec(
        model::Model,
        lambda_parameter,
        lambda_operand)
    return LinearModelMachineSpec(
        model,
        model.lambda,
        lambda_parameter,
        lambda_operand)
end

function Base.convert(::Type{LinearModelMachineSpec}, mn_spec::LinearModelMachineSpec)
    return mn_spec
end

function Base.convert(::Type{LinearModelMachineSpec}, x)
    lambda_model = if hasfield(typeof(x), :lambda_model)
        x.lambda_model
    elseif hasfield(typeof(x.model), :lambda)
        x.model.lambda
    else
        error("Unable to determine a value for lambda_model")
    end
    return LinearModelMachineSpec(
        x.model,
        lambda_model,
        x.lambda_parameter,
        x.lambda_operand,
        x.performance)
end

"""
    machine_complexity(mn_spec, m)

Numerical complexity of a machine.

Return the sum of squares of `MLJ.fitted_params(m)`
multiplied by `mn_spec.lambda_model`.
"""
function machine_complexity(mn_spec::LinearModelMachineSpec, m)
    params = MLJ.fitted_params(m)

    # Hastie et al exclude the bias from the regularization term.
    # That way, addition of a constant to every given y
    # results in that same constant being added to the prediction.
    #     c += abs(params.intercept)^2
    return mn_spec.lambda_model * sum_sq_coefs(params.coefs)
end

function sum_sq_coefs(coefs::Dict)
    c = 0.0
    for (feature, coef) in coefs
        c += abs(coef)^2
    end
    return c
end

function sum_sq_coefs(coefs::AbstractVector{<:Pair})
    c = 0.0
    for (feature, coef) in coefs
        c += abs(coef)^2
    end
    return c
end

function sum_sq_coefs(coefs::AbstractVector{<:Number})
    return m_spec.lambda_model * dot(coefs, coefs)
end
