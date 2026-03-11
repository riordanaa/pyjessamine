# Make language server happy
if false
    using SymPy
    using PyCall
    include("GenomeCore.jl")
end

export eval_time_step_sympy, show_sympy, run_genome_sympy
export replace_near_integer

"""
    eval_time_step_sympy(g_spec, genome; output_sym, scratch_sym, parameter_sym, input_sym)

Return symbolic objects representing a single time step of `genome`.
The result is a named tuple with fields
`z`, `t`, `p`, `x` for the `SymPy` objects for those variables;
`c` and `c_next` for the current and future cell state in symbolic form.
"""

function eval_time_step_sympy(
        g_spec::GenomeSpec,
        genome::Genome;
        output_sym = :z,
        scratch_sym = :t,
        parameter_sym = :p,
        input_sym = :x,
        assumptions = Dict(:extended_real => true)
)
    z = Sym{PyCall.PyObject}[symbols("$output_sym$j"; assumptions...) for j in 1:(g_spec.output_size)]
    t = Sym{PyCall.PyObject}[symbols("$scratch_sym$j"; assumptions...) for j in 1:(g_spec.scratch_size)]
    p = Sym{PyCall.PyObject}[symbols("$parameter_sym$j"; assumptions...) for j in 1:(g_spec.parameter_size)]
    x = Sym{PyCall.PyObject}[symbols("$input_sym$j"; assumptions...) for j in 1:(g_spec.input_size)]
    c = CellState(z, t, p, x)
    c_next = eval_time_step(c, genome)
    return (z = z, t = t, p = p, x = x, c = c, c_next = c_next)
end

function show_sympy(
        g_spec::GenomeSpec,
        genome::Genome;
        output_sym = :z,
        scratch_sym = :t,
        parameter_sym = :p,
        input_sym = :x,
        assumptions = Dict(:extended_real => true))
    sym_res = eval_time_step_sympy(
        g_spec,
        genome;
        output_sym = output_sym,
        scratch_sym = scratch_sym,
        parameter_sym = parameter_sym,
        input_sym = input_sym,
        assumptions = assumptions)
    return hcat(
        1:length(sym_res.c),
        flat_workspace(sym_res.c),
        flat_workspace(sym_res.c_next))
end

function run_genome_sympy(
        g_spec::GenomeSpec,
        genome::AbstractGenome;
        parameter_sym = :p,
        input_sym = :x,
        assumptions = Dict(:extended_real => true))
    p = Sym{PyCall.PyObject}[symbols("$parameter_sym$j"; assumptions...) for j in 1:(g_spec.parameter_size)]
    x = Sym{PyCall.PyObject}[symbols("$input_sym$j"; assumptions...) for j in 1:(g_spec.input_size)]
    z = run_genome(g_spec, genome, p, x)[end]
    return (p = p, x = x, z = z)
end

function replace_near_integer(expr::Sym; tolerance = 1.0e-10)
    return replace_near_integer_sympy(expr, tolerance)
end

function replace_near_integer_sympy(expr::Sym, tolerance)
    if expr.func.𝑓.__name__ == "Float"
        x = convert(Float64, expr.evalf())
        k = convert(Int64, round(x))
        if k < typemax(Int) && abs(expr - k) < tolerance
            return Sym(k)
        else
            return expr
        end
    elseif isempty(expr.args)
        return expr
    else
        new_args = map(expr.args) do expr
            replace_near_integer_sympy(expr, tolerance)
        end
        return expr.func(new_args...)
    end
end

function replace_near_integer_sympy(expr, tolerance)
    return expr
end
