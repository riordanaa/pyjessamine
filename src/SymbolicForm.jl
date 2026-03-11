# Make language server happy
if false
    using Symbolics
    using TermInterface
end

export eval_time_step_symbolic, show_symbolic, run_genome_symbolic
export replace_near_integer

"""
    eval_time_step_symbolic(g_spec, genome; output_sym, scratch_sym, parameter_sym, input_sym)

Return symbolic objects representing a single time step of `genome`.
The result is a named tuple with fields
`z`, `t`, `p`, `x` for the `Symbolics` objects for those variables;
`c` and `c_next` for the current and future cell state in symbolic form.
"""
function eval_time_step_symbolic(g_spec::GenomeSpec, genome::Genome;
        output_sym = :z,
        scratch_sym = :t,
        parameter_sym = :p,
        input_sym = :x)
    z = Symbolics.variables(output_sym, 1:(g_spec.output_size))
    t = Symbolics.variables(scratch_sym, 1:(g_spec.scratch_size))
    p = Symbolics.variables(parameter_sym, 1:(g_spec.parameter_size))
    x = Symbolics.variables(input_sym, 1:(g_spec.input_size))
    c = CellState(z, t, p, x)
    c_next = eval_time_step(c, genome)
    return (z = z, t = t, p = p, x = x, c = c, c_next = c_next)
end

"""
    show_symbolic(g_spec, genome; output_sym, scratch_sym, parameter_sym, input_sym)

Return a matrix with three columns.
The first is row indices 1, 2, etc.
The second is the initial workspace vector in symbolic form.
The third is the future workspace state in symbolic form.
"""
function show_symbolic(g_spec::GenomeSpec, genome::Genome;
        output_sym = :z,
        scratch_sym = :t,
        parameter_sym = :p,
        input_sym = :x)
    z, t, p, x, c, c_next = eval_time_step_symbolic(g_spec, genome;
        output_sym = output_sym,
        scratch_sym = scratch_sym,
        parameter_sym = parameter_sym,
        input_sym = input_sym)
    return hcat(1:length(c), flat_workspace(c), flat_workspace(c_next))
end

"""
    run_genome_symbolic(g_spec, genome; parameter_sym=:p, input_sym=:x)

Build a symbolic form for the output of the final time step of
running `genome`.  The parameter vector and input vector are
`Symbolics` objects of the form `p[j]` and `x[j]`.  The variable
names can be specified with the keyword arguments.

Return a named tuple `(p, x, z)` where `p` and `x`,
are vectors of `Symbolics` objects used to represent
genome parameters and inputs; and `w` is
a vector of genome outputs in symbolic form.
"""
function run_genome_symbolic(
        g_spec::GenomeSpec,
        genome::AbstractGenome;
        parameter_sym = :p,
        input_sym = :x)
    p = Symbolics.variables(parameter_sym, 1:(g_spec.parameter_size))
    x = Symbolics.variables(input_sym, 1:(g_spec.input_size))

    z = run_genome_to_last(g_spec, genome, p, x)
    return (p = p, x = x, z = z)
end

"""
    replace_near_integer(expr; tolerance=1.0e-10)

Round literal floating-point numbers within a symbolic expression.
Specifically, if a number `a` differs `round(a)` by less thant `tolerance`,
it gets replaced by `round(a)`.
"""
function replace_near_integer(expr::SymbolicUtils.BasicSymbolic; tolerance = 1.0e-10)
    if TermInterface.iscall(expr)
        op = TermInterface.operation(expr)
        args = TermInterface.arguments(expr)
        new_args = map(arg -> replace_near_integer(arg, tolerance = tolerance), args)
        return TermInterface.maketerm(typeof(expr), op, new_args, TermInterface.metadata(expr))
    else  # This should handle symbols and other atomic expressions
        return expr
    end
end

function replace_near_integer(expr::Num; tolerance = 1.0e-10)
    return Num(replace_near_integer(Symbolics.value(expr); tolerance = tolerance))
end

function replace_near_integer(expr::Number; tolerance = 1.0e-10)
    k = round(expr)
    if k < typemax(Int) && abs(expr - k) < tolerance
        return convert(Num, convert(Int, k))
    else
        return expr
    end
end
