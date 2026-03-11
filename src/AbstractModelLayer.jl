export AbstractModelResult
export model_predict
export model_basic_symbolic_output, model_symbolic_output
export model_basic_sympy_output, model_sympy_output

"""
Abstract base type for the results of fitting a model
to data.  This object can then be used for predictions.
"""
abstract type AbstractModelResult end

"""
    model_predict(mr::AbstractModelResult, X; kw_args...)

Given the table `X` where the columns are inputs (predictors)
and the rows are points, use `mr` to predict the target value
for all the points.
"""
function model_predict end

"""
    model_predict(g_spec::GenomeSpec, agent::Agent, xs; kw_args...)

Run `agent.genome` on inputs `xs` and `agent.parameter`, and
form the linear combination of the genome's
outputs using the coefficients `agent.extra`.
"""
function model_predict(
        g_spec::GenomeSpec,
        agent::Agent{<:Number, <:AbstractGenome, <:AbstractArray, <:AbstractModelResult},
        x_table;
        kw_args...
)
    xs = separate_columns(x_table)
    num_rows = length(xs[1])
    last_round = run_genome_to_last(g_spec, agent.genome, agent.parameter, xs)
    data_cols = map(u -> extend_if_singleton(u, num_rows), last_round)
    output_col_names = map(1:(g_spec.output_size)) do t
        "z$t"
    end
    Z_table = namedtuple(output_col_names, data_cols)
    return model_predict(agent.extra, Z_table; kw_args...)
end

"""
    model_basic_symbolic_output(g_spec, genome, parameter, model_result; kw_args...)

Build a Symbolics form for the output of the final time step of
running `genome`
Then use the `parameter` vector,
the symbolic output of the genome,
and feed the symbolic output of the genome as input to
model result in `agent.extra` to make a prediction
in symbolic form.
No simplifications are performed.

The `kw_args` are eventually splatted into `model_predict`.

Return a named tuple with lots of useful fields.

"""
function model_basic_symbolic_output(
        g_spec::GenomeSpec,
        genome::AbstractGenome,
        parameter::AbstractArray,
        mr::AbstractModelResult;
        kw_args...)
    p, x, z = run_genome_symbolic(g_spec, genome)
    p_subs = Dict(p[j] => parameter[j] for j in eachindex(p))
    z_num = map(z) do zj
        substitute(zj, p_subs)
    end
    try
        z_sym_row_mat = reshape(z, 1, :)
        y_sym = model_predict(mr, z_sym_row_mat; kw_args...)[1]
        y_sub = substitute(y_sym, p_subs)
        y_num = Symbolics.simplify(y_sub)

        return (p = p, x = x, z = z, p_subs = p_subs, z_num = z_num,
            y_sym = y_sym,
            y_sub = y_sub, y_num = y_num)
    catch e
        @info "model_predict failed, returning short symbolic report: $(e)"
        return (p = p, x = x, z = z, p_subs = p_subs, z_num = z_num)
    end
end

"""
    model_basic_symbolic_output(g_spec, agent; kw_args...)

Build a Symbolics form for the output of the final time step of
running `agent`'s `genome`
Then use the `agent`'s `parameter` vector,
and feed the symbolic output of the genome as input to
model result in `agent.extra` to make a prediction
in symbolic form.

The `kw_args` are eventually splatted into `model_predict`.
"""
function model_basic_symbolic_output(g_spec, agent; kw_args...)
    return model_basic_symbolic_output(
        g_spec, agent.genome, agent.parameter, agent.extra; kw_args...)
end

"""
    model_symbolic_output(g_spec, agent; kw_args...)

Build a Symbolics form for the output of the final time step of
running `agent`'s `genome`
Then use the `agent`'s `parameter` vector,
and feed the symbolic output of the genome as input to
model result in `agent.extra` to make a prediction
in symbolic form.
The output tuple includes the result of various simplifications.

The `kw_args` are eventually splatted into `model_predict`.
"""
function model_symbolic_output(g_spec, agent; kw_args...)
    return model_symbolic_output(
        g_spec, agent.genome, agent.parameter, agent.extra; kw_args...)
end

"""
    model_symbolic_output(g_spec, genome, parameter, model_result; kw_args...)

Build a Symbolics form for the output of the final time step of
running `genome`
Then use the `parameter` vector,
the symbolic output of the genome,
and feed the symbolic output of the genome as input to
model result in `agent.extra` to make a prediction
in symbolic form.
The output tuple includes the result of various simplifications.

The `kw_args` are eventually splatted into `model_predict`.

Return a named tuple with lots of useful fields.

"""
function model_symbolic_output(
        g_spec::GenomeSpec,
        genome::AbstractGenome,
        parameter::AbstractArray,
        mr::AbstractModelResult;
        kw_args...)
    p, x, z = run_genome_symbolic(g_spec, genome)
    p_subs = Dict(p[j] => parameter[j] for j in eachindex(p))
    z_num = map(z) do zj
        substitute(zj, p_subs)
    end
    try
        z_sym_row_mat = reshape(z, 1, :)
        y_sym = model_predict(mr, z_sym_row_mat; kw_args...)[1]
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
        y_W = substitute(y_sym, Dict([Inf => W]))
        y_lim = Symbolics.limit(y_W.val, W.val, Inf)
        y_simp = Symbolics.simplify(y_lim)
        y_sub = substitute(y_simp, p_subs)
        y_num = Symbolics.simplify(y_sub)

        return (p = p, x = x, z = z, p_subs = p_subs, z_num = z_num,
            y_sym = y_sym,
            y_lim = y_lim, y_simp = y_simp, y_sub = y_sub, y_num = y_num)
    catch e
        @info "model_predict failed, returning short symbolic report: $(e)"
        return (p = p, x = x, z = z, p_subs = p_subs, z_num = z_num)
    end
end

"""
    model_basic_sympy_output(g_spec, agent; kw_args...)

Build a SymPy form for the output of the final time step of
running `agent`'s `genome`
Then use the `agent`'s `parameter` vector,
and feed the symbolic output of the genome as input to
model result in `agent.extra` to make a prediction
in symbolic form.

Key-word arguments:

- `assumptions = Dict(:extended_real => true)`:
Assumptions to use when creating symbols.

Other `kw_args` are eventually splatted into `model_predict`.
"""
function model_basic_sympy_output(g_spec, agent; kw_args...)
    return model_basic_sympy_output(
        g_spec, agent.genome, agent.parameter, agent.extra; kw_args...)
end

"""
    model_sympy_output(g_spec, agent; kw_args...)

Build a SymPy form for the output of the final time step of
running `agent`'s `genome`
Then use the `agent`'s `parameter` vector,
and feed the symbolic output of the genome as input to
model result in `agent.extra` to make a prediction
in symbolic form.
The output tuple includes the result of various simplifications.

Key-word arguments:

- `assumptions = Dict(:extended_real => true)`:
Assumptions to use when creating symbols.

Other `kw_args` are eventually splatted into `model_predict`.
"""
function model_sympy_output(g_spec, agent; kw_args...)
    return model_sympy_output(
        g_spec, agent.genome, agent.parameter, agent.extra; kw_args...)
end

"""
    model_sympy_output(g_spec, genome, parameter, model_result; kw_args...)

Build a SymPy form for the output of the final time step of
running `genome`
Then use the `parameter` vector,
the symbolic output of the genome,
and feed the symbolic output of the genome as input to
model result in `agent.extra` to make a prediction
in symbolic form.
The output tuple includes the result of various simplifications.

Key-word arguments:

- `assumptions = Dict(:extended_real => true)`:
Assumptions to use when creating symbols.

Other `kw_args` are eventually splatted into `model_predict`.

Return a named tuple with lots of useful fields.

"""
function model_sympy_output(
        g_spec::GenomeSpec,
        genome::AbstractGenome,
        parameter::AbstractArray,
        mr::AbstractModelResult;
        assumptions = Dict(:extended_real => true),
        kw_args...)
    p, x, z = run_genome_sympy(g_spec, genome; assumptions = assumptions)
    p_subs = Dict(p[j] => parameter[j] for j in eachindex(p))
    z_num = [zj(p_subs) for zj in z]
    try
        z_sym_row_mat = reshape(z, 1, :)
        y_sym = model_predict(mr, z_sym_row_mat; kw_args...)[1]
        used_vars = y_sym.free_variables
        # To handle rational functions that have things like 1/(x/0),
        # replace Inf with W and do a limit as W -> Inf.
        # First, grind through and make sure we have a unique symbol.
        j = 0
        W = symbols("W"; assumptions...)
        if !isnothing(used_vars)
            while true
                W = symbols("W$j"; assumptions...)
                if !(W in used_vars)
                    break
                end
                j += 1
            end
        end
        # Replace all infinities with W.  The complex infinited zoo
        # is not really correct but this is the best we can do at
        # this point.
        y_W = y_sym.subs(sympy.oo, W).subs(-sympy.oo, -W).subs(sympy.zoo, W)
        y_lim = y_W.limit(W, sympy.oo)
        y_simp = SymPy.simplify(y_lim)
        y_sub = y_simp(p_subs)
        y_num = SymPy.simplify(y_sub)

        return (p = p, x = x, z = z, p_subs = p_subs, z_num = z_num,
            y_sym = y_sym,
            y_lim = y_lim, y_simp = y_simp, y_sub = y_sub, y_num = y_num)
    catch e
        @info "model_predict failed, returning short symbolic result: $(e)"
        return (p = p, x = x, z = z, p_subs = p_subs, z_num = z_num)
    end
end
