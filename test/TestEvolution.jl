module TestEvolution
using Distributions
using LinearAlgebra
using Optimization
using Random
using Symbolics
using Test
using Jessamine
using ..RD

function main()
    # Set the global RNG seed in case something outside my control uses it
    Random.seed!(19101510324)

    rng = Xoshiro(3208201)

    @show rng
    global g_spec = GenomeSpec(4, 0, 1, 2, 3)
    global index_max = workspace_size(g_spec)
    global m_spec = MutationSpec(
        p_mutate_op = 0.0,
        p_mutate_index = 0.1,
        p_duplicate_index = 0.01,
        p_delete_index = 0.01,
        p_duplicate_instruction = 0.001,
        p_delete_instruction = 0.001,
        op_inventory = [Multiply()])
    global m_dist = MutationDist(m_spec, index_max)
    global arity_dist = DiscreteNonParametric([1, 2, 3], [0.25, 0.5, 0.25])
    global s_spec = SelectionSpec(20, 30, 0.6, 0.1)
    global gen_size = generation_size(s_spec)
    global s_dist = SelectionDist(s_spec)
    global lambda_b = 1e-6
    global lambda_p = 1e-6
    global lambda_o = 1e-6

    # Adapt least_squares_ridge_grow_and_rate so that it can be used by next_generation
    function grow_and_rate(rng, g_spec, genome)
        return least_squares_ridge_grow_and_rate(
            RD.xs, RD.y, lambda_b, lambda_p, lambda_o,
            g_spec, genome)
    end

    # Indices into the state vector
    global z1, z2, z3, z4, p1, x1, x2 = 1:index_max

    # This should output [1, x1, x2, x1 * x2]
    global g_check = Genome(
        [[Instruction(Multiply(), Int[])], # creates a 1
        [Instruction(Multiply(), [x1])],
        [Instruction(Multiply(), [x2])],
        [Instruction(Multiply(), [x1, x2])]])

    println("This should result in 2 + 3 x1 + 3 x2 - 3 x1 * x2")
    global a_check = grow_and_rate(rng, g_spec, g_check)
    @show a_check
    short_show(a_check)
    @show round.(coefficients(a_check.extra))
    @test round.(coefficients(a_check.extra)) == [2, 3, 3, -3]
    @test intercept(a_check.extra) == 0

    println("After a_check")
    @show rng

    println("Before setting pop_init")
    global pop_init = random_initial_population(
        rng, g_spec, m_dist, arity_dist, s_spec, grow_and_rate)
    println("pop_init = ")
    short_show(pop_init)
    println("end pop_init")
    @show rng
    println("")

    function ev_loop(pop_init::Population, num_generations::Int)
        pop_next = pop_init
        for t in 1:num_generations
            pop_next = next_generation(
                rng, g_spec, s_dist, m_dist, pop_next, grow_and_rate; sense = MinSense)
            if mod(t, 10) == 0
                println("Generation $t, best = $(pop_next.agents[1].rating)")
            end
        end
        return pop_next
    end

    println("Before pop_next")
    @show rng
    global pop_next = ev_loop(pop_init, 100)
    println("pop_next = ")
    short_show(pop_next)
    println("end pop_next")
    @show rng
    println("")

    println("Symbolic form of best agent")

    global a_best = pop_next.agents[1]
    global basic_sym_res = model_basic_symbolic_output(g_spec, a_best)
    global sym_res = model_symbolic_output(g_spec, a_best)
    global x = sym_res.x

    short_show(a_best)
    @show basic_sym_res.y_sym
    @show basic_sym_res.y_num
    @show sym_res.y_sym
    @show a_best.parameter
    @show a_best.extra
    @show sym_res.y_num

    global y_num = simplify(sym_res.y_num; expand = true)
    @show y_num

    global c1 = Symbolics.coeff(y_num)
    global cx1 = Symbolics.coeff(Symbolics.coeff(y_num, x[1]))
    global cx2 = Symbolics.coeff(Symbolics.coeff(y_num, x[2]))
    global cx1x2 = Symbolics.coeff(Symbolics.coeff(y_num, x[2]), x[1])
    global cvec = [c1, cx1, cx2, cx1x2]
    @show cvec
    @show round.(cvec)
    @test round.(cvec) == [2, 3, 3, -3]

    global basic_sympy_res = model_basic_sympy_output(g_spec, a_best)
    global sympy_res = model_sympy_output(g_spec, a_best)

    short_show(a_best)
    @show basic_sympy_res.y_sym
    @show basic_sympy_res.y_num
    @show sympy_res.y_sym
    @show sympy_res.y_num
    @show rng
end

end
