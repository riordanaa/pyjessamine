module RD
using Distributions
using Random
rng = Xoshiro(161004)

num_points = 30
x1_dist = Normal(0.0, 1.0)
x1 = rand(rng, x1_dist, num_points)
x2_dist = Normal(0.0, 2.0)
x2 = rand(rng, x2_dist, num_points)
y = @. 2 + 3 * (x2 + x1 * (1 - x2))
# which is 2 + 3 x2 + 3 x1 ( 1 - x2)
#          2 + 3 x2 + 3 x1 - 3 x1 x2

xs = [x1, x2]
end # module RD

module TestRidge
using LinearAlgebra
using Test
using Jessamine
using ..RD

z1, z2, z3, t1, t2, p1, pm1, x1, x2 = 1:9

genome = Genome(
    [[Instruction(Add(), [x2])],
     [Instruction(Multiply(), [x1, t2])],
     [Instruction(Multiply(), [])],
     [Instruction(Multiply(), [x2, pm1])],
     [Instruction(Add(), [p1, t1])],
]
)

parameter = [1.0, -1.0]

g_spec = GenomeSpec(3, 2, 2, 2, 4)

g_res = run_genome_to_last(g_spec, genome, parameter, RD.xs)

function main()
    @show least_squares_ridge(RD.xs, RD.y, 0.0, g_spec, genome, parameter)
    @show least_squares_ridge(RD.xs, RD.y, 1e-6, g_spec, genome, parameter)
    @show least_squares_ridge(RD.xs, RD.y, 0.1, g_spec, genome, parameter)
    @show least_squares_ridge(RD.xs, RD.y, 0.5, g_spec, genome, parameter)
    @show least_squares_ridge(RD.xs, RD.y, 1.0, g_spec, genome, parameter)

    global lambda_b = 1e-6
    global lambda_p = lambda_b
    global lambda_o = lambda_b
    global agent = least_squares_ridge_grow_and_rate(
        RD.xs, RD.y, lambda_b, lambda_p, lambda_o, g_spec, genome)
    global y_rgr_pred = model_predict(g_spec, agent, RD.xs)
    global residuals = RD.y - y_rgr_pred

    @show agent.parameter
    @show agent.extra
    @show y_rgr_pred
end

end # module TestRidge

module TestSymbolics

using LinearAlgebra
using Symbolics
using Jessamine
using ..RD
using ..TestRidge

function main()
    # Things work best if we use an array of Symbolic objects
    # instead of a Symbolic array object.
    x = Symbolics.variables(:x, 1:2)

    global z_sym = run_genome_to_last(
        TestRidge.g_spec, TestRidge.genome, Num.(TestRidge.agent.parameter), x)
    global b = coefficients(TestRidge.agent.extra)
    global y_pred_sym = dot(z_sym, b)
    @show y_pred_sym
    global y_pred_simp = Symbolics.expand(y_pred_sym)
    @show y_pred_simp
end

end # module TestSymbolics

module TestSymPy
using LinearAlgebra
using SymPy
using Jessamine
using ..RD
using ..TestRidge

function main()
    println("Part 1")
    global x = [symbols("x$j", extended_real = true)
                for j in 1:(TestRidge.g_spec.input_size)]

    global z_sym = run_genome_to_last(
        TestRidge.g_spec, TestRidge.genome, map(Sym, TestRidge.agent.parameter), x)
    global b = coefficients(TestRidge.agent.extra)
    global y_pred_sym = dot(z_sym, b)
    @show y_pred_sym
    global y_pred_simp = sympy.expand(y_pred_sym)
    @show y_pred_simp

    println("Part 2")

    global show_sympy_res = show_sympy(TestRidge.g_spec, TestRidge.genome)
    @show show_sympy_res

    sym_res = run_genome_sympy(TestRidge.g_spec, TestRidge.genome)
    @show sym_res
end
end # module TestSymPy
