module TestTimeSeries
using Distributions
using Random
using Test

using Jessamine

function as_rows(m::Array)
    n = size(m, 1)
    return map(1:n) do j
        @view m[j, :]
    end
end

function main()
    rng = Xoshiro(202406181224)

    println("Test run_genome_time_series")

    a = 1.4
    ma = -a
    b = 0.3

    f((x, y), (xm, ym)) = [
        1 - a * x^2 + y + xm,
        b * x + ym]

    g_spec = GenomeSpec(2, 1, 3, 4, 2)

    input_mat = Float64[
    1 0
    0 1
    1 1
    ]

    input_series = eachrow(input_mat)
    @show input_series

    expected_output_series = Vector{Float64}[]
    u = [0, 0]
    for r in input_series
        u = f(r, u)
        push!(expected_output_series, u)
    end

    @show expected_output_series

    # Map from variables to indices
    x1, y1, t1, p1, pma, pb, x0, y0, xm1, ym1 = 1:10

    instruction_blocks = [
        [Instruction(Add(), [p1, t1, y0, xm1])],
        [Instruction(Multiply(), [pb, x0]), Instruction(Add(), [ym1])],
        [Instruction(Multiply(), [pma, x0, x0])]
    ]

    parameter = [1.0, ma, b]

    genome = Genome(instruction_blocks)

    output_series = run_genome_time_series(g_spec, genome, parameter, input_series)

    @show output_series

end

end
