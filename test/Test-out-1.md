# Test output 1

## Run 1

Julia 1.11.7, main at 67a906ca75067faa9e6acbad74fb713c5f6684cb

```
     Testing Running tests...
=== TestGenome ===
Test run_genome with something basic
run_genome(g_spec, genome, Float64[], [4.0, 5.0]) = Any[[9.0, 20.0], [29.0, 20.0]]
Test run_genome with the Henon map
h(input) = [1.35, 0.15]
Test to_expr
cg.expr = quote
    #= /home/mitchenerg/git/Research/Jessamine.jl/src/GenomeCore.jl:238 =#
    function (cs::CellState{E},) where E
        #= /home/mitchenerg/git/Research/Jessamine.jl/src/GenomeCore.jl:238 =#
        #= /home/mitchenerg/git/Research/Jessamine.jl/src/GenomeCore.jl:239 =#
        new_output = E[.+(cs.parameter[1], cs.scratch[1], cs.input[2]), cs.parameter[3] .* cs.input[1]]
        #= /home/mitchenerg/git/Research/Jessamine.jl/src/GenomeCore.jl:241 =#
        new_scratch = E[.*(cs.parameter[2], cs.input[1], cs.input[1])]
        #= /home/mitchenerg/git/Research/Jessamine.jl/src/GenomeCore.jl:243 =#
        return CellState(new_output, new_scratch, cs.parameter, cs.input)
    end
end
Test vectorization
Test vectorization:
h(v_input) = [[1.35, 1.4500000000000002, 1.744], [0.15, 0.15, 0.06]]
hc(v_input) = [[1.35, 1.4500000000000002, 1.744], [0.15, 0.15, 0.06]]
Test RandomDuplicateDelete
rdd_res = [1, 2, 3, 4, 5, 6, 7, 7, 8, 9, 10, 13, 14, 15, 15, 15, 16, 17, 18, 19, 20, 21, 23, 23, 24, 25, 26, 28, 29, 29, 30, 31, 31, 32, 33, 34, 35, 36, 37, 39, 40, 41, 42, 42, 43, 44, 45, 46, 47, 47, 48, 48, 48, 50, 51, 52, 54, 55, 56, 58, 59, 60, 62, 63, 64, 66, 67, 68, 69, 70, 71, 72, 73, 74, 74, 76, 77, 79, 81, 82, 83, 83, 84, 84, 84, 85, 86, 87, 88, 89, 91, 92, 93, 94, 94, 98, 99, 100]
collect(rdd) = [1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 10, 11, 12, 13, 14, 17, 18, 19, 20, 21, 24, 25, 25, 26, 27, 28, 29, 31, 32, 33, 33, 34, 35, 36, 37, 37, 37, 38, 40, 41, 41, 42, 44, 45, 46, 47, 48, 49, 50, 52, 52, 53, 54, 55, 56, 57, 58, 59, 60, 62, 63, 63, 65, 65, 66, 66, 67, 68, 69, 70, 70, 72, 72, 73, 74, 75, 76, 77, 77, 77, 78, 79, 82, 82, 82, 83, 84, 85, 85, 85, 86, 86, 86, 89, 89, 91, 92, 93, 94, 95, 95, 96, 97, 97, 98, 99, 100]
----------
Test mutation
inst1 = Jessamine.Instruction(Jessamine.Add(), [1, 3, 5])
inst2 = Jessamine.Instruction(Jessamine.Multiply(), [1, 3])
inst3 = Jessamine.Instruction(Jessamine.Add(), [3])
inst4 = Jessamine.Instruction(Jessamine.Add(), [3, 3])
inst5 = Jessamine.Instruction(Jessamine.Add(), [3, 8])
inst6 = Jessamine.Instruction(Jessamine.Add(), [3, 3, 8])
inst7 = Jessamine.Instruction(Jessamine.Add(), [9, 3, 1, 8])
inst8 = Jessamine.Instruction(Jessamine.Add(), [2, 3, 5, 8])
inst9 = Jessamine.Instruction(Jessamine.Add(), [3, 7, 6, 8, 5, 8])
inst10 = Jessamine.Instruction(Jessamine.Add(), [4, 1, 6, 3, 2, 8, 4, 6, 8])
genome1 = 
1 = sum [ add 1 3 5, mul 1 3, ]
2 = sum [ add 3, add 3 3, add 3 8, ]
3 = sum [ add 3 3 8, ]
4 = sum [ ]
5 = sum [ add 9 3 1 8, add 2 3 5 8, add 3 7 6 8 5 8, add 4 1 6 3 2 8 4 6 8, ]
genome2 = 
1 = sum [ add 7 3 3 4, add 9 5, add 5 3 3 8 2, ]
2 = sum [ mul 3, ]
3 = sum [ ]
4 = sum [ add 4 1 6 5 7 2 8 8 3 8 9 7, ]
5 = sum [ mul 2 8 1 2, add 2 6 3 5 7, add 8 8 5, add 7 6 3 5 5 1, mul 7, ]
----------
Test recombination
genome3 = 
1 = sum [ add 3 7 6 8 5 8, add 4 1 6 3 2 8 4 6 8, add 1 3 5, ]
2 = sum [ add 3 3 8, add 9 3 1 8, add 2 3 5 8, ]
3 = sum [ add 3 3 8, ]
4 = sum [ add 1 3 5, mul 1 3, ]
5 = sum [ mul 1 3, add 3, add 3 3, ]
genome4 = 
1 = sum [ add 3 7 6 8 5 8, add 4 1 6 3 2 8 4 6 8, add 1 3 5, ]
2 = sum [ add 3 3 8, add 9 3 1 8, add 2 3 5 8, ]
3 = sum [ add 3 3 8, ]
4 = sum [ add 1 3 5, ]
5 = sum [ mul 1 3, add 2 3 5 8, add 3 7 6 8 5 8, add 4 1 6 3 2 8 4 6 8, ]
genome5 = 
1 = sum [ add 3 7 6 8 5 8, mul 1 3, ]
2 = sum [ add 3, add 3 3, add 2 3 5 8, ]
3 = sum [ add 3 3 8, ]
4 = sum [ add 1 3 5, mul 1 3, ]
5 = sum [ mul 1 3, add 2 3 5 8, add 3 7 6 8 5 8, add 4 1 6 3 2 8 4 6 8, ]
1 = sum [ fzAnd 1 2, fzAnd 5, fzAnd, ]
2 = sum [ fzOr 1 2, fzOr 5, fzOr, ]
3 = sum [ fzNand 1 2, fzNand 5, fzNand, ]
4 = sum [ fzNor 1 2, fzNor 5, fzNor, ]
5 = sum [ ]
(compile(g_spec, genome_fzlogic)).expr = quote
    #= /home/mitchenerg/git/Research/Jessamine.jl/src/GenomeCore.jl:238 =#
    function (cs::CellState{E},) where E
        #= /home/mitchenerg/git/Research/Jessamine.jl/src/GenomeCore.jl:238 =#
        #= /home/mitchenerg/git/Research/Jessamine.jl/src/GenomeCore.jl:239 =#
        new_output = E[.+(cs.output[1] .* cs.output[2], cs.parameter[2], [1.0]), .+(1.0 .- (1.0 .- cs.output[1]) .* (1.0 .- cs.output[2]), 1.0 .- (1.0 .- cs.parameter[2]), 1.0 .- [1.0])]
        #= /home/mitchenerg/git/Research/Jessamine.jl/src/GenomeCore.jl:241 =#
        new_scratch = E[.+(1.0 .- cs.output[1] .* cs.output[2], 1.0 .- cs.parameter[2], 1.0 .- [1.0]), .+((1.0 .- cs.output[1]) .* (1.0 .- cs.output[2]), 1.0 .- cs.parameter[2], [1.0]), [0.0]]
        #= /home/mitchenerg/git/Research/Jessamine.jl/src/GenomeCore.jl:243 =#
        return CellState(new_output, new_scratch, cs.parameter, cs.input)
    end
end
1 = sum [ min 1 2, min 5, min, ]
2 = sum [ max 1 2, max 5, max, ]
3 = sum [ sign@add 1 2, sign@add 5, sign@add, ]
4 = sum [ ]
5 = sum [ ]
(compile(g_spec, genome_min_max)).expr = quote
    #= /home/mitchenerg/git/Research/Jessamine.jl/src/GenomeCore.jl:238 =#
    function (cs::CellState{E},) where E
        #= /home/mitchenerg/git/Research/Jessamine.jl/src/GenomeCore.jl:238 =#
        #= /home/mitchenerg/git/Research/Jessamine.jl/src/GenomeCore.jl:239 =#
        new_output = E[.+((#= /home/mitchenerg/git/Research/Jessamine.jl/src/Operations.jl:500 =#
                        min.(cs.output[1], cs.output[2])), cs.parameter[2], [Inf]), .+((#= /home/mitchenerg/git/Research/Jessamine.jl/src/Operations.jl:469 =#
                        max.(cs.output[1], cs.output[2])), cs.parameter[2], [-Inf])]
        #= /home/mitchenerg/git/Research/Jessamine.jl/src/GenomeCore.jl:241 =#
        new_scratch = E[.+(sign.(cs.output[1] .+ cs.output[2]), sign.(cs.parameter[2]), sign.([0.0])), [0.0], [0.0]]
        #= /home/mitchenerg/git/Research/Jessamine.jl/src/GenomeCore.jl:243 =#
        return CellState(new_output, new_scratch, cs.parameter, cs.input)
    end
end
----------
=== TestTimeSeries ===
Test run_genome_time_series
input_series = SubArray{Float64, 1, Matrix{Float64}, Tuple{Int64, Base.Slice{Base.OneTo{Int64}}}, true}[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
expected_output_series = [[-0.3999999999999999, 0.3], [1.6, 0.3], [2.2, 0.6]]
output_series = [[-0.3999999999999999, 0.3], [1.6, 0.3], [2.2, 0.6]]
=== TestRidge ===
least_squares_ridge(RD.xs, RD.y, 0.0, g_spec, genome, parameter) = (55.64844247468291, [-2.2441686048441367, 3.9192866009158536])
least_squares_ridge(RD.xs, RD.y, 1.0e-6, g_spec, genome, parameter) = (55.648442474682916, [-2.244168452607728, 3.9192864149077176])
least_squares_ridge(RD.xs, RD.y, 0.1, g_spec, genome, parameter) = (55.64853764369994, [-2.2290349823203, 3.9007808527938344])
least_squares_ridge(RD.xs, RD.y, 0.5, g_spec, genome, parameter) = (55.65072275160966, [-2.1702511158826656, 3.828609737038582])
least_squares_ridge(RD.xs, RD.y, 1.0, g_spec, genome, parameter) = (55.65710271642703, [-2.1004981011238884, 3.742351508905067])
agent.parameter = [0.9085937500000003, 1.4039062500000008]
agent.extra = Jessamine.BasicLinearModelResult([1.5985174052727, 0.310074366159137], 0.0)
y_rgr_pred = [4.960735314646249, -5.990129542138466, -1.5494479513747637, 5.216913306433437, 0.14421414297327664, -1.190949656537722, -3.1663619004852253, 2.2334473432255066, -8.653647967250876, -0.16048336639301233, 0.0960082955181931, 6.390842059882193, -0.2505643123920805, 0.2907774149399621, -1.029981284970866, 4.645905756530943, 3.066246124863775, 5.0490823800856495, 5.830163925803522, 2.0286153753030285, 5.353648241476649, -3.4737255723551654, -2.4004244728392483, 0.11444390387638187, -3.785128187662405, 3.7307556520389675, -3.5491312017289522, 1.6783439846921335, 1.3161847550569254, 1.433511346408858]
=== TestSymbolics ===
y_pred_sym = 1.5985174052727(1.4039062500000008(1.0 + 0.9085937500000003x₁) + x₁) + 0.2817316311274035x₁
y_pred_simp = 1.5985174052727(1.4039062500000008(1.0 + 0.9085937500000003x₁) + x₁) + 0.2817316311274035x₁
=== TestSymPy ===
Part 1
y_pred_sym = 3.91928657849659*x1 + 2.24416857599613
y_pred_simp = 3.91928657849659*x1 + 2.24416857599613
Part 2
show_sympy_res = SymPyCore.Sym{PyCall.PyObject}[1 z1 t2 + x1; 2 z2 p1*x1; 3 t1 t3 + z2; 4 t2 p2*t1; 5 t3 1.00000000000000; 6 p1 p1; 7 p2 p2; 8 x1 x1; 9 x2 x2]
sym_res = (p = SymPyCore.Sym{PyCall.PyObject}[p1, p2], x = SymPyCore.Sym{PyCall.PyObject}[x1, x2], z = SymPyCore.Sym{PyCall.PyObject}[p2*(p1*x1 + 1.0) + x1, p1*x1])
=== TestEvolution ===
rng = Xoshiro(0xa09fa28910f144d9, 0x580ce71f0e077561, 0xac4a61c5799670e6, 0xd38da02b37d8aadb, 0xcf19a1f021e48577)
This should result in 2 + 3 x1 + 3 x2 - 3 x1 * x2
a_check = Jessamine.Agent{Float64, Jessamine.Genome, Vector{Float64}, Jessamine.BasicLinearModelResult}(3.49999991995944e-5, Jessamine.Genome(Vector{Jessamine.Instruction}[[Jessamine.Instruction(Jessamine.Multiply(), Int64[])], [Jessamine.Instruction(Jessamine.Multiply(), [6])], [Jessamine.Instruction(Jessamine.Multiply(), [7])], [Jessamine.Instruction(Jessamine.Multiply(), [6, 7])]]), [0.0], Jessamine.BasicLinearModelResult([1.9999998660311984, 2.9999998474424903, 2.999999972097383, -3.0000000029707916], 0.0))
rating = 3.49999991995944e-5
1 = sum [ mul, ]
2 = sum [ mul 6, ]
3 = sum [ mul 7, ]
4 = sum [ mul 6 7, ]
parameter = [0.0]
extra = Jessamine.BasicLinearModelResult([1.9999998660311984, 2.9999998474424903, 2.999999972097383, -3.0000000029707916], 0.0)
round.(coefficients(a_check.extra)) = [2.0, 3.0, 3.0, -3.0]
After a_check
rng = Xoshiro(0xa09fa28910f144d9, 0x580ce71f0e077561, 0xac4a61c5799670e6, 0xd38da02b37d8aadb, 0xcf19a1f021e48577)
Before setting pop_init
pop_init = 
slot 1:
rating = 196.22805255107315
1 = sum [ mul 7, ]
2 = sum [ mul 6 7, ]
3 = sum [ mul 7 5 2, ]
4 = sum [ mul 6 1, ]
parameter = [0.5500000000000002]
extra = Jessamine.BasicLinearModelResult([2.8746832188160516, -1.663109102776688, 0.3263523843367109, -1.6631091028123328], 0.0)

slot 2:
rating = 230.4639338955937
1 = sum [ mul 3 6, ]
2 = sum [ mul 4 6, ]
3 = sum [ mul 5 1 4, ]
4 = sum [ mul 5 7, ]
parameter = [2.1390625000000005]
extra = Jessamine.BasicLinearModelResult([0.0, -1.618249605249805, 0.0, 1.2834795442892661], 0.0)

slot 3:
rating = 695.4972302835774
1 = sum [ mul 4 6, ]
2 = sum [ mul 7 6, ]
3 = sum [ mul 5 7 1, ]
4 = sum [ mul 2 6, ]
parameter = [0.0]
extra = Jessamine.BasicLinearModelResult([0.997488093602185, -3.779363461675345, 0.0, 3.1967218822021395], 0.0)

slot 4:
rating = 961.9553419052736
1 = sum [ mul 6 2, ]
2 = sum [ mul 3 5, ]
3 = sum [ mul 7 6, ]
4 = sum [ mul 1, ]
parameter = [1.6328125000000007]
extra = Jessamine.BasicLinearModelResult([0.6881500405970651, -1.5059405676625244, -0.9222983915036971, 0.0], 0.0)

slot 5:
rating = 1212.345425749155
1 = sum [ mul 7 2, ]
2 = sum [ mul 7 5, ]
3 = sum [ mul 2 2, ]
4 = sum [ mul 1 1, ]
parameter = [2.054687500000001]
extra = Jessamine.BasicLinearModelResult([0.07799270806440764, 2.031468475063376, 0.16024859838356856, -0.008911622786290381], 0.0)

slot 6:
rating = 1212.3454354928738
1 = sum [ mul 2 2, ]
2 = sum [ mul 7 3, ]
3 = sum [ mul 7, ]
4 = sum [ mul 3 3 5, ]
parameter = [0.0]
extra = Jessamine.BasicLinearModelResult([-0.03762256001310397, 0.8367786614346232, 4.174032853211568, 0.0], 0.0)

slot 7:
rating = 1282.0051934112348
1 = sum [ mul 5 4 4, ]
2 = sum [ mul 5, ]
3 = sum [ mul 7, ]
4 = sum [ mul 2 3, ]
parameter = [1.7171875000000008]
extra = Jessamine.BasicLinearModelResult([0.026569093184420783, 0.2668110633860057, 0.9784112436134763, 1.680115660271796], 0.0)

slot 8:
rating = 1286.0128682025008
1 = sum [ mul 2, ]
2 = sum [ mul 7, ]
3 = sum [ mul 3 5, ]
4 = sum [ mul 7 7, ]
parameter = [0.0]
extra = Jessamine.BasicLinearModelResult([1.9224291590773717, 1.9224292079626746, 0.0, 0.16869559738674447], 0.0)

slot 9:
rating = 1286.0128765939694
1 = sum [ mul 7, ]
2 = sum [ mul 3 6, ]
3 = sum [ mul 3 5, ]
4 = sum [ mul 7 1, ]
parameter = [0.0]
extra = Jessamine.BasicLinearModelResult([3.844858353673578, 0.0, 0.0, 0.16869559757445562], 0.0)

slot 10:
rating = 1295.954096346741
1 = sum [ mul 3 3 3, ]
2 = sum [ mul 6 2, ]
3 = sum [ mul 7, ]
4 = sum [ mul 1 7, ]
parameter = [0.0]
extra = Jessamine.BasicLinearModelResult([0.05630903777593947, 0.0, 3.0753852507487505, 0.003339974736121222], 0.0)

...
end pop_init
rng = Xoshiro(0x8d5c24a5edda71c1, 0x21a21c721ec865fc, 0x88c40b7833e28795, 0xbfc4364feed385ce, 0xcf19a1f021e48577)

Before pop_next
rng = Xoshiro(0x8d5c24a5edda71c1, 0x21a21c721ec865fc, 0x88c40b7833e28795, 0xbfc4364feed385ce, 0xcf19a1f021e48577)
Generation 10, best = 3.121177170831467e-5
Generation 20, best = 2.4381787216386437e-5
Generation 30, best = 1.5603429459816484e-5
Generation 40, best = 1.314062877328546e-5
Generation 50, best = 1.2820248716083718e-5
Generation 60, best = 1.2820248716083718e-5
Generation 70, best = 1.2820248716083718e-5
Generation 80, best = 1.2820248716083718e-5
Generation 90, best = 1.2820248716083718e-5
Generation 100, best = 1.2820248716083718e-5
pop_next = 
slot 1:
rating = 1.2820248716083718e-5
1 = sum [ mul 4 6, ]
2 = sum [ mul 1 3, ]
3 = sum [ mul 4 7, ]
4 = sum [ mul 5 5 5, ]
parameter = [1.6750000000000007]
extra = Jessamine.BasicLinearModelResult([0.6383763943967041, -0.13584147441211855, 0.6383763956525583, 0.42558426262141863], 0.0)

slot 2:
rating = 1.2820248716083718e-5
1 = sum [ mul 4 6, ]
2 = sum [ mul 1 3, ]
3 = sum [ mul 4 7, ]
4 = sum [ mul 5 5 5, ]
parameter = [1.6750000000000007]
extra = Jessamine.BasicLinearModelResult([0.6383763943967041, -0.13584147441211855, 0.6383763956525583, 0.42558426262141863], 0.0)

slot 3:
rating = 1.2820248716083718e-5
1 = sum [ mul 4 6, ]
2 = sum [ mul 1 3, ]
3 = sum [ mul 4 7, ]
4 = sum [ mul 5 5 5, ]
parameter = [1.6750000000000007]
extra = Jessamine.BasicLinearModelResult([0.6383763943967041, -0.13584147441211855, 0.6383763956525583, 0.42558426262141863], 0.0)

slot 4:
rating = 1.2820248716083718e-5
1 = sum [ mul 4 6, ]
2 = sum [ mul 1 3, ]
3 = sum [ mul 4 7, ]
4 = sum [ mul 5 5 5, ]
parameter = [1.6750000000000007]
extra = Jessamine.BasicLinearModelResult([0.6383763943967041, -0.13584147441211855, 0.6383763956525583, 0.42558426262141863], 0.0)

slot 5:
rating = 1.2820248716083718e-5
1 = sum [ mul 4 6, ]
2 = sum [ mul 1 3, ]
3 = sum [ mul 4 7, ]
4 = sum [ mul 5 5 5, ]
parameter = [1.6750000000000007]
extra = Jessamine.BasicLinearModelResult([0.6383763943967041, -0.13584147441211855, 0.6383763956525583, 0.42558426262141863], 0.0)

slot 6:
rating = 1.2820248716083718e-5
1 = sum [ mul 4 6, ]
2 = sum [ mul 1 3, ]
3 = sum [ mul 4 7, ]
4 = sum [ mul 5 5 5, ]
parameter = [1.6750000000000007]
extra = Jessamine.BasicLinearModelResult([0.6383763943967041, -0.13584147441211855, 0.6383763956525583, 0.42558426262141863], 0.0)

slot 7:
rating = 1.2820248716083718e-5
1 = sum [ mul 4 6, ]
2 = sum [ mul 1 3, ]
3 = sum [ mul 4 7, ]
4 = sum [ mul 5 5 5, ]
parameter = [1.6750000000000007]
extra = Jessamine.BasicLinearModelResult([0.6383763943967041, -0.13584147441211855, 0.6383763956525583, 0.42558426262141863], 0.0)

slot 8:
rating = 1.2820248716083718e-5
1 = sum [ mul 4 6, ]
2 = sum [ mul 1 3, ]
3 = sum [ mul 4 7, ]
4 = sum [ mul 5 5 5, ]
parameter = [1.6750000000000007]
extra = Jessamine.BasicLinearModelResult([0.6383763943967041, -0.13584147441211855, 0.6383763956525583, 0.42558426262141863], 0.0)

slot 9:
rating = 1.2820248716083718e-5
1 = sum [ mul 4 6, ]
2 = sum [ mul 1 3, ]
3 = sum [ mul 4 7, ]
4 = sum [ mul 5 5 5, ]
parameter = [1.6750000000000007]
extra = Jessamine.BasicLinearModelResult([0.6383763943967041, -0.13584147441211855, 0.6383763956525583, 0.42558426262141863], 0.0)

slot 10:
rating = 1.2820248716083718e-5
1 = sum [ mul 4 6, ]
2 = sum [ mul 1 3, ]
3 = sum [ mul 4 7, ]
4 = sum [ mul 5 5 5, ]
parameter = [1.6750000000000007]
extra = Jessamine.BasicLinearModelResult([0.6383763943967041, -0.13584147441211855, 0.6383763956525583, 0.42558426262141863], 0.0)

...
end pop_next
rng = Xoshiro(0x073a8dca12e6d51e, 0x91bc0b5cd7581891, 0xeb85fd1b34e477ff, 0x3dfffe098e37896d, 0xcf19a1f021e48577)

Symbolic form of best agent
rating = 1.2820248716083718e-5
1 = sum [ mul 4 6, ]
2 = sum [ mul 1 3, ]
3 = sum [ mul 4 7, ]
4 = sum [ mul 5 5 5, ]
parameter = [1.6750000000000007]
extra = Jessamine.BasicLinearModelResult([0.6383763943967041, -0.13584147441211855, 0.6383763956525583, 0.42558426262141863], 0.0)
basic_sym_res.y_sym = b₀ + b₄*(p₁^3) + b₁*(p₁^3)*x₁ + b₃*(p₁^3)*x₂ + b₂*(p₁^6)*x₁*x₂
basic_sym_res.y_num = 1.999999993418842 + 2.9999999923115026x₁ + 2.9999999982132914x₂ - 3.0000000016532926x₁*x₂
sym_res.y_sym = b₀ + b₄*(p₁^3) + b₁*(p₁^3)*x₁ + b₃*(p₁^3)*x₂ + b₂*(p₁^6)*x₁*x₂
a_best.parameter = [1.6750000000000007]
a_best.extra = Jessamine.BasicLinearModelResult([0.6383763943967041, -0.13584147441211855, 0.6383763956525583, 0.42558426262141863], 0.0)
sym_res.y_num = 1.999999993418842 + 2.9999999923115026x₁ + 2.9999999982132914x₂ - 3.0000000016532926x₁*x₂
y_num = 1.999999993418842 + 2.9999999923115026x₁ + 2.9999999982132914x₂ - 3.0000000016532926x₁*x₂
cvec = [1.999999993418842, 2.9999999923115026, 2.9999999982132914, -3.0000000016532926]
round.(cvec) = [2.0, 3.0, 3.0, -3.0]
rating = 1.2820248716083718e-5
1 = sum [ mul 4 6, ]
2 = sum [ mul 1 3, ]
3 = sum [ mul 4 7, ]
4 = sum [ mul 5 5 5, ]
parameter = [1.6750000000000007]
extra = Jessamine.BasicLinearModelResult([0.6383763943967041, -0.13584147441211855, 0.6383763956525583, 0.42558426262141863], 0.0)
basic_sympy_res.y_sym = b0 + b1*p1^3*x1 + b2*p1^6*x1*x2 + b3*p1^3*x2 + b4*p1^3
basic_sympy_res.y_num = -3.00000000165329*x1*x2 + 2.9999999923115*x1 + 2.99999999821329*x2 + 1.99999999341884
sympy_res.y_sym = b0 + b1*p1^3*x1 + b2*p1^6*x1*x2 + b3*p1^3*x2 + b4*p1^3
sympy_res.y_num = -3.00000000165329*x1*x2 + 2.9999999923115*x1 + 2.99999999821329*x2 + 1.99999999341884
rng = Xoshiro(0x073a8dca12e6d51e, 0x91bc0b5cd7581891, 0xeb85fd1b34e477ff, 0x3dfffe098e37896d, 0xcf19a1f021e48577)
     Testing Jessamine tests passed
```
