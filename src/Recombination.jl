export recombine

"""
    recombine(rng, g_left, g_right)

Given two genomes, line up the instruction blocks.  For each
pair, choose a prefix of the block from the left, drop its length
from the block from the right, and concatenate the resulting
arrays to form a new block.

The resulting recombined blocks are used to build a new `Genome`.
No mutations are applied.
"""

function recombine(rng::AbstractRNG, g_left::Genome, g_right::Genome)
    @assert length(g_left.instruction_blocks) == length(g_right.instruction_blocks)
    mixed = similar(g_left.instruction_blocks)
    for j in eachindex(g_left.instruction_blocks)
        if rand(rng, Bernoulli(0.5))
            head_block = g_left.instruction_blocks[j]
            tail_block = g_right.instruction_blocks[j]
        else
            tail_block = g_left.instruction_blocks[j]
            head_block = g_right.instruction_blocks[j]
        end
        l_m = length(head_block)
        m = rand(rng, DiscreteUniform(0, l_m))
        h_frag = head_block[1:m]
        t_frag = tail_block[(1 + m):end]
        new_block = vcat(h_frag, t_frag)
        mixed[j] = new_block
    end
    return Genome(mixed)
end

function recombine(rng::AbstractRNG, g_left::CompiledGenome, g_right::CompiledGenome)
    return recombine(rng, g_left.genome, g_right.genome)
end

# function recombine_never_empty(rng::AbstractRNG, g_left::Genome, g_right::Genome)
#     @assert length(g_left.instruction_blocks) == length(g_right.instruction_blocks)
#     mixed = similar(g_left.instruction_blocks)
#     for j in eachindex(g_left.instruction_blocks)
#         if rand(rng, Bernoulli(0.5))
#             head_block = g_left.instruction_blocks[j]
#             tail_block = g_right.instruction_blocks[j]
#         else
#             tail_block = g_left.instruction_blocks[j]
#             head_block = g_right.instruction_blocks[j]
#         end
#         if isempty(head_block)
#             new_block = tail_block
#         elseif isempty(tail_block)
#             new_block = head_block
#         else
#             l_m = length(head_block)
#             m = rand(rng, DiscreteUniform(0, l_m))
#             h_frag = head_block[1:m]
#             t_frag = tail_block[1+m:end]
#             new_block = vcat(h_frag, t_frag)
#             @assert !isempty(new_block) "$(length(head_block)) $(length(tail_block)) $m $(length(h_frag)) $(length(t_frag))"
#         end
#         mixed[j] = new_block
#     end
#     return Genome(mixed)
# end
