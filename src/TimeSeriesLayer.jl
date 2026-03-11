if false
    include("GenomeCore.jl")
end

export run_genome_time_series

"""
TODO add documentation
"""
function run_genome_time_series(
    g_spec::GenomeSpec,
    genome::AbstractGenome,
    parameter::AbstractArray,
    input_series::AbstractArray;
    memory_steps = 1
    )
    @assert memory_steps > 0

    # Initialize memory vector with zeros
    point_type = eltype(input_series)
    point_eltype = eltype(point_type)
    memory_v = zeros(point_eltype, g_spec.output_size * memory_steps)
    outputs_series = Vector{Vector{point_eltype}}(undef, length(input_series))
    for t in 1:length(input_series)
        point = input_series[t]
        # Pull one point from time series and append the memory vector
        input_v = vcat(point, memory_v)
        # Feed [point memory_v] as input to the genome
        last_output = run_genome_to_last(g_spec, genome, parameter, input_v)
        outputs_series[t] = last_output
        # Shift the memory vector by one notch
        memory_v = vcat(last_output, memory_v[(g_spec.output_size + 1):end])
    end
    return outputs_series
end
