abstract MRFSampler

"""
The MRFGibbsSampler type houses the parameters of the Gibbs sampling algorithm.  The parameters are defined below:

burn_in:  The first burn_in samples will be discarded.  They will not be returned.
The thinning parameter does not affect the burn in period.
This is used to ensure that the Gibbs sampler converges to the target stationary distribution before actual samples are drawn.

thinning: For every thinning + 1 number of samples drawn, only the last is kept.
Thinning is used to reduce autocorrelation between samples.
Thinning is not used during the burn in period.
e.g. If thinning is 1, samples will be drawn in groups of two and only the second sample will be in the output.

evidence: the assignment that all samples must be consistent with (ie, Assignment(:A=>1) means all samples must have :A=1).
Use to sample conditional distributions.

initial_sample:  The inital assignment to variables to use.  If null, the initial sample is chosen at random
"""
type MRFGibbsSampler <: MRFSampler

    evidence::Assignment
    burn_in::Int
    thinning::Int
    initial_sample::Nullable{Assignment}

    function MRFGibbsSampler(evidence::Assignment=Assignment();
                             burn_in::Int=100,
                             thinning::Int=0,
                             initial_sample::Nullable{Assignment}=Nullable{Assignment}()
                             )
        new(evidence, burn_in, thinning, initial_sample)
    end
end
    
"""
Implements Gibbs sampling for MRFs.
This Gibbs sample only supports discrete MRFs, and samples are
drawn following a Categorical Distribution with probabilities
equal to the normalized potentials

Sampling requires an MRFGibbsSampler object which contains the parameters
"""
function Base.rand(mrf::MRF, sampler::MRFGibbsSampler, nsamples::Integer)

    return gibbs_sample(mrf, nsamples, sampler.burn_in, thinning=sampler.thinning,
                        evidence=sampler.evidence, initial_sample=sampler.initial_sample)
end


function gibbs_sample(mrf::MRF, nsamples::Integer, burn_in::Integer;
                      thinning::Integer=0,
                      evidence::Assignment=Assignment(),
                      initial_sample::Nullable{Assignment}=Nullable{Assignment}()
                      )
    # Check parameters for correctness
    nsamples > 0 || throw(ArgumentError("nsamples parameter less than 1"))
    burn_in >= 0 || throw(ArgumentError("Negative burn_in parameter"))
    if ~ isnull(initial_sample)
        init_sample = get(initial_sample)
        for vertex in vertices(mrf.ug)
            haskey(init_sample, Symbol(vertex)) || throw(ArgumentError("Gibbs sample initial_sample must be an assignment with all variables in the Bayes Net"))
        end
        for (vertex, value) in evidence
            init_sample[vertex] == value || throw(ArgumentError("Gibbs sample initial_sample was inconsistent with evidence"))
        end
    end

    if isnull(initial_sample)
        # Hacky
        initial_sample = Assignment()
        for factor in mrf.factors
            for (i, vertex) in enumerate(factor.dimensions)
                n_categories = length(factor.potential[:, i])
                initial_sample[vertex] = rand(Categorical(n_categories))
            end
        end
    end

    # create the data frame
    t = Dict{Symbol, Vector{Any}}()
    for name in keys(initial_sample)
        t[name] = Any[]
    end
    
    # initialize the sample to our initial sample
    current_sample = initial_sample

    # burn in, if present
    if burn_in != 0
        for burn_in_sample in 1:burn_in
            for v in vertices(mrf.ug)
                other_assg = filter((k, val) -> k != Symbol(v), current_sample)
                new_f = reduce(*, [f[other_assg] for f in mrf.factors])
                current_sample = merge(current_sample, rand(new_f))
            end
        end
    end

    # main loop
    for sample_iter in 1:nsamples

        # first skip over the thinning 
        for skip_iter in 1:thinning
            for v in vertices(mrf.ug)
                other_assg = filter((k, val) -> k != Symbol(v), current_sample)
                new_f = reduce(*, [f[other_assg] for f in mrf.factors])
                current_sample = merge(current_sample, rand(new_f))
            end
        end
        
        # real loop, we store the values in the dict
        for v in vertices(mrf.ug)
            other_assg = filter((k, val) -> k != Symbol(v), current_sample)
            new_f = reduce(*, [f[other_assg] for f in mrf.factors])
            current_sample = merge(current_sample, rand(new_f))
            push!(t[Symbol(v)], current_sample[Symbol(v)])
        end
           
    end

    return DataFrame(t)
end

        
