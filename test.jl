using BayesNets
using Juno

bn = DiscreteBayesNet()
push!(bn, DiscreteCPD(:sprinkler, ([.2, .8])))
push!(bn, DiscreteCPD(:rain, ([.4, .6])))
push!(bn, DiscreteCPD(:wetgrass, [:sprinkler, :rain], [2,2], [Categorical([.95, .05]),
    Categorical([.9, .1]),
    Categorical([.9, .1]),
    Categorical([.1, .9])]))
is = InferenceState(bn, NodeName(:sprinkler), Assignment(:wetgrass=>1, :rain=>1))
@step infer(LoopyBelief(), is)
