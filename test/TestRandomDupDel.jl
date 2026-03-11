# Test RandomDuplicateDelete

rdd = RandomDuplicateDelete(Random.default_rng(), Bernoulli(0.2), Bernoulli(0.2))

for t in iterate((rdd, 1:100))
    print("Saw $t\n")
end
