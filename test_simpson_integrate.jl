using Test
include("simpson_integrate.jl")

testfunc(x) = 4.0x^2 + 2.0x - 1.0
stepsize = 1e-2
xrange = 0:stepsize:1
funcvals = testfunc.(xrange)
integral = simpson_integrate(funcvals,stepsize)
exactintegral = 4.0/3
@test integral ≈ exactintegral
