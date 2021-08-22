using LinearAlgebra
include("poisson_linear_system.jl")

function step_jacobi!(unext, uprev, rhs, stepsize)
    numpts = length(unext)
    for i = 2:numpts-1
        unext[i] = 0.5 * (uprev[i-1] + uprev[i+1] + stepsize^2 * rhs[i])
    end
end

function run_jacobi_iterations(u0, rhs, stepsize, numiter)
    uprev = copy(u0)
    unext = copy(u0)
    for iter = 1:numiter
        step_jacobi!(unext, uprev, rhs, stepsize)
        uprev .= unext
    end
    return unext
end

function run_jacobi_measure_error(u0, rhs, stepsize, numiter, uexact)
    numpts = length(u0)
    uprev = copy(u0)
    unext = copy(u0)
    err = zeros(numiter)
    for iter = 1:numiter
        step_jacobi!(unext, uprev, rhs, stepsize)
        uprev .= unext
        err[iter] = norm(unext - uexact) / numpts
    end
    return unext, err
end

function solve_direct(rhs, bcleft, bcright, stepsize)
    numpts = length(rhs)
    op = -poisson_linear_system(numpts)
    copyrhs = copy(rhs)
    copyrhs[1] = bcleft / stepsize^2
    copyrhs[numpts] = bcright / stepsize^2
    sol = stepsize^2 * (op \ copyrhs)
    return sol
end

function jacobi_error_vs_iterations(numiter, numelements, wavenumber)
    xL = -1.0
    xR = +1.0
    bcleft = bcright = 0.0

    numpts = numelements + 1
    stepsize = (xR - xL) / numelements
    xrange = range(xL, stop = xR, length = numpts)
    rhs = cos.(wavenumber * xrange)

    u0 = zeros(numpts)
    udirect = solve_direct(rhs, bcleft, bcright, stepsize)

    ujacobi, err = run_jacobi_measure_error(u0, rhs, stepsize, numiter, udirect)

    return err
end

numelements = [4, 8, 16, 32]
numiter = numelements .^ 2
wavenumber = 2pi
err = [
    jacobi_error_vs_iterations(numiter[i], numelements[i], wavenumber) for
    i = 1:length(numelements)
]


using Plots
plot( err[1], yscale = :log10, label = string(numelements[1]))
plot!(err[2], yscale = :log10, label = string(numelements[2]))
plot!(err[3], yscale = :log10, label = string(numelements[3]))
plot!(err[4], yscale = :log10, label = string(numelements[4]))
plot!(err[5], yscale = :log10, label = string(numelements[5]))
