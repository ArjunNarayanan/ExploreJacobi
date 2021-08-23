using SparseArrays

function poisson_linear_system(ndofs)
        operator = spdiagm(
                -1 => 1.0 * ones(ndofs - 1),
                0 => -2.0 * ones(ndofs),
                1 => 1.0 * ones(ndofs - 1),
        )
        operator[1,1] = 1.0
        operator[1,2] = 0.0
        operator[ndofs,ndofs-1] = 0.0
        operator[ndofs,ndofs] = 1.0
        return operator
end

function solve_direct(operator, rhs, bcleft, bcright, stepsize)
    numpts = length(rhs)
    copyrhs = copy(rhs)
    copyrhs[1] = bcleft / stepsize^2
    copyrhs[numpts] = bcright / stepsize^2
    sol = stepsize^2 * (operator \ copyrhs)
    return sol
end

function solve_direct_all_rhs(operator, bcleft, bcright, stepsize)
    ndofs, ndofs = size(operator)
    rhs = diagm(ones(ndofs))[:, 2:ndofs-1]
    rhs[1, :] .= bcleft / stepsize^2
    rhs[ndofs, :] .= bcright / stepsize^2
    sol = stepsize^2 * (operator \ rhs)
    return sol
end
