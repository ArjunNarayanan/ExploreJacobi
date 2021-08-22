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
