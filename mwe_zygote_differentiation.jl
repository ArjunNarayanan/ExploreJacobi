function step!(unext, uprev, parameters)
    wL, wC, wR = parameters
    numpts = length(unext)
    for i = 2:numpts-1
        unext[i] = wL * uprev[i-1] + wC * uprev[i] + wR * uprev[i]
    end
end

function single_step_error(parameters, u0, uexact)
    T = eltype(parameters)
    unext = zeros(T,length(u0))
    step!(unext, u0, parameters)
    err = sum((unext - uexact) .^ 2)
    return err
end

numpts = 10
u0 = ones(numpts)
uexact = range(0.0,stop=1.0,length=numpts)
parameters = [0.,0.5,0.8]
err = single_step_error(parameters,u0,uexact)

using ForwardDiff
df = ForwardDiff.gradient(x->single_step_error(x,u0,uexact),parameters)
