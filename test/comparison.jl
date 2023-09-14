cd(@__DIR__)
using Pkg
Pkg.activate("..")
using NLsolve, NonlinearSolve, LinearAlgebra, FileIO

# Default options for NLsolve 
# ? 

# Default options for NonlinearSolve.TrustRegion()
# ?

# powell's test function
function powell_singular_function!(out, x, p, io)
    out[1] = x[1] + 10.0 * x[2]
    out[2] = sqrt(5.0) * ( x[3] - x[4] )
    out[3] = ( x[2] - 2.0 * x[3] )^2
    out[4] = sqrt(10.0) * ( x[1] - x[4] ) * ( x[1] - x[4] )
    if out isa Array{Float64}
        println(io, out)
    end
    nothing
end

n = 4
x = zeros(n)
x[1]   = -1.2
x[2:n] .= 1.0
out = similar(x)

io = stdout # open("nr_out.txt", "w") # stdout
func! = (out,x,p=nothing) -> powell_singular_function!(out,x,p,io)
nlprob = NonlinearProblem(func!, x)
sol_newton = solve(nlprob, NewtonRaphson(), abstol=1e-15, reltol=1e-15)
func!(out, sol_newton.u)
println(io, "NewtonRaphson residual norm: $(norm(out))")
#close(io)

io = stdout #open("tr_out.txt","w") 
func! = (out,x,p=nothing) -> powell_singular_function!(out,x,p,io)
nlprob = NonlinearProblem(func!, x)
tr_alg = TrustRegion(step_threshold = 1e-4)
sol_tr = solve(nlprob, tr_alg, abstol=1e-15, reltol=1e-15)
func!(out, sol_tr.u)
println(io, "TrustRegion residual norm: $(norm(out))")
#close(io)

io = stdout #open("nlsolve_out.txt","w") 
func! = (out,x,p=nothing) -> powell_singular_function!(out,x,p,io)
sol = nlsolve(func!, x, xtol=1e-15, ftol=1e-15)
println(io,"NLsolve residual norm: $(sol.residual_norm)")
#close(io)

#freudenstein roth test function -> even more atrocious 
function freudenstein_roth!(out, x, p = nothing)
    out[1] = x[1] - x[2]^3 + 5.0 * x[2]^2 - 2.0 * x[2] - 13.0
    out[2] = x[1] + x[2]^3 + x[2]^2 - 14.0 * x[2] - 29.0
end
x = [1.0; 0.0]
nlprob = NonlinearProblem(freudenstein_roth!, x)
out = similar(x)
freudenstein_roth!(out, solve(nlprob, NewtonRaphson(), abstol=1e-15, reltol=1e-15).u, nothing)
println("NewtonRaphson residual norm: $(norm(out))")
freudenstein_roth!(out, solve(nlprob, TrustRegion(step_threshold=1e-4), abstol=1e-15, reltol=1e-15).u, nothing)
println("TrustRegion residual norm: $(norm(out))")
sol = nlsolve(freudenstein_roth!, x, xtol=1e-15, ftol=1e-15)
println("NLsolve residual norm: $(sol.residual_norm)")
