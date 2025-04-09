using LinearAlgebra
using FileIO, JLD2

function A_scal(A, a, b)
    return (A * a)' * b
end

function print_matrix(A, msg="")
    println("Matrix ", msg)
    display(A)
end

function qr_gram_schmidt(A, B, inner_product::Function)
    m, n = size(A)
    Q = zeros(ComplexF64, m, n)
    R = zeros(ComplexF64, n, n)

    for j = 1:n
        v = A[:, j]  

        for i = 1:j-1
            R[i, j] = inner_product(B, Q[:, i], v)
            v = v - R[i, j] * Q[:, i]
        end
        
        for i = 1:j-1
            R[i, j] += inner_product(B, Q[:, i], v)
            v = v - R[i, j] * Q[:, i]
        end

        R[j, j] = sqrt(real(inner_product(B, v, v)))
        Q[:, j] = v / R[j, j]
    end

    return Q, R
end

function check(T, eps)               # find index small diag elem
    n = size(T, 1)
    ind = n + 1
    for i in 1:n
        if abs(T[i, i]) < eps
            ind = i
            break
        end
    end
    return ind
end

function check_orth(A, P, P_prev)
    d1 = diag(P_prev' * A * P_prev)
    d1 = [1 / sqrt(d1[i]) for i in 1:size(d1, 1)]
    D1 = zeros(size(d1, 1), size(d1, 1))

    d2 = diag(P' * A * P)
    d2 = [1 / sqrt(d2[i]) for i in 1:size(d2, 1)]
    D2 = zeros(size(d2, 1), size(d2, 1))

    for i in 1:size(D1, 1)
        D1[i, i] = d1[i]
    end
    for i in 1:size(D2, 1)
        D2[i, i] = d2[i]
    end
    
    T = D2 * P' * A * P_prev * D1
    res = maximum(abs.(T))

    return res
end

function norms2(R)
    norms = Float64[]

    for i in 1:size(R, 2)
        push!(norms, norm(R[:, i])) 
    end    

    return norms
end

function traspose_perm_matrix(a)
    n = size(a, 1)
    b = zeros(Int, n)
    for i in 1:n
        b[a[i]] = i
    end
    return b
end

function scale_columns(A, b)
    """
    res is matrix A: A[:, i] = A[:, i] * b[i] 
    """
    return A * Diagonal(b)
end

function choose_new_residuals(R, m, S_cap, ind, eps)
    r = size(S_cap, 1)
    # S_cap = S[M[1:ind-1]]
    S_wave = setdiff(1:m, S_cap)
    Q_cap, _ = qr(R[:, S_cap])
    TMP = (qr((Q_cap' * R[:, S_wave])[size(S_cap, 1)+1:end, :])).R
    F = qr(TMP, ColumnNorm())
    ind2 = check(F.R, eps)
    take_cols = min(r - ind + 1, ind2)
    return F, take_cols
end


function BCG(A, B, eps)
    n = size(A, 1)
    n1, m = size(B) # amount right sides
    if n1 != n
        println("DimensionMismatch, rows A != rows B")
        exit()
    end
    r = 10          # block size
    k = m           # amount not converged columns

    norms2_B = norms2(B)

    B = scale_columns(B, 1 ./ norms2_B)

    X = zeros(n, m)     # starting value
    R = B - A * X               # n x m

    F = qr(R, ColumnNorm()) 
    S = copy(F.p[1:r])
#    S[1] = 685
#    println("S = ", S)
    P = Matrix(qr(copy(R[:, S])).Q)                   # n x r
    P_prev = copy(P)

    println(size(P))

    converged_columns = Int32[]
    perm = collect(1:m)
    c_norms = Float64[]
    V = Matrix{Float64}(undef, n, 0)
    AV = Matrix{Float64}(undef, n, 0)
    L_VAV = Matrix{Float64}(undef, 0, 0)

    #histP = Matrix{Float64}(undef, n, 0)

    iteration = 1
    maxiter = 5000
    while iteration < maxiter

        if iteration % 1 == 0
            normsR = norms2(R)
            println("iteration = $iteration")
            println("norm2 RS: ", normsR[S])
            println("maximum norm R = ", maximum(normsR))
            if (maximum(normsR)) < eps
#                println("Norms R = ", normsR)
                break
            end
#            println("cnorm B - AX: ", maximum(abs.(R)))
#            println("max(2norm(R[i])): ", maximum(normsR))
        end

        # if iteration > 2 && iteration % 2 == 0
        #     push!(c_norms, check_orth(A, P, P_prev))
        #     P_prev = Peps
        # end
  

        AP = A * P;
        PAP = Hermitian(P' * AP);
        L = cholesky(PAP).L;

#        normP = (L \ copy(P)')';
#        if (size(histP, 2) > 0)
#            println("Dot = ", maximum(abs.((A * normP)' * histP)))
#        end
#        histP = [histP copy(normP)];

        alpha = L' \ (L \ (P' * R)) 
        X += P * alpha              
        R -= AP * alpha             

        alpha2 = L_VAV' \ (L_VAV \ (V' * R));
        X += V * alpha2             
        R -= AV * alpha2            

        alpha = L' \ (L \ (P' * R)) 
        X += P * alpha              
        R -= AP * alpha              

        alpha2 = L_VAV' \ (L_VAV \ (V' * R));
        X += V * alpha2                   
        R -= AV * alpha2               

        

        _, TMP = qr(R[:, S])
        F = qr(TMP, ColumnNorm())                   # PQR
        M = F.p
        T = F.R

        ind = check(T, eps)                              # find index small elem on diag(T)
        if ind == size(T, 1) + 1                    # if all elements on diag(T) is large
            beta = L' \ (L \ (AP' * R[:,S]))
            Pn = R[:, S] - P * beta

            if size(V)[2] > 0
                gamma = L_VAV' \ (L_VAV \ (AV' * R[:, S]))
                Pn -= V * gamma
            end

            beta = L' \ (L \ (AP' * Pn))
            Pn -= P * beta

            if size(V)[2] > 0
                gamma = L_VAV' \ (L_VAV \ (AV' * Pn))
                Pn -= V * gamma
            end
            P = Matrix(qr(Pn).Q);

        else
            print("\nCONVERGE: (iter = $iteration)")

            add_used_cols = [ S[F.p[i]] for i in ind:size(F.p, 1) ]
            converged_columns = [ converged_columns ; add_used_cols ]                      # indexes of columns that converged
            others_cols = setdiff( 1:m, [ converged_columns ; S[ F.p[1:ind-1] ] ] )         # indexes of cols that in R and not in converged_columns and not in current block
            
            if isempty(others_cols) == true                                # stopping criteria
                println("!!! others_cols is empty !!!")                     # if all indexes of cols in converged_columns than all converged than break 
                break
            end


#             if ind > 1
#                 TMP_Q = Matrix(qr(R[:, S[ F.p[1:ind-1] ]]).Q);
#                 TMP = qr(R[:, others_cols] - TMP_Q * (TMP_Q' * R[:, others_cols])).R
#             else
#                 TMP = qr(R[:, others_cols]).R
#             end
#             F = qr(TMP, ColumnNorm())                            # PQR
#             ind2 = check(F.R, eps);
            
#             take_cols = min(r - ind + 1, ind2)     # amount columns to take in block
# #            print(" take_cols = $take_cols ")
#             S_cap = others_cols[ F.p[1:take_cols] ]
#             S = [ S[M[1:ind-1]] ; S_cap]

            G, take_cols = choose_new_residuals(R, m, S, ind, eps)
            other = setdiff(1:m, S)
            S_cap = other[G.p[1:take_cols]]
            S = [ S[M[1:ind-1]] ; S_cap]

            
            Q_cap, _ = qr_gram_schmidt(P[:, M], A, A_scal) # QR decomp with A-scalar product

            if ind > 1
                P_cap = Q_cap[:, 1:ind - 1]
                P_wave = Q_cap[:, ind:size(Q_cap, 2)]

                AP = A * P_cap;
                PAP = Hermitian(AP' * P_cap);
                L = cholesky(PAP);

                nu = L' \ (L \ (AP' * P_wave));
                P_wave -= P_cap * nu;

                V = hcat(V, P_wave)                                          # column concatenation

                AV = A * V
                VAV = Hermitian(V' * AV)
                L_VAV = cholesky(VAV)

                beta = L' \ (L \ (AP' * R[:, S]))
                gamma = L_VAV' \ (L_VAV \ (AV' * R[:, S]))
                P = R[:, S] - P_cap * beta - V * gamma

                beta = L' \ (L \ (AP' * P))
                gamma = L_VAV' \ (L_VAV \ (AV' * P))
                P -= P_cap * beta + V * gamma
            else

                V = [V copy(Q_cap)];                                         # column concatenation

#                println("V' A  norm_P = ", (A * normP)' * V);

                AV = [AV A * Q_cap];
                VAV = Hermitian(V' * AV)
                L_VAV = cholesky(VAV)

                gamma = L_VAV' \ (L_VAV \ (AV' * R[:, S]))
                P = R[:, S] - V * gamma

                gamma = L_VAV' \ (L_VAV \ (AV' * P))
                P -= V * gamma
            end
            
            P = Matrix(qr(P).Q);
            
            println("end")
            #println("cnorm VAP: ",maximum(abs.(AV' * P)))

        end

        iteration += 1
    end
    if iteration == maxiter
        println("!! Reached max iteration !!")
    end
    if isempty(c_norms) == false
        println("max(c_norms of P_{k+2}' A P_k): ", maximum(c_norms))
    end
    X = scale_columns(X, norms2_B)
    
    return X, iteration
end

f = FileIO.load("matrix.jld2");
A = (f["A"]);
B = (f["B"]);;

println("n = ", size(A, 1))
println("right sides = ", size(B, 2))

println("function started")

X, iter = BCG(A, B, 1e-2)

println("iterations = $iter")
#println("C_norm of B - AX is ", maximum(abs.(B - A*X)))
println("max(2norm(B - AX)) is ", maximum(norms2(B - A*X) ./ norms2(B)))
