using FileIO
f = FileIO.load("alm.jld2");
A = f["A"];
B = f["B"];
A = Array{ComplexF64}(A);
B = Array{ComplexF64}(B);
A = A * A'
save("matrix.jld2", Dict("A" => A, "B" => B))