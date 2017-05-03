using PyPlot
x = readdlm("trx.dat");
y = readdlm("try.dat");
(m,n) = size(x);
x = reshape(x, m, n);
x = [ones(m,1) x];

theta = zeros(n+1,1);
function g(n)
    return 1.0 ./(1.0+ exp(-n))
end

Iter = 10;
J = zeros(Iter, 1);

# Loop
for i in 1:Iter
    # Calculate the hypothesis fucntion
    z = x*theta;
    # Calculate sigmoid
    h = g(z);
    # Calculate gradient and hession.
    # The formulas below are equivalent to the summation formulars
    # Given in the lecture videos.
    grad = (1/m) .* x' * (h-y);
    H = (1/m) .* x' * diagm(vec(h)) * diagm(vec(1-h)) * x;
    # Calculate J for testing convergence
    J[i] = (1/m)*sum(-y .* log(h) - (1-y) .* log(1-h));
    theta = theta - H\grad;
end
print(theta)

print(J)
print(size(theta))
print( size(x))

smx = -10:10
smy = g(smx)
plot(smx,smy)
tex = readdlm("tex.dat");
tey = readdlm("tey.dat");
#print(size(x));
#print(size(y));
#
(m,n) = size(tex);
tex = reshape(tex, m, n);
# add x_0 (bias)
tex = [ones(m,1) tex];
println(size(tex));

smx = -10:10
smy = g(smx)

#plot(smx,smy)

resx =  tex * theta;
resy = g(resx);

println(size(resx),size(resy))

plot(resx[1:20],resy[1:20],"r*")
plot(resx[21:40],resy[21:40],"bo")

succ = 0.
fail = 0.
all = 0.
for i in 1:20
    if resx[i] >= 0.5
        succ +=1
    end
end
println(100. *succ/20)
for i in 21:40
    if resx[i] <0.5
        fail +=1
    end
end
println(100. *fail/20)
println(100 *(succ + fail)/40)