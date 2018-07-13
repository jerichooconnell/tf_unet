function ys = net_val(x)

load('net')

xnew = reshape(x,4,numel(x)/4);

ys = net(xnew);

end