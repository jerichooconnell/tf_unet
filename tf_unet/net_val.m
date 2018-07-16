function ys = net_val(x)

load('net')

xnew = reshape(x,numel(x)./4,4);

ys = net(xnew');

end