function ys = net_val(x)

load('/home/jericho/Downloads/bin/pcd_planar_imaging/DES_radiography/DES_3.0/net.mat')

xnew = reshape(x,numel(x)./4,4);

ys = net(xnew');

end