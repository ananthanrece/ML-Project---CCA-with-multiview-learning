function [wx,wy] = fn_cca(X,Y,m)

% X = dxN, Y = dxN
[d_x,N_x] = size(X);
[d_y,N_y] = size(Y);

% mu_x = sum(X,2)/N_x;
% mu_y = sum(Y,2)/N_y;
% 
% s_x = X - repmat(mu_x,1,N_x);
% s_y = Y - repmat(mu_y,1,N_y);

s_x = X;
s_y = Y;

% Cxy = ((s_x*s_y')/N_x);
% Cyx = Cxy';
% Cxx = ((s_x*s_x')/N_x) + (eye(d_x));
% Cyy = ((s_y*s_y')/N_y) + (eye(d_y));

Z = [X;Y];
C = cov(Z.');
Cxx = C(1:d_x, 1:d_x) + eye(d_x);
Cxy = C(1:d_x, d_x+1:d_x+d_y);
Cyx = Cxy';
Cyy = C(d_x+1:d_x+d_y,d_x+1:d_x+d_y) + eye(d_y);

Rx = chol(Cxx);
invRx = inv(Rx);
Ry = chol(Cyy);
invRy = inv(Ry);
invCyy = inv(Cyy);

%% method 1
% omega = invRx'*Cxy*invRy;
% 
% [U,S,V]=svd(omega);
% wx = invRx*U;
% wy = invRy*V;
%% alternate 2
omega = invRx*Cxy*invCyy*Cyx*invRx;
omega = (1/2)*(omega'+omega);
[V,D] = eig(omega);
D = sqrt(real(D));
wx = invRx*V;
wy = (invCyy*Cyx*wx)./repmat(diag(D)',d_y,1);


%% alternate 3
% omega = invRx'*Cxy*(Cyy\Cyx)*invRx;
% omega = (1/2)*(omega'+omega);
% 
% [Wx,D] = eig(omega);
% r = sqrt(real(D));
% % Wy = (Cyy\Cyx)*Wx;
% 
% wx = invRx * Wx;
% Wy = (Cyy\Cyx)*wx;
% wy = Wy./repmat(diag(r)',d_y,1);




end