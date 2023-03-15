function [unext] = mpctank(x0, nu, A,B, maxu, maxh,N, Q, R, href)
u=sdpvar(repmat(nu,1,N),ones(1,N));
constraints=[];
objective=0;


for i=1:N
    x1=A*x0+B*u{i};
    constraints=[constraints; [0;0]<=x1<=[3.2;maxh]; 0<=u{i}<=maxu];
    objective=objective+(x1(2)-href)'*Q*(x1(2)-href)+u{i}'*R*u{i};
    x0=x1;


end

constraints=[constraints; x1==A*x1+B*u{N}];

ops=sdpsettings('verbose', 1, 'solver', 'quadprog');
diagnostic=optimize(constraints,objective,ops);
unext=value(u{1});



end

