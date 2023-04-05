function [unext] = mpc_controller(x0, nu, A,B, maxu, maxh,N, Q, R, href)
u=sdpvar(repmat(nu,1,N),ones(1,N));
xa=sdpvar(2,1);
ua=sdpvar(2,1);
constraints=[];
objective=0;

for i=1:N
    x1=A*x0+B*u{i};
    constraints=[constraints; [0;0]<=x1<=[maxh;maxh]; 0<=u{i}<=maxu];
    objective=objective+(x1(2)-xa(2))'*Q*(x1(2)-xa(2))+(u{i}-ua)'*R*(u{i}-ua);
    x0=x1;
end

constraints=[constraints; xa==A*xa+B*ua; x1==xa; ua==u{N}];
objective= objective+(href-xa(2))'*10*Q*(href-xa(2));

ops=sdpsettings('verbose', 1, 'solver', 'quadprog');
diagnostic=optimize(constraints,objective,ops);
unext=value(u{1});

end
