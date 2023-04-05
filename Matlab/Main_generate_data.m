clc
clear
close all

% system parameters
A1 = 2.5; % area of tank 1 (m^2)
A2 = 4.2; % area of tank 2 (m^2)
qin=4; %inflow
R1=.5; %resistence first pipe
R2=.7; %resistence second pipe
maxu=10;
maxh=7;
Q=10;
R=0.5*eye(2);
N=10;
href=10;
nu=2;
nx = 2;


A=[-1/(A1*R1), 1/(A1*R1);
    1/(A2*R1), -1/A2*(1/R1+1/R2)];

B=[1/A1 0; 0 1/A2];

C=eye(2);

D=0;

Ts=0.25;

sys = ss(A,B,C,D);

sysd = c2d(sys,Ts);

Niter = 340;


    
Tsim=Niter*Ts;

n_exp = 50;













x_exp = {};
u_exp = {};
href_exp = {};

u = zeros(n_exp,size(A,1)+1,Niter);
y = zeros(n_exp,size(B,1),Niter);

for exp = 1:n_exp
    
    % initial conditions
    xmin=0.8; xmax=7;
    x0=xmin+rand(nx,1)*(xmax-xmin);
    hmin=0; hmax=12;            
    href =hmin+rand(1,1)*(hmax-hmin);

    Xm=[];
    Um=[];
    Href = [];

    InputconstantFor = 3;

    for i=1:Niter
        if mod(i,InputconstantFor)==0
            hmin=0; hmax=12;
            href =hmin+rand(1,1)*(hmax-hmin);
        end
        unext=mpc_controller(x0, nu, sysd.A,sysd.B, maxu, maxh,N, Q, R, href);
        x1=sysd.A*x0+sysd.B*unext;
        Xm = [Xm;x0'];
        Um = [Um;unext'];
        Href = [Href;href];
        x0 = x1;
    end
    
    
    u_temp = [Xm Href];
    u(exp,:,:)=u_temp';
    y(exp,:,:)=Um';

    x_exp{exp} = Xm;
    u_exp{exp} = Um;
    href_exp{exp} = Href;

end

%%%%%%% VALIDATION


figure
plot(Xm(:,1))
title("x1")

figure
plot(Xm(:,2))
hold on
plot(Href)
title("x2")
legend('h2','href')
hold off



figure

plot(Um(:,1))
title("u1")


figure
plot(Um(:,2))
title("u2")


save('dataset.mat','u','y','Ts','Niter','n_exp')

% 
% figure
% scatter(Um(:,1), Um(:,2))
% title("u1u2")









