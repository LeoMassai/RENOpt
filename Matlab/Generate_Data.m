clc
clear
close all

% system parameters
A1 = 2.5; % area of tank 1 (m^2)
A2 = 4.2; % area of tank 2 (m^2)
qin=4; %inflow
R1=.5; %resistence first pipe
R2=.7; %resistence second pipe
maxu=5;
maxh=7;
Q=10;
R=0.5*eye(2);
N=10;
href=10;
nu=2;


A=[-1/(A1*R1), 1/(A1*R1);
    1/(A2*R1), -1/A2*(1/R1+1/R2)];

B=[1/A1 0; 0 1/A2];

C=eye(2);

D=0;

Ts=0.25;

sys = ss(A,B,C,D);

sysd = c2d(sys,Ts);

h0=[2 5]';

T=90;
numt=100;
X=[];
U=zeros(numt,2,T/Ts);

y=zeros(numt,T/Ts);


for j=1:numt
    X=[];
    h0=[2 5]';
    U(j,:,:)=10*rand(2,T/Ts);

    for i=1:T/Ts
        h1 = sysd.A*h0+sysd.B*U(j,:,i)';
        X = [X;h0'];
        h0 = h1;

    end

    y(j,:)=X(:,2)';
end

u=U;

t=0:Ts:T-Ts;

save('datasetb', "u", "y", "Ts")




figure;plot(0:Ts:T-Ts,X(:,1));title('Tank1')

figure;plot(0:Ts:T-Ts,X(:,2));title('Tank2')

figure;plot(0:Ts:T-Ts,X(:,1));title('Tank1')




% Tsim=160;
% x0=[2 5]';
% Xm=[];
% Um=[];
%
% hrefv=zeros(size(1:Tsim/Ts,2));
% for i=1:Tsim/Ts
%
%     if mod(i,50)==0
%         href=9*rand+1;
%     end
%
%     unext=mpctankv(x0, nu, sysd.A,sysd.B, maxu, maxh,N, Q, R, href);
%     x1=sysd.A*x0+sysd.B*unext;
%     Xm = [Xm;x0'];
%     Um = [Um;unext'];
%     x0 = x1;
%     hrefv(i)=href;
% end
%
%
% figure
% plot(Xm(:,1))
% title("x1")
%
% figure
% plot(Xm(:,2))
% hold on
% plot(hrefv)
% title("x2")
% hold off
%
%
%
% figure
% plot(Um(:,1))
% title("u1")
%
%
% figure
% plot(Um(:,2))
% title("u2")
%
%
% figure
% scatter(Um(:,1), Um(:,2))
% title("u1u2")









