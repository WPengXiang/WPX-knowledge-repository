function [P,T,Pb,Tb,element_order]=information_input(a,b,N,element_order)
%information_input(a,b,N)  ���ڶ�ȡ����͵�Ԫ��Ϣ
%a:��˵�����
%b:�Ҷ˵�����asasd
%N:��Ԫ����
%P:����ڵ�������Ϣ����for trial function
%T:����Ԫ��Ϣ����for trial function
%Pb:����Ԫ�ڵ�������Ϣ����for test function
%Tb:����Ԫ��Ԫ��Ϣ����for test function
%�����Ƿ�Ϊ���Ե�Ԫ�ж�P��T�Ƿ����Pb��Tb
%element_order:��Ԫ�״�
%  �ż�

%���Ե�Ԫ
if element_order==1
P=sparse(1,N+1);
T=sparse(2,N);
Pb=sparse(1,N+1);
Tb=sparse(2,N);
P=linspace(a,b,N+1);
T=zeros(2,N);
Pb=P;
T(1,:)=1:N;
T(2,:)=2:N+1;
Tb=T;
else    %��ʱû�б�д
    P=sparse(1,N+1);
    T=sparse(2,N);
    Pb=sparse(1,(N+1)*element_order);
    T=sparse(2,N);
    P=linspace(a,b,N+1);
    T=zeros(2,N);
    Pb=P;
    Tb=T;
    Tb(1,:)=1:N;
    Tb(2,:)=2:N+1;
end
%Draw node label
xx=Pb;
yy=zeros(1,N+1);
plot(xx,yy,'-o','LineWidth',2)
for i=1:size(Pb,2);
    tempstr=['' int2str(i)];
    text(xx(i),yy(i),tempstr,'Color',[1 0 0],'FontSize',14,'HorizontalAlignment','right');
end
end
