 
function [A]=stiffness_Algorithm4(Pb,Tb,r,s,element_order,N)
%ʹ�ú���ʦ����֮����㷨4���γ�����նȾ���
%Pb:����Ԫ�ڵ�������Ϣ����for test function
%Tb:����Ԫ��Ԫ��Ϣ����for test function
%r:��̽�������󵼽״�
%s:���Ժ������󵼽״�
%element_order:��Ԫ�״�
%Pb��Tb��ǰ���information_input����
%KΪ�����ȫ�ָնȾ���
%�ż���
if r==1 && s==1    %��ʱû�������ôд
if element_order==1
    Nb_trial=size(Tb,1);%ÿ����Ԫ�ڵ�����Ļ�����������������Tb��������ȡ
    Nb_test=size(Tb,1); %ÿ����Ԫ�ڵ�����Ļ�����������������Tb��������ȡ
    A=sparse(size(Pb,2),size(Pb,2));
    for i=1:N
        T_current=Tb(:,i);
        phi_n1=-1/(Pb(T_current(2))-Pb(T_current(1)));
        phi_n2=1/(Pb(T_current(2))-Pb(T_current(1)));
        N_shape_matrix_deriv=[phi_n1,phi_n2];
        c_e=exp(Pb(T_current(2)))-exp(Pb(T_current(1)));
        for m=1: Nb_trial
            for n=1: Nb_test
            r=N_shape_matrix_deriv(m)*N_shape_matrix_deriv(n)*c_e;
            A(Tb(m,i),Tb(n,i))=r+A(Tb(m,i),Tb(n,i)); % A(Tb(n,i),Tb(m,i))Ҳ����
            end
        end
    end
end
end 
if element_order==2
    Nb_trial=size(Pb,2);
    Nb_test=size(Pb,2);
    A=sparse(Nb_trial,Nb_test);
    for i=1:N
        T_current=Tb(:,i);
        phi_n1=-1/(Pb(T_current(2))-Pb(T_current(1)));
        phi_n2=1/(Pb(T_current(2))-Pb(T_current(1)));
        N_shape_matrix_deriv=[phi_n1,phi_n2];
        c_e=-(exp(T_current(2))-exp(T_current(1)));
        for m=1: Nb_trial
            for n=1: Nb_test
            r=N_shape_matrix_deriv(Nb_trial)*N_shape_matrix_deriv(Nb_test)*c_e;
            A(Tb(m,i),Tb(n,i))=r
            end
        end
    end
end
end
 