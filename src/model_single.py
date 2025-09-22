import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

from scipy.signal import find_peaks
import os.path



class ComplexModel(nn.Module):
    def __init__(self, params):
        super(ComplexModel, self).__init__()
        self.params=params
        self.models=nn.ModuleList([CustomModel(*p) for p in params[1:]])

        self.r_global=torch.nn.Parameter(params[0], requires_grad=True)
        
            
    def forward(self,target_Hz=torch.tensor([0])):
        res=[]
        protons=[]
        for m in self.models:
            protons.append(m.proton_config.sum().long())
        sm=F.softmax(self.r_global,dim=0).reshape(-1,1)
        for i,m in enumerate(self.models):
            r=m(target_Hz)
            res.append(r[:,1])
        return torch.stack((r[:,0], (torch.stack(res)*sm).sum(dim=0)), -1)
    

class CustomModel(nn.Module):
    # J here is nmrsim-like matrix - zero diagonal, symmetric
    def __init__(self, v, J, w, r,protons_number,freq,ppm0):
        super(CustomModel, self).__init__()
        self.v0 = torch.nn.Parameter(v, requires_grad=True)
        self.J0 = torch.nn.Parameter(J, requires_grad=True)
        self.w0 = torch.nn.Parameter(w, requires_grad=True)
        self.r0 = torch.nn.Parameter(r, requires_grad=True)
        self.PPrint = False
        self._nuclei_number = protons_number.sum().long() if protons_number.shape[0]>1 else torch.Tensor([1]).long()
        self.proton_config=torch.nn.Parameter(protons_number, requires_grad=False)
        self.proton_config_short=protons_number if protons_number.shape[0]>1 else torch.Tensor([1])
        self.freq=torch.nn.Parameter(freq, requires_grad=False)
        self.ppm0=torch.nn.Parameter(ppm0, requires_grad=False)
        #self.v0 = torch.nn.Parameter(ppm0*499.84, requires_grad=True)
        self.v0 = torch.nn.Parameter(ppm0*freq, requires_grad=True)
        
#         print(freq)
        self.mat1=torch.zeros((self.proton_config_short.shape[0],self.proton_config_short.sum().numpy().astype(int)),dtype=torch.double)
        ids=torch.cumsum(self.proton_config_short,dim=0).long()
        self.mat1[0,:ids[0]]=1
        if ids.shape[0]>1:
            for i in range(1,ids.shape[0]):
                self.mat1[i,ids[i-1]:ids[i]]=1
        self.T1, self.T2 = self._transition_matrix_dense(self._nuclei_number)
        self.Lz, self.Lproduct = self._so_dense(self._nuclei_number)
        
        
    def initiate(self):
        self.mat1=torch.zeros((self.proton_config_short.shape[0],self.proton_config_short.sum().numpy().astype(int)),dtype=torch.double)
        ids=torch.cumsum(self.proton_config_short,dim=0).long()
        self.mat1[0,:ids[0]]=1
        if ids.shape[0]>1:
            for i in range(1,ids.shape[0]):
                self.mat1[i,ids[i-1]:ids[i]]=1

        self.mat2=self.mat1.T
        
    def power_iteration(self,A, num_iters=1000):
        """
        Power iteration method for approximating the dominant eigenvector of a matrix.
        """
        n = A.size(0)
        b = torch.randn(n, 1,requires_grad=True)

        for _ in range(num_iters):
            b = F.normalize(torch.matmul(A, b), dim=0)

        return b
    def secondorder_dense(self, normalize=True, **kwargs):
        nspins = self._nuclei_number
        if self.PPrint:    
            print("nspins = ", nspins)
        H = self.hamiltonian_dense()#.real
        if self.PPrint:    
            print("H = ", H)
        with torch.no_grad():
            E0, V0 = torch.linalg.eig(H) 
        if self.v0.requires_grad:
            E = torch.linalg.eigvals(H)
        else:
            E = torch.linalg.eigvals(H.requires_grad_(True)).detach()
        eigenvectors = []
        for el in V0.reshape(-1).real:
            if torch.isinf(el) or el==0 or torch.isnan(el):
                eigenvectors.append(torch.Tensor([0]))
            else:
                eigenvectors.append(el)
        V = torch.Tensor(eigenvectors).reshape(2**self._nuclei_number,2**self._nuclei_number)
        if self.PPrint:    
            print("E = ", E)
            print("V = ", V)
        V = V.real
        I = torch.square(torch.mm(V.T, torch.mm(self.T1, V)))
        W = torch.square(torch.mm(V.T, torch.mm(self.T2, V)))
        if self.PPrint:    
            print("I = ", I)
            print("W = ", W)
        
            
        peaklist = self._compile_peaklist(I, E, W, **kwargs)
        
        if self.PPrint:    
            print("peaklist = ", peaklist)
        if normalize:
            peaklist = self.normalize_peaklist(peaklist, nspins)
        return peaklist

    def hamiltonian_dense(self):
        
        nspins = self._nuclei_number
        Lz = self.Lz.double()

        H = torch.tensordot(torch.mm(self.v0.reshape(1,-1),self.mat1).squeeze(0).double(), Lz, dims=1)
        H = H.to(torch.complex64)

        J_processed=torch.mm(self.mat2,torch.mm(self.J0,self.mat1))
        scalars = 0.5 * J_processed
        scalars = scalars.to(torch.complex64)
        H = H + torch.tensordot(scalars, self.Lproduct, dims=2)

        return H

    def _so_dense(self, nspins):
        
        sigma_x = torch.tensor([[0, 1 / 2], [1 / 2, 0]])
        real = torch.tensor([[0, 0], [0, 0]], dtype=torch.float32)
        imag = torch.tensor([[0, -1 / 2], [1 / 2, 0]], dtype=torch.float32)
        sigma_y = torch.complex(real, imag)
        sigma_z = torch.tensor([[1 / 2, 0], [0, -1 / 2]])
        unit = torch.tensor([[1, 0], [0, 1]])
        L = torch.empty((3, nspins, 2 ** nspins, 2 ** nspins),
                        dtype=torch.cfloat)
        for n in range(nspins):
            Lx_current = torch.tensor([1])
            Ly_current = torch.tensor([1])
            Lz_current = torch.tensor([1])

            for k in range(nspins):
                if k == n:
                    Lx_current = torch.kron(Lx_current, sigma_x)
                    Ly_current = torch.kron(Ly_current, sigma_y)
                    Lz_current = torch.kron(Lz_current, sigma_z)
                else:
                    Lx_current = torch.kron(Lx_current, unit)
                    Ly_current = torch.kron(Ly_current, unit)
                    Lz_current = torch.kron(Lz_current, unit)

            L[0][n] = Lx_current
            L[1][n] = Ly_current
            L[2][n] = Lz_current

        L_T = torch.transpose(L, 1, 0)
        Lproduct = torch.tensordot(L_T, L, dims=([1, 3], [0, 2])).swapaxes(1, 2)
        return L[2], Lproduct

    def _transition_matrix_dense(self, nspins):
        
        n = 2 ** nspins
        T1 = torch.zeros((n, n))
        T2 = torch.zeros((n, n))
        for i in range(n-1):
            for j in range(i+1,n):
                if bin(i ^ j).count('1') == 1:
                    rr=torch.mm(torch.abs(self.r0).reshape(1,-1),self.mat1).squeeze(0)[int(nspins - 1 - torch.log2(torch.tensor(j - i)))] #if self.r0.shape[0]>1 else torch.abs(self.r0)[int(nspins - 1 - torch.log2(torch.tensor(j - i)))]
                    T1[i, j] = torch.sqrt(rr)
                    T2[i, j] = torch.sqrt(torch.mm(torch.abs(self.w0).reshape(1,-1),self.mat1).squeeze(0)[int(nspins - 1 - torch.log2(torch.tensor(j - i)))]*rr)# if self.r0.shape[0]>1 else torch.sqrt(torch.abs(self.w0)[int(nspins - 1 - torch.log2(torch.tensor(j - i)))]*rr) 
        if self.PPrint:    
            print("T1 = ", T1)
            print("T2 = ", T2)
        return T1+T1.T-2*(T1.detach().clone())*torch.eye(T1.shape[0]),T2+T2.T-2*(T2.detach().clone())*torch.eye(T2.shape[0])

    def _compile_peaklist(self, I, E, W, cutoff=0.001):
        I_upper = torch.triu(I)
        W_upper = torch.triu(W)
        E_matrix = torch.abs(E[:, None] - E)
        E_upper = torch.triu(E_matrix)
        if self.PPrint:    
            print("I_upper = ", I_upper)
            print("W_upper = ", W_upper)
            print("E_matrix = ", E_matrix)
            print("E_upper = ", E_upper)
        
        
        combo = torch.stack([E_upper, I_upper, W_upper])
        iv = combo.reshape(3, I.shape[0] ** 2).T
        result = iv[iv[:, 1] >= cutoff]
        if self.PPrint:    
            print("result = ", result)
        res=result[:,2]/result[:,1]
        resu = torch.hstack((result[:,:1],result[:,1:2],res.reshape(-1,1)))
        return resu
        #return resu[resu[:,2]<100]

    def normalize_peaklist(self, peaklist, n=1):
        
        freq, int_, w = peaklist[:, 0], peaklist[:, 1], peaklist[:, 2]
        self._normalize(int_, n)
        t = torch.stack((freq, int_, w), dim=1)
        return t

    def _normalize(self, intensities, n=1):
        
        factor = n / torch.sum(intensities)

        return factor * intensities

    def add_lorentzians(self, linspace, peaklist):
        
        result = self.lorentz(linspace, peaklist[0][0], peaklist[0][1], peaklist[0][2])
        for v, i, w in peaklist[1:]:
            result = result + self.lorentz(linspace, v, i, w)
        #         print('add_lor', result)

        return result

    def lorentz(self, v, v0, I, w):
        
        scaling_factor = 0.5 / w  # i.e. a 1 Hz wide peak will be half as high

        return scaling_factor * I * ((0.5 * w) ** 2 / ((0.5 * w) ** 2 + (v - v0) ** 2))

    def low_high(self, t):
        
        return torch.min(t[0], t[1]), torch.max(t[0], t[1])

    def __len__(self):
        return len(v)
    
    def trapz_diff(self, y, x):
        '''
        Расчет интеграла, такая имплементация дифференциируема в torch
        '''
        dx = x[1:] - x[:-1]
        dy = (y[1:] + y[:-1]) / 2
        area = dx * dy
        integral = torch.sum(area)
        return abs(integral)

    def forward(self, target_Hz=torch.tensor([0])):
        '''
        Метод forward, который определяет, как модель self применяется к входным данным.
        Если на вход подаются точки target_Hz, то спектр генерируется в этих точках, а если нет, то в захардкоженных пределах 
        '''
        self.initiate()
        self.T1, self.T2 = self._transition_matrix_dense(self._nuclei_number)
        self.Lz, self.Lproduct = self._so_dense(self._nuclei_number)
        peaklist = self.secondorder_dense()
        first_column = peaklist[:, 0]
        sorted_indices = torch.argsort(first_column, descending=True)
        peaklist_ = peaklist[sorted_indices]
       
        
        # number of points influence on peak multiplicity
        points = 12758
        limits = [torch.tensor(550), torch.tensor(2500)]

        if limits:
            l_limit, r_limit = self.low_high(limits)
        else:
            l_limit = peaklist[0][0] - 50
            r_limit = peaklist[-1][0] + 50

        xx = torch.linspace(l_limit, r_limit, points)
        yy = self.add_lorentzians(xx, peaklist_)

        iintegral = self.trapz_diff(yy, xx)
        
        if target_Hz.shape[0]>1:        
            x_ = torch.FloatTensor(target_Hz)
            y = self.add_lorentzians(x_, peaklist_)
            iintegral = self.trapz_diff(y, x_)
            
        else:
            x_ = xx
            y = yy
        y_norm = torch.FloatTensor(torch.div(y, iintegral))

        out = torch.stack((x_, y_norm), -1)

        return out
    
