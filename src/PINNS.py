import DNN
from time import perf_counter
from DNN import torch,np,plt
from torch import cos,sin,tan,cosh,sinh,tanh,pow,exp,log,log10
class ODE():
    def __init__(self,x1:float,x2:float,size:int):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.x = torch.linspace(x1, x2, size,requires_grad=True).unsqueeze(-1).to(self.device)
        self.start = x1
        self.end = x2
        self.size = size
        self.y_train = torch.rand(self.size).to(self.device)
        self.dx = torch.rand(self.size).to(self.device)
        self.dxx = torch.rand(self.size).to(self.device)
        self.dxxx = torch.rand(self.size).to(self.device)
        self.diff_left = torch.zeros(self.size).to(self.device)
        self.equ_terms = ""
        self.boundary_diff = []
        self.boundary_valuex = []
        self.boundary_valuey = []
        self.MSEf = []
        self.MSEu = None
        self.MSE = None
        self.MSE_part = None
        self.network = None
        self.optimizer = None
        self.equ_coeffs = []
        self.diff_parts = []
    
    def constant(self,c):
        return torch.from_numpy(np.array([c]*self.size)).view(self.size,1).float().to(self.device)

    def xVar(self):
        return self.x

    def yVar(self):
        return self.y_train
    
    def diff(self,level:int,diff_coeff):
        if type(diff_coeff) == torch.Tensor:   
            self.diff_coeff = diff_coeff
        else:
            self.diff_coeff = self.constant(diff_coeff)
        if level == 1:
            return self.dx
        if level == 2:
            return self.dxx
        if level == 3:
            return self.dxxx
        raise ValueError("导数需小于三阶！")

    def diffInSystems(self,level:int,x:torch.Tensor):
        if level == 0:
            self.y_train = self.network(x)
            return self.y_train
        if level == 1:
            self.y_train = self.network(x)
            self.dx = torch.autograd.grad(self.network(x), x, grad_outputs=torch.ones_like(self.y_train), create_graph=True)[0]
            return self.dx
        if level == 2:
            self.y_train = self.network(x)
            self.dx = torch.autograd.grad(self.network(x), x, grad_outputs=torch.ones_like(self.y_train), create_graph=True)[0]
            self.dxx = torch.autograd.grad(self.dx,x,grad_outputs=torch.ones_like(self.dx), create_graph=True)[0]
            return self.dxx
        if level == 3:
            self.y_train = self.network(x)
            self.dx = torch.autograd.grad(self.network(x), x, grad_outputs=torch.ones_like(self.y_train), create_graph=True)[0]
            self.dxx = torch.autograd.grad(self.dx,x,grad_outputs=torch.ones_like(self.dx), create_graph=True)[0]
            self.dxxx = torch.autograd.grad(self.dxx,x,grad_outputs=torch.ones_like(self.dxx), create_graph=True)[0]
            return self.dxxx
        raise ValueError("导数需小于三阶！")

    def setEquation(self,coeffs:list,variables:list,diffs:list,terms:str):
        if variables[0] != None:
            self.variables = variables
            for i in range(len(coeffs)):
                if type(coeffs[i]) == torch.Tensor:
                    self.equ_coeffs.append(coeffs[i].view(self.size,1).to(self.device))
                else:
                    self.equ_coeffs.append(torch.from_numpy(np.array([coeffs[i]]*self.size)).view(self.size,1).to(self.device))
            self.equ_diffs = diffs
            self.equ_terms = terms.replace("T","self.x").replace("y","ODE2.y_train").replace("x", "self.y_train").replace("z","ODE3.y_train").replace("ODE.constant","self.constant")
        else:
            for i in range(len(coeffs)):
                if type(coeffs[i]) == torch.Tensor:
                    if(self.dx.equal(coeffs[i]) or self.dxx.equal(coeffs[i]) or self.dxxx.equal(coeffs[i])):
                        if(self.dx.equal(coeffs[i])):
                            self.equ_coeffs.append("self.dx")
                        if(self.dxx.equal(coeffs[i])):
                            self.equ_coeffs.append("self.dxx")
                        if(self.dxxx.equal(coeffs[i])):
                            self.equ_coeffs.append("self.dxxx")
                    else:
                        self.equ_coeffs.append(coeffs[i].view(self.size,1).to(self.device))
                else:
                    self.equ_coeffs.append(torch.from_numpy(np.array([coeffs[i]]*self.size)).view(self.size,1).to(self.device))
            self.equ_diffs = diffs
            self.equ_terms = terms.replace("x", "self.x").replace("y","self.y_train").replace("ODE","self")
    
    def setExact(self,exactY:torch.Tensor,exactf:str):
        if exactY == None:
            self.show_exact_flag = False
            self.y = None
        else:
            self.show_exact_flag = True
            self.y = exactY.to(self.device)
        if exactf == None:
            self.f = None
        else:
            self.f = exactf

    def addBoundary(self,diff:float,valuex:float,valuey:float):
        self.boundary_diff.append(diff)
        self.boundary_valuex.append(valuex)
        self.boundary_valuey.append(valuey)
    
    def switch(self,_value, *_cases):
        _script = ''

        for _var in globals().keys():
            _script += _var + ','
        _script = 'global ' + _script + '__name__\n'

        for _case in _cases:
            if _case[0] in (_value, None):
                for _statement in _case[1:]:
                    _script += _statement + '\n'
                exec(_script)
                break
    
    def setLoss(self):
        for i in range(len(self.equ_diffs)):
            self.switch(self.equ_diffs[i],
            (0,'self.y_train = self.network(self.x)','self.diff_parts.append(self.y_train)'),
            (1,'self.y_train = self.network(self.x)','self.dx = torch.autograd.grad(self.network(self.x), self.x, grad_outputs=torch.ones_like(self.y_train), create_graph=True)[0]','self.diff_parts.append(self.dx)'),
            (2,'self.y_train = self.network(self.x)','self.dx = torch.autograd.grad(self.network(self.x), self.x, grad_outputs=torch.ones_like(self.y_train), create_graph=True)[0]','self.dxx = torch.autograd.grad(self.dx,self.x,grad_outputs=torch.ones_like(self.dx), create_graph=True)[0]','self.diff_parts.append(self.dxx)'),
            (3,'self.y_train = self.network(self.x)','self.dx = torch.autograd.grad(self.network(self.x), self.x, grad_outputs=torch.ones_like(self.y_train), create_graph=True)[0]','self.dxx = torch.autograd.grad(self.dx,self.x,grad_outputs=torch.ones_like(self.dx), create_graph=True)[0]','self.dxxx = torch.autograd.grad(self.dxx,self.x,grad_outputs=torch.ones_like(self.dxx), create_graph=True)[0]','self.diff_parts.append(self.dxxx)'),
            )
        for i in range(len(self.boundary_valuey)):
            if self.boundary_diff[i] == 0:
                fx = self.network(self.boundary_valuex[i])
                print("%.4f,%.4f--"%(float(fx.cpu().detach().numpy()),float(self.boundary_valuey[i].cpu().detach().numpy())),end =" ")
                self.MSEf.append(self.loss_fn(fx,self.boundary_valuey[i]))
            if self.boundary_diff[i] == 1:
                if int((self.boundary_valuex[i]-self.start)/(self.end-self.start)*self.size) == self.size:
                    fx_diff1 = self.dx[int((self.boundary_valuex[i]-self.start)/(self.end-self.start)*self.size)-1]
                else:
                    fx_diff1 = self.dx[int((self.boundary_valuex[i]-self.start)/(self.end-self.start)*self.size)]
                print("%.4f,%.4f--"%(float(fx_diff1.cpu().detach().numpy()),float(self.boundary_valuey[i].cpu().detach().numpy())),end =" ")
                self.MSEf.append(self.loss_fn(fx_diff1,self.boundary_valuey[i]))
            if self.boundary_diff[i] == 2:
                if int((self.boundary_valuex[i]-self.start)/(self.end-self.start)*self.size) == self.size:
                    fx_diff2 = self.dxx[int((self.boundary_valuex[i]-self.start)/(self.end-self.start)*self.size)-1]
                else:
                    fx_diff2 = self.dxx[int((self.boundary_valuex[i]-self.start)/(self.end-self.start)*self.size)]
                print("%.4f,%.4f--"%(float(fx_diff2.cpu().detach().numpy()),float(self.boundary_valuey[i].cpu().detach().numpy())),end =" ")
                self.MSEf.append(self.loss_fn(fx_diff2,self.boundary_valuey[i]))
        
        if(len(self.equ_coeffs) == 0):
            raise ValueError("未设置方程左侧系数")
       
        if(len(self.equ_coeffs) == 1):
            if type(self.equ_coeffs[i]) == str:
                exec("self.diff_left = %s * self.diff_parts[0]",self.equ_coeffs)
            else:
                self.diff_left = self.diff_coeff * self.equ_coeffs[0] * self.diff_parts[0]
        
        if(len(self.equ_coeffs) == 2):
            tempi = None
            for i in range(len(self.equ_coeffs)):
                if type(self.equ_coeffs[i]) == str:
                    tempi = i
                    break
            if tempi == None:
                self.diff_left = self.equ_coeffs[0] * self.diff_parts[0] + self.equ_coeffs[1] * self.diff_parts[1]
            else:
                if tempi == 0:
                    tempa = ''.join(["self.diff_left = self.diff_coeff * ",self.equ_coeffs[0]," * self.diff_parts[0] + self.equ_coeffs[1] * self.diff_parts[1]"])
                    self.switch(tempi,
                    (0,tempa),
                    )
                if tempi == 1:
                    tempb = ''.join(["self.diff_left = self.equ_coeffs[0] * self.diff_parts[0] + self.diff_coeff * ",self.equ_coeffs[1]," * self.diff_parts[1]"])
                    self.switch(tempi,
                    (1,tempb),
                    )
                
        if(len(self.equ_coeffs) == 3):
            tempi = None
            for i in range(len(self.equ_coeffs)):
                    if type(self.equ_coeffs[i]) == str:
                        tempi = i
                        break
            if tempi == None:
                self.diff_left = self.equ_coeffs[0] * self.diff_parts[0] + self.equ_coeffs[1] * self.diff_parts[1]
                if tempi == 0:
                    tempa = ''.join(["self.diff_left = self.diff_coeff * ",self.equ_coeffs[0]," * self.diff_parts[0] + self.equ_coeffs[1] * self.diff_parts[1] + self.equ_coeffs[2] * self.diff_parts[2]"])
                    self.switch(tempi,
                    (0,tempa),
                    )
                if tempi == 1:
                    tempb = ''.join(["self.diff_left = self.equ_coeffs[0] * self.diff_parts[0] + self.diff_coeff * ",self.equ_coeffs[1]," * self.diff_parts[1] + self.equ_coeffs[2] * self.diff_parts[2]"])
                    self.switch(tempi,
                    (1,tempb),
                    )
                if tempi == 2:
                    tempc = ''.join(["self.diff_left = self.equ_coeffs[0] * self.diff_parts[0] + self.equ_coeffs[1] * self.diff_parts[1] + self.diff_coeff * ",self.equ_coeffs[2]," * self.diff_parts[2]"])
                    self.switch(tempi,
                    (2,tempc),
                    )
                
        self.MSEu_script = ["self.MSEu = self.loss_fn(self.diff_left.float(),",self.equ_terms,")"]
        exec("".join(self.MSEu_script))
        print(float(self.MSEu.cpu().detach().numpy()),end =" ")
        self.switch(len(self.boundary_valuey),
        (0,'self.MSE = self.MSEu'),
        (1,'self.MSE = self.MSEu + self.MSEf[0]'),
        (2,'self.MSE = self.MSEu + self.MSEf[0] + self.MSEf[1]'),
        (3,'self.MSE = self.MSEu + self.MSEf[0] + self.MSEf[1] + self.MSEf[2]'),
        )
        self.MSEf = []
        self.diff_parts = []
        print("")

    def setSystemLeftPartCMD(self,ODE2:object,ODE3:object):
        self.cmd_diff_left = ""
        ODE2.cmd_diff_left = ""
        if ODE3 != None:
            ODE3.cmd_diff_left = ""
        for i in range(len(self.variables)):
            if self.variables[i] == 'x':
                self.cmd_diff_left += "+self.equ_coeffs[%d]*self.diffInSystems(self.equ_diffs[%d],self.x)"%(i,i)
            if self.variables[i] == 'y':
                self.cmd_diff_left += "+self.equ_coeffs[%d]*ODE2.diffInSystems(self.equ_diffs[%d],self.x)"%(i,i)
            if self.variables[i] == 'z':
                self.cmd_diff_left += "+self.equ_coeffs[%d]*ODE3.diffInSystems(self.equ_diffs[%d],self.x)"%(i,i)
        self.cmd_diff_left = "self.diff_left = " + self.cmd_diff_left
        for i in range(len(ODE2.variables)):
            if ODE2.variables[i] == 'x':
                ODE2.cmd_diff_left += "+ODE2.equ_coeffs[%d]*self.diffInSystems(ODE2.equ_diffs[%d],self.x)"%(i,i)
            if ODE2.variables[i] == 'y':
                ODE2.cmd_diff_left += "+ODE2.equ_coeffs[%d]*ODE2.diffInSystems(ODE2.equ_diffs[%d],self.x)"%(i,i)
            if ODE2.variables[i] == 'z':
                ODE2.cmd_diff_left += "+ODE2.equ_coeffs[%d]*ODE3.diffInSystems(ODE2.equ_diffs[%d],self.x)"%(i,i)
        ODE2.cmd_diff_left = "ODE2.diff_left = " + ODE2.cmd_diff_left
        if ODE3 != None:
            for i in range(len(ODE3.variables)):
                if ODE3.variables[i] == 'x':
                    ODE3.cmd_diff_left += "+ODE3.equ_coeffs[%d]*self.diffInSystems(ODE3.equ_diffs[%d],self.x)"%(i,i)
                if ODE3.variables[i] == 'y':
                    ODE3.cmd_diff_left += "+ODE3.equ_coeffs[%d]*ODE2.diffInSystems(ODE3.equ_diffs[%d],self.x)"%(i,i)
                if ODE3.variables[i] == 'z':
                    ODE3.cmd_diff_left += "+ODE3.equ_coeffs[%d]*ODE3.diffInSystems(ODE3.equ_diffs[%d],self.x)"%(i,i)
                    ODE3.cmd_diff_left = "ODE3.diff_left = " + ODE3.cmd_diff_left

    def setSystemLoss(self,ODE2:object,ODE3:object):
        exec(self.cmd_diff_left)
        exec(ODE2.cmd_diff_left)
        if ODE3 != None:
            exec(ODE3.cmd_diff_left)
        for i in range(len(self.boundary_diff)):
            if self.boundary_diff[i] == 0:
                fx = self.network(self.boundary_valuex[i])
                print("%.4f,%.4f--"%(float(fx.cpu().detach().numpy()),float(self.boundary_valuey[i].cpu().detach().numpy())),end =" ")
                self.MSEf.append(self.loss_fn(fx,self.boundary_valuey[i]))
            if self.boundary_diff[i] == 1:
                if int((self.boundary_valuex[i]-self.start)/(self.end-self.start)*self.size) == self.size:
                    fx_diff1 = self.dx[int((self.boundary_valuex[i]-self.start)/(self.end-self.start)*self.size)-1]
                else:
                    fx_diff1 = self.dx[int((self.boundary_valuex[i]-self.start)/(self.end-self.start)*self.size)]
                print("%.4f,%.4f--"%(float(fx_diff1.cpu().detach().numpy()),float(self.boundary_valuey[i].cpu().detach().numpy())),end =" ")
                self.MSEf.append(self.loss_fn(fx_diff1,self.boundary_valuey[i]))
            if self.boundary_diff[i] == 2:
                if int((self.boundary_valuex[i]-self.start)/(self.end-self.start)*self.size) == self.size:
                    fx_diff2 = self.dxx[int((self.boundary_valuex[i]-self.start)/(self.end-self.start)*self.size)-1]
                else:
                    fx_diff2 = self.dxx[int((self.boundary_valuex[i]-self.start)/(self.end-self.start)*self.size)]
                print("%.4f,%.4f--"%(float(fx_diff2.cpu().detach().numpy()),float(self.boundary_valuey[i].cpu().detach().numpy())),end =" ")
                self.MSEf.append(self.loss_fn(fx_diff2,self.boundary_valuey[i]))
        for i in range(len(ODE2.boundary_diff)):
            if ODE2.boundary_diff[i] == 0:
                fy = ODE2.network(ODE2.boundary_valuex[i])
                print("%.4f,%.4f--"%(float(fy.cpu().detach().numpy()),float(ODE2.boundary_valuey[i].cpu().detach().numpy())),end =" ")
                ODE2.MSEf.append(ODE2.loss_fn(fy,ODE2.boundary_valuey[i]))
            if ODE2.boundary_diff[i] == 1:
                if int((ODE2.boundary_valuex[i]-ODE2.start)/(ODE2.end-ODE2.start)*ODE2.size) == ODE2.size:
                    fx_diff1 = ODE2.dx[int((ODE2.boundary_valuex[i]-ODE2.start)/(ODE2.end-ODE2.start)*ODE2.size)-1]
                else:
                    fx_diff1 = ODE2.dx[int((ODE2.boundary_valuex[i]-ODE2.start)/(ODE2.end-ODE2.start)*ODE2.size)]
                print("%.4f,%.4f--"%(float(fx_diff1.cpu().detach().numpy()),float(ODE2.boundary_valuey[i].cpu().detach().numpy())),end =" ")
                ODE2.MSEf.append(ODE2.loss_fn(fx_diff1,ODE2.boundary_valuey[i]))
            if ODE2.boundary_diff[i] == 2:
                if int((ODE2.boundary_valuex[i]-ODE2.start)/(ODE2.end-ODE2.start)*ODE2.size) == ODE2.size:
                    fx_diff2 = ODE2.dxx[int((ODE2.boundary_valuex[i]-ODE2.start)/(ODE2.end-ODE2.start)*ODE2.size)-1]
                else:
                    fx_diff2 = ODE2.dxx[int((ODE2.boundary_valuex[i]-ODE2.start)/(ODE2.end-ODE2.start)*ODE2.size)]
                print("%.4f,%.4f--"%(float(fx_diff2.cpu().detach().numpy()),float(ODE2.boundary_valuey[i].cpu().detach().numpy())),end =" ")
                ODE2.MSEf.append(ODE2.loss_fn(fx_diff2,ODE2.boundary_valuey[i]))
        if ODE3 != None:
            for i in range(len(ODE3.boundary_diff)):
                if ODE3.boundary_diff[i] == 0:
                    fz = ODE3.network(ODE3.boundary_valuex[i])
                    print("%.4f,%.4f--"%(float(fz.cpu().detach().numpy()),float(ODE3.boundary_valuey[i].cpu().detach().numpy())),end =" ")
                    ODE3.MSEf.append(ODE3.loss_fn(fz,ODE3.boundary_valuey[i]))
                if ODE3.boundary_diff[i] == 1:
                    if int((ODE3.boundary_valuex[i]-ODE3.start)/(ODE3.end-ODE3.start)*ODE3.size) == ODE3.size:
                        fx_diff1 = ODE3.dx[int((ODE3.boundary_valuex[i]-ODE3.start)/(ODE3.end-ODE3.start)*ODE3.size)-1]
                    else:
                        fx_diff1 = ODE3.dx[int((ODE3.boundary_valuex[i]-ODE3.start)/(ODE3.end-ODE3.start)*ODE3.size)]
                    print("%.4f,%.4f--"%(float(fx_diff1.cpu().detach().numpy()),float(ODE3.boundary_valuey[i].cpu().detach().numpy())),end =" ")
                    ODE3.MSEf.append(ODE3.loss_fn(fx_diff1,ODE3.boundary_valuey[i]))
                if ODE3.boundary_diff[i] == 2:
                    if int((ODE3.boundary_valuex[i]-ODE3.start)/(ODE3.end-ODE3.start)*ODE3.size) == ODE3.size:
                        fx_diff2 = ODE3.dxx[int((ODE3.boundary_valuex[i]-ODE3.start)/(ODE3.end-ODE3.start)*ODE3.size)-1]
                    else:
                        fx_diff2 = ODE3.dxx[int((ODE3.boundary_valuex[i]-ODE3.start)/(ODE3.end-ODE3.start)*ODE3.size)]
                    print("%.4f,%.4f--"%(float(fx_diff2.cpu().detach().numpy()),float(ODE3.boundary_valuey[i].cpu().detach().numpy())),end =" ")
                    ODE3.MSEf.append(ODE3.loss_fn(fx_diff2,self.boundary_valuey[i]))
        exec(''.join(["self.MSEu = self.loss_fn(self.diff_left,",self.equ_terms,")"]))
        exec(''.join(["ODE2.MSEu = ODE2.loss_fn(ODE2.diff_left,",ODE2.equ_terms,")"]))
        if ODE3 != None:exec(''.join(["ODE3.MSEu = ODE3.loss_fn(ODE3.diff_left,",ODE3.equ_terms,")"]))
        if len(self.boundary_diff) == 0:self.MSE = self.MSEu
        if len(self.boundary_diff) == 1:self.MSE = self.MSEu + self.MSEf[0]
        if len(self.boundary_diff) == 2:self.MSE = self.MSEu + self.MSEf[0] + self.MSEf[1]
        if len(self.boundary_diff) == 3:self.MSE = self.MSEu + self.MSEf[0] + self.MSEf[1] + self.MSEf[2]

        if len(ODE2.boundary_diff) == 0:ODE2.MSE = ODE2.MSEu
        if len(ODE2.boundary_diff) == 1:ODE2.MSE = ODE2.MSEu + ODE2.MSEf[0]
        if len(ODE2.boundary_diff) == 2:ODE2.MSE = ODE2.MSEu + ODE2.MSEf[0] + ODE2.MSEf[1]
        if len(ODE2.boundary_diff) == 3:ODE2.MSE = ODE2.MSEu + ODE2.MSEf[0] + ODE2.MSEf[1] + ODE2.MSEf[2]
        
        if ODE3!=None:
            if len(ODE3.boundary_diff) == 0:ODE3.MSE = ODE3.MSEu
            if len(ODE3.boundary_diff) == 1:ODE3.MSE = ODE3.MSEu + ODE3.MSEf[0]
            if len(ODE3.boundary_diff) == 2:ODE3.MSE = ODE3.MSEu + ODE3.MSEf[0] + ODE3.MSEf[1]
            if len(ODE3.boundary_diff) == 3:ODE3.MSE = ODE3.MSEu + ODE3.MSEf[0] + ODE3.MSEf[1] + ODE3.MSEf[2]
        self.MSEf = []
        ODE2.MSEf = []
        if ODE3!=None:
            ODE3.MSEf = []
        print(float(self.MSE.cpu().detach().numpy()+ODE2.MSE.cpu().detach().numpy()),float(self.MSEu.cpu().detach().numpy()),float(ODE2.MSEu.cpu().detach().numpy()),end =" ")
        print("")

    def solve(self,network_layers:int,cells:int,learn_rate:float,weight_decay:float,max_steps:int,threads:int,reduction:str):
        start_time = perf_counter()
        self.loss_fn = torch.nn.MSELoss(reduction=reduction)
        torch.set_num_threads(threads) 
        self.boundary_diff = torch.tensor(self.boundary_diff).view(len(self.boundary_diff), 1).to(self.device)
        self.boundary_valuex = torch.tensor(self.boundary_valuex, requires_grad=True).view(len(self.boundary_valuex),1).to(self.device)
        self.boundary_valuey = torch.tensor(self.boundary_valuey, requires_grad=True).view(len(self.boundary_valuey),1).to(self.device)
        #print(self.boundary_valuey[0])
        dnn = DNN.Net(network_layers,cells,self.device)
        dnn = dnn.to(self.device)
        Adam = torch.optim.Adam(dnn.parameters(), lr = learn_rate,weight_decay = weight_decay)
        self.network = dnn
        self.optimizer = Adam
        plt.ion()
        for i in range(max_steps):
            print("Iter:%d"%i,end =" ")
            self.setLoss()
            loss_numpy = self.MSE.item()
            self.MSE.backward()
            if i % 200 == 0:
                DNN.updateGraph(self.x,self.y,loss_numpy,self.y_train,self.f,self.show_exact_flag)
                if loss_numpy < 0.005:
                    end_time = perf_counter()
                    print("用时：%fs"%(end_time-start_time))
                    DNN.finalGraph(self.x,self.y,loss_numpy,self.y_train,self.f,self.show_exact_flag)
                    exit()
            self.optimizer.step()
            self.optimizer.zero_grad()
        end_time = perf_counter()
        print("用时：%fs"%(end_time-start_time))
        DNN.finalGraph(self.x,self.y,loss_numpy,self.y_train,self.f,self.show_exact_flag)
        exit()
    
    def solve_with(self,ODE2:object,ODE3:object,network_layers:int,cells:int,learn_rate:float,weight_decay:float,max_steps:int,threads:int,reduction:str):
        start_time = perf_counter()
        self.loss_fn = torch.nn.MSELoss(reduction=reduction)
        ODE2.loss_fn = torch.nn.MSELoss(reduction=reduction)
        if ODE3 != None:ODE3.loss_fn = torch.nn.MSELoss(reduction=reduction)
        torch.set_num_threads(threads) 
        self.boundary_diff = torch.tensor(self.boundary_diff).view(len(self.boundary_diff), 1).to(self.device)
        self.boundary_valuex = torch.tensor(self.boundary_valuex, requires_grad=True).view(len(self.boundary_valuex),1).to(self.device)
        self.boundary_valuey = torch.tensor(self.boundary_valuey, requires_grad=True).view(len(self.boundary_valuey),1).to(self.device)
        ODE2.boundary_diff = torch.tensor(ODE2.boundary_diff).view(len(ODE2.boundary_diff), 1).to(ODE2.device)
        ODE2.boundary_valuex = torch.tensor(ODE2.boundary_valuex, requires_grad=True).view(len(ODE2.boundary_valuex),1).to(ODE2.device)
        ODE2.boundary_valuey = torch.tensor(ODE2.boundary_valuey, requires_grad=True).view(len(ODE2.boundary_valuey),1).to(ODE2.device)
        if ODE3!=None:
            ODE3.boundary_diff = torch.tensor(ODE3.boundary_diff).view(len(ODE3.boundary_diff), 1).to(ODE3.device)
            ODE3.boundary_valuex = torch.tensor(ODE3.boundary_valuex, requires_grad=True).view(len(ODE3.boundary_valuex),1).to(ODE3.device)
            ODE3.boundary_valuey = torch.tensor(ODE3.boundary_valuey, requires_grad=True).view(len(ODE3.boundary_valuey),1).to(ODE3.device)
        self.network = DNN.Net(network_layers,cells,self.device).to(self.device)
        ODE2.network = DNN.Net(network_layers,cells,self.device).to(self.device)
        if ODE3 != None:ODE3.network = DNN.Net(network_layers,cells,self.device).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr = learn_rate,weight_decay = weight_decay)
        ODE2.optimizer = torch.optim.Adam(ODE2.network.parameters(), lr = learn_rate,weight_decay = weight_decay)
        if ODE3 != None:ODE3.optimizer = torch.optim.Adam(ODE3.network.parameters(), lr = learn_rate,weight_decay = weight_decay)
        self.setSystemLeftPartCMD(ODE2,ODE3)
        plt.ion()
        for i in range(max_steps):
            print("Iter:%d"%i,end =" ")
            self.setSystemLoss(ODE2,ODE3)
            self.MSE.backward(retain_graph=True)
            ODE2.MSE.backward(retain_graph=True)
            if ODE3!=None:ODE3.MSE.backward(retain_graph=True)
            loss_numpy = self.MSE.item()
            if i % 200 == 0:
                if ODE3!= None:
                    DNN.updateGraphSystem(self.x,self.y,ODE2.y,ODE3.y,loss_numpy,self.y_train,ODE2.y_train,ODE3.y_train,self.f,self.show_exact_flag)
                else:
                    DNN.updateGraphSystem(self.x,self.y,ODE2.y,None,loss_numpy,self.y_train,ODE2.y_train,None,self.f,self.show_exact_flag)
                if loss_numpy < 0.005:
                    end_time = perf_counter()
                    print("用时：%fs"%(end_time-start_time))
                    if ODE3!= None:
                        DNN.finalGraphSystem(self.x,self.y,ODE2.y,ODE3.y,loss_numpy,self.y_train,ODE2.y_train,ODE3.y_train,self.f,self.show_exact_flag)
                    else:
                        DNN.finalGraphSystem(self.x,self.y,ODE2.y,None,loss_numpy,self.y_train,ODE2.y_train,None,self.f,self.show_exact_flag)
                    exit()
            self.optimizer.step()
            ODE2.optimizer.step()
            if ODE3!=None:ODE3.optimizer.step()
            self.optimizer.zero_grad()
            ODE2.optimizer.zero_grad()
            if ODE3!=None:ODE3.optimizer.zero_grad()
        end_time = perf_counter()
        print("用时：%fs"%(end_time-start_time))
        if ODE3!= None:
            DNN.finalGraphSystem(self.x,self.y,ODE2.y,ODE3.y,loss_numpy,self.y_train,ODE2.y_train,ODE3.y_train,self.f,self.show_exact_flag)
        else:
            DNN.finalGraphSystem(self.x,self.y,ODE2.y,None,loss_numpy,self.y_train,ODE2.y_train,None,self.f,self.show_exact_flag)
        exit()