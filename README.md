# ODEsolver-PINNS
solve ODE, ODE system and PDE with DNN in pytorch

## Requirements
**packages**: python==3.9, pytorch== CUDA1.16, numpy, matplotlib.pyplot
## Instruction for single ODE
### ===1.create an equation===
**Input**: xmin, xmax, samples
ODE1 = ODE(1,5,16000)<br>
### ===2.set the equation===
**Input**: coeffs, variables, left_part, right_part
example:
x*f''(x) - 3*f'(x)f'(x) + f(x)*f'(x) = cosx/sinx + x^2 - 3*e^x + f(x)*lnx + logx +5<br>
 ==> ODE1.setEquation([ODE1.xVar(),ODE1.diff(1,-3),ODE1.yVar()],[None],[2,1,1],"cos(x)/sin(x)+pow(x,2)- 3*exp(x)+y*log(x)+log10(x)+5")<br>
single constant in right part should use ODE.constant()<br>
if only 1 equation set the variables as [None]<br>
ODE1.setEquation([ODE1.diff(1,1),1],[None],[2,0],"log(x)-pow(x,-3)")<br>
### ===3.set exact solution===
**Input**: function,display function,<br>
ODE1.setExact(log(ODE1.xVar()),"lnx")<br>
**Input** None if not known<br>
### ===4.add boundaries===
**Input**: (max<=3), x(float), value(float)<br>
ODE1.addBoundary(0,1.0,0.0)<br>
ODE1.addBoundary(1,1.0,1.0)<br>
ODE1.addBoundary(2,0.0,0.5)<br>
### ===5.start solving===
**Input**: network_layers, cells, learn_rate, weight_decay, max_steps, threads, reduction<br>
ODE1.solve(7,40,1e-4,0.0001,20001,2,"sum")<br>

## Instruction for ODE system
### ===1.create equations===
**Input**: xmin, xmax, samples
ODE1 = ODE(1,5,16000)<br>
ODE2 = ODE(1,5,16000)<br>
ODE3 = ODE(1,5,16000)<br>
### ===2.set equations (at least 2)===
example:<br>
         dy/dt-dx/dt = x*(A-B*y)<br>
         dz/dt-dy/dt = -y*(C-D*x)<br>
         dz/dt*dx/dt = z*x-y*z+t)<br>
 ==> ODE1.setEquation([1,-1],[y,x],[1,1],"x*(A-B*y)")<br>
     ODE2.setEquation([1,-1],[y,x],[1,1],"-y*(C-D*x)")<br>
     ODE3.setEquation([ODE1.dx],[z],[1],"z*x-y*z+ODE1.xVar()")<br>
### ===3.set exact solution===
**Input**: function,display function,<br>
ODE1.setExact(log(ODE1.xVar()),"lnx")<br>
ODE2.setExact(None,None)<br>
**Input** None if not known<br>
### ===4.add boundaries===
**Input**: diff_level(max<=3), x(float), value(float)<br>
ODE1.addBoundary(0,2.0,6.86)<br>
ODE2.addBoundary(0,2.0,3.46)<br>
ODE3.addBoundary(0,2.0,5.21)<br>
### ===5.start solving===
**Input**: network_layers, cells, learn_rate, weight_decay, max_steps, threads, reduction<br>
ODE1.solve_with(ODE2,ODE3,8,60,1e-5,0.005,40001,2,"mean")<br>
if only 2 equations set ODE3 as None<br>
## Instruction for PDE
### ===CODING===
