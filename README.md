# ODEsolver-PINNS
solve ODE, ODE system and PDE with DNN in pytorch

Instruction
===create an equation===
#input: xmin, xmax, samples
ODE1 = ODE(1,5,16000)

#===set the equation===
#input: coeffs, variables, left_part, right_part
#example1:x*f''(x) - 3*f'(x)f'(x) + f(x)*f'(x) = cosx/sinx + x^2 - 3*e^x + f(x)*lnx + logx +5
# ==> ODE1.setEquation([ODE1.xVar(),ODE1.diff(1,-3),ODE1.yVar()],[None],[2,1,1],"cos(x)/sin(x)+pow(x,2)- 3*exp(x)+y*log(x)+log10(x)+5")
#example2:dy/dt-dx/dt = x*(A-B*y)
#         dz/dt-dy/dt = -y*(C-D*x)
#         dz/dt*dx/dt = z*x-y*z+t)
# ==> ODE1.setEquation()
# single constant in right part should use ODE.constant()
# if only 1 equation set the variable as [None]
ODE1.setEquation([ODE1.diff(1,1),1],[None],[2,0],"log(x)-pow(x,-3)")

#===set exact solution===
#input: None if not known
#ODE1.setExact(-pow(ODE1.xVar(),3)/6,"-x^3/6")
ODE1.setExact(log(ODE1.xVar()),"lnx")

#===add boundaries===
#input: diff_level, x, f(x)
ODE1.addBoundary(0,1.0,0.0)           
ODE1.addBoundary(1,1.0,1.0)
# ODE1.addBoundary(2,0.0,0.5)

#===start solving===
#input: network_layers, cells, learn_rate, weight_decay, max_steps, threads, reduction
ODE1.solve(7,40,1e-4,0.0001,20001,2,"sum")
