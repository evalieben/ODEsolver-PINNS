# coding=utf-8
from PINNS import ODE
from PINNS import cos,sin,tan,cosh,sinh,tanh,pow,exp,log,log10
# def main():
    # #===create an equation===
    # #input: xmin, xmax, samples
    # ODE1 = ODE(1,5,16000)
    
    # #===set the equation===
    # #input: coeffs, variables, left_part, right_part
    # #example1:x*f''(x) - 3*f'(x)f'(x) + f(x)*f'(x) = cosx/sinx + x^2 - 3*e^x + f(x)*lnx + logx +5
    # # ==> ODE1.setEquation([ODE1.xVar(),ODE1.diff(1,-3),ODE1.yVar()],[None],[2,1,1],"cos(x)/sin(x)+pow(x,2)- 3*exp(x)+y*log(x)+log10(x)+5")
    # #example2:dy/dt-dx/dt = x*(A-B*y)
    # #         dz/dt-dy/dt = -y*(C-D*x)
    # #         dz/dt*dx/dt = z*x-y*z+t)
    # # ==> ODE1.setEquation()
    # # single constant in right part should use ODE.constant()
    # # if only 1 equation set the variable as [None]
    # ODE1.setEquation([ODE1.diff(1,1),1],[None],[2,0],"log(x)-pow(x,-3)")
    
    # #===set exact solution===
    # #input: None if not known
    # #ODE1.setExact(-pow(ODE1.xVar(),3)/6,"-x^3/6")
    # ODE1.setExact(log(ODE1.xVar()),"lnx")
    
    # #===add boundaries===
    # #input: diff_level, x, f(x)
    # ODE1.addBoundary(0,1.0,0.0)           
    # ODE1.addBoundary(1,1.0,1.0)
    # # ODE1.addBoundary(2,0.0,0.5)

    # #===start solving===
    # #input: network_layers, cells, learn_rate, weight_decay, max_steps, threads, reduction
    # ODE1.solve(7,40,1e-4,0.0001,20001,2,"sum")
    
  
def main():        
    ODE1 = ODE(2,5,9000)
    ODE2 = ODE(2,5,9000)
    #ODE3 = ODE(0,5,16000)
    ODE1.setEquation([1],['x'],[1],"-2*x+2*y")
    ODE2.setEquation([1],['y'],[1],"2*x-5*y")
    ODE1.addBoundary(0,2.0,6.86)
    ODE2.addBoundary(0,2.0,3.46)
    ODE1.setExact(50*exp(-ODE1.xVar())-10*exp(-6*ODE1.xVar()),"50*exp(-t)-10*exp(-6t)")
    ODE2.setExact(25*exp(-ODE1.xVar())+20*exp(-6*ODE1.xVar()),"25*exp(-t)+20*exp(-6t)")
    #ODE3.setEquation([ODE3.xVar(),7],['x'],[2],"ODE.constant(0)")
    ODE1.solve_with(ODE2,None,8,60,1e-5,0.005,40001,2,"mean")

if __name__ == '__main__':
  main()
