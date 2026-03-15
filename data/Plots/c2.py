import sys
import matplotlib.pyplot as plt
import matplotlib.font_manager
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = "12"
plt.rcParams["figure.figsize"] = (8,4)
import numpy


fig = plt.figure()

xLabel='Temperature (°C)'
yLabel='Height (m)'

plt.subplot(1, 2, 1)
RB=numpy.genfromtxt("previousDesign/data.csv",delimiter=',')
RB1=numpy.genfromtxt("previousDesign/exhaust_temp.csv",delimiter=',')
plt.scatter(RB1[:,3],RB1[:,4],label='Previous design (Experimental)', color='k', marker='v', linewidth=2.2)
plt.plot(RB[:,0],RB[:,1],label='Previous design (Numerical)', color='r', linewidth=3.0)

plt.tick_params(axis='both', which='both', direction='in')
plt.tick_params(bottom='True', top='True', left='True', right='True', which='both')
plt.minorticks_on()

plt.axis([20.0, 36.0 ,0.0, 2.0])

plt.xlabel(xLabel)
plt.ylabel(yLabel)

plt.legend(loc="lower right",fontsize=10 ,markerscale=1.0)



plt.subplot(1, 2, 2)
RB=numpy.genfromtxt("retrofittedDesign/data.csv",delimiter=',')
RB1=numpy.genfromtxt("retrofittedDesign/exhaust_temp.csv",delimiter=',')
plt.scatter(RB1[:,2],RB1[:,4],label='Retrofitted design (Experimental)', color='k', marker='s', linewidth=2.2)
plt.plot(RB[:,0],RB[:,1],label='Retrofitted design (Numerical)', color='r', linewidth=3.0)

plt.tick_params(axis='both', which='both', direction='in')
plt.tick_params(bottom='True', top='True', left='True', right='True', which='both')
plt.minorticks_on()

plt.axis([20.0, 36.0 ,0.0, 2.0])

plt.xlabel(xLabel)
plt.ylabel(yLabel)

plt.legend(loc="lower right",fontsize=10 ,markerscale=1.0)


fig.tight_layout()
fig.subplots_adjust(hspace=.35)

plt.savefig('Temp1.png', dpi=1000)
