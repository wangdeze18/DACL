import numpy as np

def root_2(x, t, c0):
	return ((x* ((1-(c0**2.0))/t)) + (c0**2.0))**(1./2)

c0=0.33

t = 39120
now_step = 3912
for i in range(20):

	i_step = root_2(now_step,t,c0)
	print(i, now_step, i_step)
	now_step += i_step * t / 10