import numpy as np

avg=0
sum=0
for i in np.arange(0.1, 0.6,0.1):
    print(i)
    sum = sum+i
    avg = (avg+i)/2
    
print(avg)
print(sum/5)

