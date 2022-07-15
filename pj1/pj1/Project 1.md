# Project 1

赵心怡 19307110452



##### Task1： Change the network structure: the vector nHidden specifies the number of hidden units in each layer.

nHidden 表示了每个layer的hidden unit。 ‘nHidden = [n]’表示了单层的神经网络， [n1,n2,n3]代表了三层的神经网络，以此类推。

nhidden的长度代表了神经网络的深度，因此我们可以通过修改这个参数来改变网络的结构。

n-hidden=[10]:          test-error=0.513000

<img src="C:\Users\13374\AppData\Roaming\Typora\typora-user-images\image-20220321140045185.png" alt="image-20220321140045185" style="zoom:50%;" />

n-hidden=[30]:       test error = 0.313000

<img src="C:\Users\13374\AppData\Roaming\Typora\typora-user-images\image-20220321140229083.png" alt="image-20220321140229083" style="zoom:50%;" />

n-hidden=[10,10]         test error: 0.59300

<img src="C:\Users\13374\AppData\Roaming\Typora\typora-user-images\image-20220321140750889.png" alt="image-20220321140750889" style="zoom:50%;" />

n-hidden = [10,30,30]           test error=0.48900

<img src="C:\Users\13374\AppData\Roaming\Typora\typora-user-images\image-20220321141005408.png" alt="image-20220321141005408" style="zoom:50%;" />

从结果上可以看出n-hidden 单层的神经元增加会减少test error，使模型更加准确，层数的增加似乎不能让test error降低，同时它的运行速度减慢，因此当单层的神经元为30的时候模型表现最好。





#####  **Task2:** Change the training procedure by modifying the sequence of step-sizes or using different step-sizes for different variables.

step-size的大小可以影响决定什么时候模型函数能达到局部收敛已经什么时候停止。

step-size = 1e-4, test-error=0.542000

<img src="C:\Users\13374\AppData\Roaming\Typora\typora-user-images\image-20220324193233957.png" alt="image-20220324193233957" style="zoom:50%;" />

step-size=1e-3,  test-error = 0.514000

<img src="C:\Users\13374\AppData\Roaming\Typora\typora-user-images\image-20220324192751857.png" alt="image-20220324192751857" style="zoom:50%;" />

step-size =1e-2, test-error = 0.397000

<img src="C:\Users\13374\AppData\Roaming\Typora\typora-user-images\image-20220324192901544.png" alt="image-20220324192901544" style="zoom:50%;" />

step-size=1e-1, test-error = 0.943000

<img src="C:\Users\13374\AppData\Roaming\Typora\typora-user-images\image-20220324193144426.png" alt="image-20220324193144426" style="zoom:50%;" />

从图中可以发现当stepsize设置过大会导致走的过快而不收敛，而步长过小会导致收敛速度很慢，由上结果可以看出最优的梯度是1e-2





