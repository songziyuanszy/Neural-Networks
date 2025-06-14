import numpy as np
import matplotlib.pyplot as plt
from time import time

# 加载数据
train_data = np.loadtxt('D:/homework2/handwritten_dataset/optdigits.tra', delimiter=',')
test_data = np.loadtxt('D:/homework2/handwritten_dataset/optdigits.tes', delimiter=',')

# 预处理
X_train, y_train = train_data[:, :-1], train_data[:, -1].astype(int)
X_test, y_test = test_data[:, :-1], test_data[:, -1].astype(int)

# 归一化 (样本数, 通道, 高, 宽 )
X_train = X_train.reshape(-1, 1, 8, 8) / 16.0
X_test = X_test.reshape(-1, 1, 8, 8) / 16.0

# one-hot编码
def one_hot(y, num_classes=10):
    n = y.shape[0]
    y_one_hot = np.zeros((n, num_classes))
    y_one_hot[np.arange(n), y] = 1
    return y_one_hot

y_train_onehot = one_hot(y_train)
y_test_onehot = one_hot(y_test)

# 卷积层
class ConvLayer:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # He初始化
        std_dev = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * std_dev
        self.bias = np.zeros(out_channels)
        
    def forward(self, x):
        self.input = x
        n, c, h, w = x.shape
        
        # 输出尺寸
        out_h = (h + 2*self.padding - self.kernel_size) // self.stride + 1
        out_w = (w + 2*self.padding - self.kernel_size) // self.stride + 1        
        
        # 加padding
        if self.padding > 0:
            x_padded = np.pad(x, ((0,0), (0,0), (self.padding, self.padding), (self.padding, self.padding)))
        else:
            x_padded = x            

        output = np.zeros((n, self.out_channels, out_h, out_w))
        
        # 向量化卷积
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * self.stride
                h_end = h_start + self.kernel_size
                w_start = j * self.stride
                w_end = w_start + self.kernel_size                
                # 提取所有样本和通道的切片
                x_slice = x_padded[:, :, h_start:h_end, w_start:w_end]
                
                # 向量化计算 (n, out_channels)
                output[:, :, i, j] = np.tensordot(
                    x_slice, self.weights, axes=([1, 2, 3], [1, 2, 3])
                ) + self.bias                
        return output
    
    def backward(self, dout):
        n, c, h, w = self.input.shape
        out_h = dout.shape[2]
        out_w = dout.shape[3]
        
        # 添加padding
        if self.padding > 0:
            x_padded = np.pad(self.input, ((0,0), (0,0), (self.padding, self.padding), (self.padding, self.padding)))
            dx_padded = np.zeros_like(x_padded)
        else:
            x_padded = self.input
            dx_padded = np.zeros_like(x_padded)
            
        dweights = np.zeros_like(self.weights)
        dbias = np.zeros_like(self.bias)
        
        # 向量化计算偏置梯度
        dbias = np.sum(dout, axis=(0, 2, 3))
        
        # 向量化计算权重梯度和输入梯度
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * self.stride
                h_end = h_start + self.kernel_size
                w_start = j * self.stride
                w_end = w_start + self.kernel_size
                
                # 提取所有样本和通道的切片
                x_slice = x_padded[:, :, h_start:h_end, w_start:w_end]
                
                # 向量化计算权重梯度
                dweights += np.tensordot(
                    dout[:, :, i, j], x_slice, axes=([0], [0]))
                
                # 向量化计算输入梯度
                dx_slice = np.tensordot(
                    dout[:, :, i, j], self.weights, axes=([1], [0]))
                dx_padded[:, :, h_start:h_end, w_start:w_end] += dx_slice
        
        # 保存梯度
        self.dweights = dweights
        self.dbias = dbias
        
        # 移除padding
        if self.padding > 0:
            dx = dx_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            dx = dx_padded
            
        return dx

# 平均池化
class AvgPoolLayer:
    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride
        self.pool_area = kernel_size * kernel_size
        
    def forward(self, x):
        self.input = x
        n, c, h, w = x.shape
        out_h = (h - self.kernel_size) // self.stride + 1
        out_w = (w - self.kernel_size) // self.stride + 1

        output = np.zeros((n, c, out_h, out_w))
        
        # 平均池化
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * self.stride
                h_end = h_start + self.kernel_size
                w_start = j * self.stride
                w_end = w_start + self.kernel_size
                
                # 提取所有样本和通道的切片
                x_slice = x[:, :, h_start:h_end, w_start:w_end]
                
                # 向量化计算平均值
                output[:, :, i, j] = np.mean(x_slice, axis=(2, 3))
                
        return output
    
    def backward(self, dout):
        x = self.input
        n, c, h, w = x.shape
        out_h = dout.shape[2]
        out_w = dout.shape[3]       
        dx = np.zeros_like(x)        
        # 向量化梯度计算
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * self.stride
                h_end = h_start + self.kernel_size
                w_start = j * self.stride
                w_end = w_start + self.kernel_size                
                # 梯度平均分配
                grad = dout[:, :, i, j] / self.pool_area                
                # 创建与池化区域相同形状的梯度数组
                grad_expanded = np.repeat(
                    np.repeat(
                        grad[:, :, np.newaxis, np.newaxis], 
                        self.kernel_size, axis=2
                    ), 
                    self.kernel_size, axis=3
                )                
                dx[:, :, h_start:h_end, w_start:w_end] += grad_expanded                
        return dx

# 激活函数
class ReLU:
    def forward(self, x):
        self.mask = (x > 0)
        return x * self.mask
    
    def backward(self, dout):
        return dout * self.mask

class LeakyReLU:
    def __init__(self, alpha=0.1):
        self.alpha = alpha

    def forward(self, x):
        self.x = x
        return np.where(x > 0, x, self.alpha * x)

    def backward(self, dout):
        dx = np.ones_like(self.x)
        dx[self.x <= 0] = self.alpha
        return dout * dx

# 全连接层
class FCLayer:
    def __init__(self, input_size, output_size):
        # He初始化
        std_dev = np.sqrt(2.0 / input_size)
        self.weights = np.random.randn(input_size, output_size) * std_dev
        self.bias = np.zeros(output_size)
        
    def forward(self, x):
        self.input = x
        return np.dot(x, self.weights) + self.bias
    
    def backward(self, dout):
        dx = np.dot(dout, self.weights.T)
        self.dweights = np.dot(self.input.T, dout)
        self.dbias = np.sum(dout, axis=0)
        return dx

# Softmax交叉熵损失
class SoftmaxCrossEntropyLoss:
    def forward(self, x, y):
        # 数值稳定性的softmax
        max_vals = np.max(x, axis=1, keepdims=True)
        exp_vals = np.exp(x - max_vals)
        self.probs = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
        
        # 交叉熵损失
        n = y.shape[0]
        log_probs = -np.log(self.probs[np.arange(n), np.argmax(y, axis=1)] + 1e-8)
        loss = np.sum(log_probs) / n
        return loss
    
    def backward(self, y):
        n = y.shape[0]
        dx = self.probs.copy()
        dx[np.arange(n), np.argmax(y, axis=1)] -= 1
        return dx / n

# 展平层
class Flatten:
    def forward(self, x):
        self.input_shape = x.shape
        return x.reshape(x.shape[0], -1)
    
    def backward(self, dout):
        return dout.reshape(self.input_shape)

# LeNet-5网络（8x8输入）
class LeNet5:
    def __init__(self):
        self.layers = [                                 # 加了padding不见得好，反而计算慢，设为0
            ConvLayer(1, 32, kernel_size=3, padding=0),  # 输入:1x8x8 -> 输出:16x6x6
            LeakyReLU(),
            AvgPoolLayer(kernel_size=2, stride=2),       # 输入:16x6x6 -> 输出:16x3x3
            ConvLayer(32, 64, kernel_size=3, padding=0),  # 输入:32x3x3 -> 输出:32x1x1
            LeakyReLU(),
            Flatten(),                                   # 输入:16x1x1 -> 输出:16
            FCLayer(64, 128), 
            LeakyReLU(),
            FCLayer(128, 64),
            LeakyReLU(),
            FCLayer(64, 10)
        ]
        self.loss = SoftmaxCrossEntropyLoss()
        
    def forward(self, x, y=None):
        output = x

        for i, layer in enumerate(self.layers):
            output = layer.forward(output)
        
        if y is not None:
            loss = self.loss.forward(output, y)
            return output, loss
        return output, None
    
    def backward(self, y):
        dout = self.loss.backward(y)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
    
    def update_params(self, lr):
        for layer in self.layers:
            if isinstance(layer, (ConvLayer, FCLayer)):
                layer.weights -= lr * layer.dweights
                layer.bias -= lr * layer.dbias
                
    def predict(self, x):
        output, _ = self.forward(x)
        return np.argmax(output, axis=1)

# ============= 训练参数 =============
epochs = 2000
batch_size = 32
learning_rate = 0.02
n_train = X_train.shape[0]
n_batches = n_train // batch_size

# 创建网络
print("创建网络...")
net = LeNet5()

# 记录训练过程
train_losses = []
test_accuracies = []
epoch_times = []

print("开始训练LeNet-5...")
start_time = time()

for epoch in range(epochs):
    epoch_start = time()
    epoch_loss = 0.0
    
    # 打乱数据
    indices = np.arange(n_train)
    np.random.shuffle(indices)
    X_train_shuffled = X_train[indices]    
    y_train_shuffled = y_train_onehot[indices]

    # 输入加噪声
    X_train_shuffled += np.random.normal(0, 0.2, X_train_shuffled.shape)
    X_train_shuffled = np.clip(X_train_shuffled, 0., 1.)

    for i in range(n_batches):
        # 获取小批量数据
        start = i * batch_size
        end = start + batch_size
        X_batch = X_train_shuffled[start:end]
        y_batch = y_train_shuffled[start:end]
        
        # 前向传播
        _, loss = net.forward(X_batch, y_batch)
        epoch_loss += loss
        
        # 反向传播
        net.backward(y_batch)
        
        # 更新参数
        # net.update_params(learning_rate)
        net.update_params(learning_rate * (1 - epoch / epochs ))
    
    # 计算平均训练损失
    avg_loss = epoch_loss / n_batches
    train_losses.append(avg_loss)
    
    # 在测试集上评估
    predictions = net.predict(X_test)
    accuracy = np.mean(predictions == y_test)
    test_accuracies.append(accuracy)
    
    epoch_time = time() - epoch_start
    epoch_times.append(epoch_time)
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Test Acc: {accuracy:.4f}, Time: {epoch_time:.2f}s")

total_time = time() - start_time
print(f"训练完成! 总时间: {total_time:.2f}秒, 平均每轮: {np.mean(epoch_times):.2f}秒")


plt.rcParams['font.sans-serif'] = ['SimHei']    # 或 ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False      # 解决负号显示为方块的问题

# 1. 训练损失
plt.figure(figsize=(7, 5))
plt.plot(train_losses, color='royalblue', linewidth=2, marker='o', markersize=6, label='训练损失')
plt.xlabel('轮次', fontsize=14)
plt.ylabel('损失值', fontsize=14)
plt.title('训练损失变化曲线', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(r'D:\homework2\pictures\train_loss.png', dpi=300)
plt.close()

# 2. 测试准确率
plt.figure(figsize=(7, 5))
plt.plot(test_accuracies, color='orange', linewidth=2, marker='s', markersize=6, label='测试准确率')
plt.xlabel('轮次', fontsize=14)
plt.ylabel('准确率', fontsize=14)
plt.title('测试集准确率变化曲线', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.ylim(0.96, 0.99)
plt.savefig(r'D:\homework2\pictures\test_accuracy.png', dpi=300)
plt.close()

# 3. 每轮耗时
plt.figure(figsize=(7, 5))
plt.plot(epoch_times, color='seagreen', linewidth=2, marker='^', markersize=6, label='每轮耗时')
plt.xlabel('轮次', fontsize=14)
plt.ylabel('时间（秒）', fontsize=14)
plt.title('每轮训练耗时曲线', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(r'D:\homework2\pictures\epoch_time.png', dpi=300)
plt.close()


# 最终测试精度
final_predictions = net.predict(X_test)
final_accuracy = np.mean(final_predictions == y_test)
print(f"最终测试精度: {final_accuracy:.4f}")