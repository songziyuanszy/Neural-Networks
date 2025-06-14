import numpy as np
import matplotlib.pyplot as plt

def load_data(data_path):
    data = np.loadtxt(data_path, delimiter=',')
    X = data[:, :-1]            # 前面的数据
    y = data[:, -1].astype(int) # 最后的分类
    return X, y

def preprocess_data(X, y):
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    X, y = X[indices], y[indices]
    
    train_size = int(0.8 * len(X)) 
    val_size = int(0.1 * len(X)) 
    
    X_train, y_train = X[:train_size], y[:train_size] # 80%训练集
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size] # 10%验证集
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:] # 10%测试集
    
    X_min = X_train.min(axis=0)
    X_max = X_train.max(axis=0)
    X_train = (X_train - X_min) / (X_max - X_min + 1e-8)
    X_val = (X_val - X_min) / (X_max - X_min + 1e-8)
    X_test = (X_test - X_min) / (X_max - X_min + 1e-8)
    
    num_classes = len(np.unique(y)) # 分类数
    y_train = np.eye(num_classes)[y_train]
    y_val = np.eye(num_classes)[y_val]
    y_test = np.eye(num_classes)[y_test]
    
    return X_train, y_train, X_val, y_val, X_test, y_test, num_classes

class NeuralNetwork:
    def __init__(self, input_dim, hidden_dims, output_dim, lambda_reg=0.001):
        self.lambda_reg = lambda_reg
        self.params = {}
        dims = [input_dim] + hidden_dims + [output_dim] # [40, 128, 64, 3]
        
        for i in range(1, len(dims)):
            # 标准差为 sqrt(2./前一层维度)，帮助缓解梯度消失/爆炸问题
            self.params[f'W{i}'] = np.random.randn(dims[i-1], dims[i]) * np.sqrt(2. / dims[i-1])
            self.params[f'b{i}'] = np.zeros(dims[i]) # 偏置初始化为全零向量
        self.num_layers = len(dims) - 1

    def forward(self, X):
        self.cache = {'A0': X} # 初始化缓存字典
        for i in range(1, self.num_layers):
            W = self.params[f'W{i}'] # 当前层权重矩阵
            b = self.params[f'b{i}'] # 当前层偏置向量
            Z = np.dot(self.cache[f'A{i-1}'], W) + b
            A = np.maximum(0, Z)
            self.cache[f'Z{i}'] = Z # 线性变换结果
            self.cache[f'A{i}'] = A # 激活后结果
        
        W = self.params[f'W{self.num_layers}']
        b = self.params[f'b{self.num_layers}']

        # Softmax激活
        Z = np.dot(self.cache[f'A{self.num_layers-1}'], W) + b
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        A = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
        self.cache[f'Z{self.num_layers}'] = Z 
        self.cache[f'A{self.num_layers}'] = A

        return A

    def compute_loss(self, A, y_true):
        m = y_true.shape[0]
        correct_log_probs = -np.log(np.sum(A * y_true, axis=1))
        loss = np.sum(correct_log_probs) / m
        reg_loss = sum(np.sum(W**2) for W in [self.params[f'W{i}'] for i in range(1, self.num_layers+1)])
        return loss + (self.lambda_reg / (2 * m)) * reg_loss

    def backward(self, X, y_true):
        m = X.shape[0]
        grads = {}
        dZ = self.cache[f'A{self.num_layers}'] - y_true
        
        for i in reversed(range(1, self.num_layers+1)):
            A_prev = self.cache[f'A{i-1}']
            grads[f'dW{i}'] = (A_prev.T @ dZ) / m + (self.lambda_reg / m) * self.params[f'W{i}']
            grads[f'db{i}'] = np.sum(dZ, axis=0) / m
            
            if i > 1:
                dA_prev = dZ @ self.params[f'W{i}'].T
                dZ = dA_prev * (A_prev > 0)
        
        return grads

    def update_params(self, grads, learning_rate):
        for key in self.params:
            self.params[key] -= learning_rate * grads[f'd{key}']

def train_model(X_train, y_train, X_val, y_val, input_dim, num_classes, 
               hidden_dims=[128, 64], learning_rate=0.01, 
               batch_size=64, num_epochs=1000, patience=20):
    
    model = NeuralNetwork(input_dim, hidden_dims, num_classes)
    best_val_loss = float('inf') # 存储最佳损失
    best_params = None
    counter = 0
    

    for epoch in range(num_epochs):
        permutation = np.random.permutation(X_train.shape[0])
        X_shuffled = X_train[permutation] # 打乱后的特征数据
        y_shuffled = y_train[permutation] # 对应打乱后的标签数据
        
        for i in range(0, X_train.shape[0], batch_size): ## 从0开始，每次步进batch_size大小，直到遍历
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            
            A = model.forward(X_batch) # 前向传播
            loss = model.compute_loss(A, y_batch) # 损失
            grads = model.backward(X_batch, y_batch) # 反向传播
            model.update_params(grads, learning_rate)

        train_loss_visual[epoch] = loss
        
        # 验证集评估
        val_probs = model.forward(X_val)
        val_loss = model.compute_loss(val_probs, y_val)
        val_preds = np.argmax(val_probs, axis=1) # 将概率输出转换为类别预测
        val_acc = np.mean(val_preds == np.argmax(y_val, axis=1))
        
        val_loss_visual[epoch] = val_loss

        print(f"Epoch {epoch+1:3d} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = {k: v.copy() for k, v in model.params.items()}
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
                
    model.params = best_params
    return model

# 数据预处理
# X, y = load_data('D:\\homework1\\waveform.data')
X, y = load_data('D:\\homework1\\waveform-+noise.data')

X_train, y_train, X_val, y_val, X_test, y_test, num_classes = preprocess_data(X, y)

num_epochs=5000
train_loss_visual = np.zeros(num_epochs)
val_loss_visual = np.zeros(num_epochs)

# 训练模型
model = train_model(
    X_train, y_train, 
    X_val, y_val,
    input_dim=X_train.shape[1],
    num_classes=num_classes,
    hidden_dims=[128, 64],
    learning_rate=0.001,
    batch_size=128,
    num_epochs=num_epochs,
    patience=100
)

# 测试评估
test_probs = model.forward(X_test)
test_preds = np.argmax(test_probs, axis=1)
test_accuracy = np.mean(test_preds == np.argmax(y_test, axis=1))
print(f"\nFinal Test Accuracy: {test_accuracy:.4f}")


plt.figure(figsize=(10, 6))
plt.plot(train_loss_visual[:], label='Train Loss', color='blue')
plt.plot(val_loss_visual[:], label='Valid Loss', color='orange', linestyle='--')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('D:\\homework1\\loss_curve.png', dpi=300, bbox_inches='tight')
plt.show()