# Importar dependencias
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

# Definir Hiperparámetros
input_size = 784
hidden_size = 512  # Aumento del tamaño de la capa oculta
num_classes = 10
num_epochs = 20  # Aumento del número de épocas
batch_size = 128  # Ajuste del tamaño del lote
lr = 5e-4  # Reducción de la tasa de aprendizaje

# Aplicar Data Augmentation al conjunto de entrenamiento
train_data = dsets.FashionMNIST(
    root='./data',
    train=True,
    transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),  # Aumento de datos
        transforms.RandomRotation(10),      # Rotación aleatoria
        transforms.ToTensor()
    ]),
    download=True
)

test_data = dsets.FashionMNIST(
    root='./data',
    train=False,
    transform=transforms.ToTensor()
)

# Leyendo la data
train_gen = torch.utils.data.DataLoader(dataset=train_data,
                                        batch_size=batch_size,
                                        shuffle=True)

test_gen = torch.utils.data.DataLoader(dataset=test_data,
                                       batch_size=batch_size,
                                       shuffle=False)

# Definir modelo
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.1)  # Reducción del dropout
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.drop(out)
        out = self.fc2(out)
        return out

# Instancia del modelo
net = Net(input_size, hidden_size, num_classes)

# Verificar si CUDA está disponible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net.to(device)

# Compilación
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

# Entrenamiento
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_gen):
        images = images.view(-1, 28*28).to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f'Epoca [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_data) // batch_size}], Loss: {loss.item():.4f}')

# Evaluación en el conjunto de prueba
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_gen:
        images = images.view(-1, 28*28).to(device)
        labels = labels.to(device)

        output = net(images)
        _, predicted = torch.max(output, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

# Imprimir la precisión
print(f'Accuracy: {100 * correct / total:.3f} %')
