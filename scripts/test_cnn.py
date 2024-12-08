import torch
from torchvision import datasets, transforms
from train_cnn import DigitRecognizer
import matplotlib.pyplot as plt

def test_cnn_model(model_path='multi_digit_ocr/models/digit_cnn_epoch_25.pth'):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the model
    model = DigitRecognizer().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Load MNIST test dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST('multi_digit_ocr/data', train=False, 
                                download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)
    
    # Test the model
    correct = 0
    total = 0
    
    print("\nTesting CNN model on MNIST test set...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Visualize first batch of predictions
            if total <= 1000:
                fig, axes = plt.subplots(2, 5, figsize=(12, 6))
                for idx in range(10):
                    i, j = idx // 5, idx % 5
                    axes[i, j].imshow(images[idx].cpu().squeeze(), cmap='gray')
                    axes[i, j].set_title(f'Pred: {predicted[idx]}\nTrue: {labels[idx]}')
                    axes[i, j].axis('off')
                plt.tight_layout()
                plt.savefig('multi_digit_ocr/data/cnn_test_results.png')
                plt.close()
    
    accuracy = 100 * correct / total
    print(f'Accuracy on test set: {accuracy:.2f}%')
    print(f'Test results visualization saved to cnn_test_results.png')

if __name__ == '__main__':
    test_cnn_model() 