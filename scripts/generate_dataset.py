import cv2
import numpy as np
from random import randint, choice
from torchvision import datasets, transforms
import os

def load_mnist():
    # Load MNIST dataset
    transform = transforms.ToTensor()
    mnist_data = datasets.MNIST('multi_digit_ocr/data', train=True, download=True, transform=transform)
    
    # Extract individual images and labels
    digit_images = mnist_data.data.numpy()  # Shape: (N, 28, 28), pixel values 0-255
    digit_labels = mnist_data.targets.numpy()
    
    print(f"Loaded MNIST dataset: {len(digit_images)} images")
    print(f"Image shape: {digit_images[0].shape}")
    print(f"Pixel value range: [{digit_images[0].min()}, {digit_images[0].max()}]")
    
    return digit_images, digit_labels

def create_blank_canvas(height=32, width=128):
    return np.ones((height, width), dtype=np.uint8) * 255

def add_digits_to_canvas(digit_images, digit_labels, num_digits):
    canvas = create_blank_canvas()
    x_offset = 5
    sequence = []
    
    for _ in range(num_digits):
        # Select random digit
        idx = randint(0, len(digit_images) - 1)
        digit = digit_images[idx]
        label = digit_labels[idx]
        
        # Ensure digit is in correct range (0-255)
        if digit.max() <= 1.0:
            digit = digit * 255
        
        # Convert to uint8
        digit = digit.astype(np.uint8)
        
        # Calculate vertical position to center digit
        y_offset = (32 - 28) // 2
        
        # Check if we have space
        if x_offset + 28 > 128:
            break
            
        # Place digit on canvas
        canvas[y_offset:y_offset+28, x_offset:x_offset+28] = digit
        sequence.append(str(label))
        x_offset += 20
    
    return canvas, ''.join(sequence)

def apply_transformations(image):
    # Add random rotation
    rows, cols = image.shape
    angle = randint(-10, 10)
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    image = cv2.warpAffine(image, M, (cols, rows))
    
    # Add slight noise
    noise = np.random.randint(0, 25, image.shape, dtype=np.uint8)
    image = cv2.add(image, noise)
    
    return image

def generate_dataset(num_samples=10000):
    # Load MNIST data
    digit_images, digit_labels = load_mnist()
    
    # Create output directories
    os.makedirs('multi_digit_ocr/data/images/train', exist_ok=True)
    os.makedirs('multi_digit_ocr/data/labels', exist_ok=True)
    
    print("\nGenerating dataset...")
    with open('multi_digit_ocr/data/labels/train.txt', 'w') as f:
        successful = 0
        attempts = 0
        
        while successful < num_samples and attempts < num_samples * 2:
            attempts += 1
            
            # Generate sequence
            num_digits = randint(2, 4)
            canvas, sequence = add_digits_to_canvas(digit_images, digit_labels, num_digits)
            
            if len(sequence) >= 2:  # Ensure we have at least 2 digits
                # Apply transformations
                canvas = apply_transformations(canvas)
                
                # Save image and label
                cv2.imwrite(f'multi_digit_ocr/data/images/train/sequence_{successful}.png', canvas)
                f.write(f'sequence_{successful}.png {sequence}\n')
                successful += 1
                
                if successful % 100 == 0:
                    print(f"Generated {successful}/{num_samples} images")
    
    print(f"\nDataset generation complete.")
    print(f"Created {successful} images out of {attempts} attempts")
    print(f"Success rate: {successful/attempts*100:.2f}%")

if __name__ == '__main__':
    generate_dataset(10000) 