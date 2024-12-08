import torch
import cv2
import torchvision.transforms as transforms
from preprocess import preprocess_image
from segmentation import segment_digits
from train_cnn import DigitRecognizer
# ... (previous imports) ...
from sequence_model import SequenceRecognizer
import torch.nn.functional as F

class DigitPredictor:
    def __init__(self, cnn_model_path=None, sequence_model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load CNN model for individual digits
        if cnn_model_path:
            self.cnn_model = DigitRecognizer().to(self.device)
            self.cnn_model.load_state_dict(
                torch.load(cnn_model_path, map_location=self.device, weights_only=True)
            )
            self.cnn_model.eval()
        else:
            self.cnn_model = None
            
        # Load sequence model for entire numbers
        if sequence_model_path:
            try:
                self.sequence_model = SequenceRecognizer().to(self.device)
                checkpoint = torch.load(
                    sequence_model_path, 
                    map_location=self.device
                )
                if isinstance(checkpoint, dict):
                    self.sequence_model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.sequence_model.load_state_dict(checkpoint)
                self.sequence_model.eval()
                print("Successfully loaded sequence model")
            except Exception as e:
                print(f"Error loading sequence model: {str(e)}")
                self.sequence_model = None
        else:
            self.sequence_model = None

        # Define transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def preprocess_image(self, image_path):
        # Read image in grayscale
        print(f"Loading image from: {image_path}")
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"Failed to load image at: {image_path}")
            
        print(f"Original image shape: {image.shape}")
        
        # Resize to fixed height while maintaining aspect ratio
        target_height = 32
        aspect_ratio = image.shape[1] / image.shape[0]
        target_width = int(target_height * aspect_ratio)
        target_width = min(target_width, 512)  # Cap maximum width
        
        image = cv2.resize(image, (target_width, target_height))
        print(f"Resized image shape: {image.shape}")
        
        # Convert to tensor and normalize
        image = self.transform(image)
        print(f"Tensor shape after transform: {image.shape}")
        
        # Add batch dimension
        image = image.unsqueeze(0)
        print(f"Final tensor shape: {image.shape}")
        return image

    def predict_sequence(self, image_path):
        """Predict using the sequence model"""
        try:
            if self.sequence_model is None:
                raise RuntimeError("Sequence model not loaded")
                
            # Preprocess image
            image = self.preprocess_image(image_path)
            image = image.to(self.device)
            print(f"Input tensor shape: {image.shape}")
            
            # Get prediction
            with torch.no_grad():
                # Model returns tuple of (logits, length_logits)
                logits, _ = self.sequence_model(image)
                
                print(f"Output logits shape: {logits.shape}")
                
                # Get predictions
                log_probs = F.log_softmax(logits, dim=2)
                predictions = log_probs.argmax(dim=2)
                
                # Convert predictions to sequence
                pred_sequence = []
                for pred in predictions[0]:  # Take first batch
                    if pred < 10:  # Ignore blank token (10)
                        pred_sequence.append(str(pred.item()))
                
                return ''.join(pred_sequence)
                
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            import traceback
            traceback.print_exc()
            return None