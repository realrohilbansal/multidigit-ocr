from predict import DigitPredictor
import os

def test_models():
    # Check if models exist
    cnn_path = 'multi_digit_ocr/models/digit_cnn_epoch_25.pth'
    seq_path = 'multi_digit_ocr/models/sequence_model_best.pth'
    
    if not os.path.exists(cnn_path):
        print(f"CNN model not found at: {cnn_path}")
        return
    if not os.path.exists(seq_path):
        print(f"Sequence model not found at: {seq_path}")
        return
    
    # Initialize predictor
    predictor = DigitPredictor(
        cnn_model_path=cnn_path,
        sequence_model_path=seq_path
    )
    
    # Test sequence model
    test_image = 'multi_digit_ocr/data/images/test/17.jpeg'
    
    # Check if test image exists
    if not os.path.exists(test_image):
        print(f"Test image not found at: {test_image}")
        return
        
    print(f"Processing image: {test_image}")
    prediction = predictor.predict_sequence(test_image)
    
    if prediction:
        print(f"Predicted sequence: {prediction}")
    else:
        print("Failed to get prediction")

if __name__ == '__main__':
    test_models() 