import os
import tempfile
import base64
import json

class TestFixtures:
    """Provides test fixtures for medical bot tests."""
    
    @staticmethod
    def get_sample_image_path():
        """Create a sample image file and return its path."""
        test_image_path = os.path.join(tempfile.gettempdir(), "sample_medical_image.jpg")
        
        # Create a minimal valid JPEG file if it doesn't exist
        if not os.path.exists(test_image_path):
            with open(test_image_path, "wb") as f:
                # Create a minimal valid JPEG file
                f.write(b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00C\x00\xff\xc0\x00\x11\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x14\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x09\xff\xc4\x00\x14\x10\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xda\x00\x08\x01\x01\x00\x01?\x10\xff\xd9')
        
        return test_image_path
    
    @staticmethod
    def get_sample_audio_path():
        """Create a sample audio file and return its path."""
        test_audio_path = os.path.join(tempfile.gettempdir(), "sample_patient_audio.mp3")
        
        # Create a dummy audio file if it doesn't exist
        if not os.path.exists(test_audio_path):
            with open(test_audio_path, "wb") as f:
                f.write(b'RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00')
        
        return test_audio_path
    
    @staticmethod
    def get_mock_groq_image_response():
        """Return a mock Groq image analysis response."""
        return {
            "choices": [
                {
                    "message": {
                        "content": "With what I see, I think you have a mild case of eczema. I recommend applying a moisturizer and avoiding harsh soaps. If it persists, consult a dermatologist."
                    }
                }
            ]
        }
    
    @staticmethod
    def get_mock_groq_audio_response():
        """Return a mock Groq audio transcription response."""
        return {
            "text": "I have a rash on my arm that has been itching for three days and getting worse."
        }
    
    @staticmethod
    def get_test_env_vars():
        """Return a dictionary of environment variables for testing."""
        return {
            "GROQ_API_KEY": "test_groq_api_key"
        }
    
    @staticmethod
    def setup_test_env():
        """Set up the testing environment."""
        env_vars = TestFixtures.get_test_env_vars()
        for key, value in env_vars.items():
            if key not in os.environ:
                os.environ[key] = value
    
    @staticmethod
    def get_sample_queries():
        """Return a list of sample patient queries for testing."""
        return [
            "What is this rash on my arm?",
            "I've had this mole for years, should I be concerned?",
            "My skin is itchy and red, what could it be?",
            "Is this acne or something else?",
            "I have a growth on my neck, is it dangerous?",
            "Should I be worried about these spots?"
        ]
    
    @staticmethod
    def get_sample_doctor_responses():
        """Return a list of sample doctor responses for testing."""
        return [
            "With what I see, I think you have contact dermatitis. I recommend using a mild cortisone cream and avoiding whatever triggered it.",
            "With what I see, I think you have a benign mole with regular borders and coloration. Keep monitoring for any changes in size, shape, or color.",
            "With what I see, I think you have eczema. I recommend moisturizing regularly and avoiding hot showers.",
            "With what I see, I think you have mild acne. I suggest gentle cleansing twice daily and an over-the-counter benzoyl peroxide treatment.",
            "With what I see, I think you have a lipoma, which is a harmless fatty lump. No treatment is necessary unless it bothers you.",
            "With what I see, I think you have tinea versicolor, a common fungal infection. Try an over-the-counter antifungal cream."
        ]
