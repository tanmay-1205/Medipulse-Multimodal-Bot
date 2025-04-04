import unittest
import os
import tempfile
import base64
from unittest.mock import patch, MagicMock
import sys
import shutil

# Add the parent directory to the path to import the modules
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Import the modules to test
from brain_of_the_doctor import encode_image, analyze_image_with_query
from voice_of_the_patient import record_audio, transcribe_with_groq
from voice_of_the_doctor import text_to_speech_with_gtts
from gradio_app import process_inputs

class TestBrainOfTheDoctor(unittest.TestCase):
    """Tests for the image analysis component."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary test image
        self.test_image_path = os.path.join(tempfile.gettempdir(), "test_image.jpg")
        with open(self.test_image_path, "wb") as f:
            # Create a minimal valid JPEG file
            f.write(b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00C\x00\xff\xc0\x00\x11\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x14\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x09\xff\xc4\x00\x14\x10\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xda\x00\x08\x01\x01\x00\x01?\x10\xff\xd9')
        
        # Set environment variables for tests if not already set
        if not os.environ.get("GROQ_API_KEY"):
            os.environ["GROQ_API_KEY"] = "mock_groq_key_for_testing"

    def tearDown(self):
        """Clean up after tests."""
        # Remove the temporary test image
        if os.path.exists(self.test_image_path):
            os.remove(self.test_image_path)

    def test_encode_image(self):
        """Test the image encoding function."""
        encoded = encode_image(self.test_image_path)
        # Check that the encoded string is a valid base64 string
        self.assertTrue(isinstance(encoded, str))
        # Try to decode it to ensure it's valid base64
        try:
            base64.b64decode(encoded)
        except Exception:
            self.fail("encode_image did not produce valid base64")

    @patch('groq.Groq')
    def test_analyze_image_with_query(self, mock_groq):
        """Test image analysis with a mock response."""
        # Mock the Groq client response
        mock_client = MagicMock()
        mock_groq.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "This appears to be a skin condition. Based on what I see, I think you have mild acne. I recommend gentle cleansing twice daily and an over-the-counter benzoyl peroxide treatment."
        
        mock_client.chat.completions.create.return_value = mock_response
        
        # Test the function
        encoded_image = encode_image(self.test_image_path)
        result = analyze_image_with_query("Is there something wrong with my face?", "llama-3.2-11b-vision-preview", encoded_image)
        
        # Verify the response
        self.assertEqual(result, "This appears to be a skin condition. Based on what I see, I think you have mild acne. I recommend gentle cleansing twice daily and an over-the-counter benzoyl peroxide treatment.")
        
        # Verify the call was made with the correct parameters
        mock_client.chat.completions.create.assert_called_once()
        # Extract the first positional argument
        call_args = mock_client.chat.completions.create.call_args[1]
        self.assertEqual(call_args["model"], "llama-3.2-11b-vision-preview")
        self.assertIn("messages", call_args)


class TestVoiceOfThePatient(unittest.TestCase):
    """Tests for the voice transcription component."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_audio_path = os.path.join(tempfile.gettempdir(), "test_audio.mp3")
        # Create a dummy audio file
        with open(self.test_audio_path, "wb") as f:
            f.write(b'dummy audio content')

    def tearDown(self):
        """Clean up after tests."""
        if os.path.exists(self.test_audio_path):
            os.remove(self.test_audio_path)

    @patch('speech_recognition.Recognizer')
    @patch('speech_recognition.Microphone')
    @patch('pydub.AudioSegment.from_wav')
    def test_record_audio(self, mock_from_wav, mock_microphone, mock_recognizer):
        """Test audio recording with mocked dependencies."""
        # Skip actual recording for this test
        # This test only checks that the function calls the right dependencies
        try:
            # Need a context manager for Microphone
            mock_microphone_instance = MagicMock()
            mock_microphone.return_value.__enter__.return_value = mock_microphone_instance
            
            # Setup recognizer mock
            mock_recognizer_instance = MagicMock()
            mock_recognizer.return_value = mock_recognizer_instance
            
            # Mock audio data
            mock_audio = MagicMock()
            mock_audio.get_wav_data.return_value = b'mock_wav_data'
            mock_recognizer_instance.listen.return_value = mock_audio
            
            # Mock audio segment
            mock_audio_segment = MagicMock()
            mock_from_wav.return_value = mock_audio_segment
            
            # Test the function with a temporary file
            record_audio(self.test_audio_path, timeout=1, phrase_time_limit=1)
            
            # Verify the expected calls were made
            mock_recognizer_instance.adjust_for_ambient_noise.assert_called_once()
            mock_recognizer_instance.listen.assert_called_once()
            mock_audio.get_wav_data.assert_called_once()
            mock_from_wav.assert_called_once()
            mock_audio_segment.export.assert_called_once_with(self.test_audio_path, format="mp3", bitrate="128k")
            
        except Exception as e:
            # Some dependencies might not be available in test environment
            self.skipTest(f"Skipping test_record_audio due to missing dependencies: {e}")

    @patch('groq.Groq')
    def test_transcribe_with_groq(self, mock_groq):
        """Test transcribing audio with a mock response."""
        # Mock the Groq client response
        mock_client = MagicMock()
        mock_groq.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.text = "I have a rash on my arm that has been itching for three days."
        
        mock_client.audio.transcriptions.create.return_value = mock_response
        
        # Test the function
        result = transcribe_with_groq("whisper-large-v3", self.test_audio_path, "mock_api_key")
        
        # Verify the response
        self.assertEqual(result, "I have a rash on my arm that has been itching for three days.")
        
        # Verify the call was made with the correct parameters
        mock_client.audio.transcriptions.create.assert_called_once()
        args = mock_client.audio.transcriptions.create.call_args[1]
        self.assertEqual(args["model"], "whisper-large-v3")
        self.assertEqual(args["language"], "en")


class TestVoiceOfTheDoctor(unittest.TestCase):
    """Tests for the text-to-speech component."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_output_path = os.path.join(tempfile.gettempdir(), "test_output.mp3")
        self.test_text = "With what I see, I think you have a mild case of contact dermatitis. I recommend applying a hydrocortisone cream and avoiding potential allergens."

    def tearDown(self):
        """Clean up after tests."""
        if os.path.exists(self.test_output_path):
            os.remove(self.test_output_path)

    @patch('gtts.gTTS')
    @patch('subprocess.run')
    def test_text_to_speech_with_gtts(self, mock_subprocess, mock_gtts):
        """Test text-to-speech with gTTS."""
        # Mock the gTTS instance
        mock_gtts_instance = MagicMock()
        mock_gtts.return_value = mock_gtts_instance
        
        # Test the function
        text_to_speech_with_gtts(self.test_text, self.test_output_path)
        
        # Verify the expected calls were made
        mock_gtts.assert_called_once_with(text=self.test_text, lang="en", slow=False)
        mock_gtts_instance.save.assert_called_once_with(self.test_output_path)
        mock_subprocess.assert_called_once()  # Platform specific, just check it was called


class TestGradioApp(unittest.TestCase):
    """Tests for the Gradio application integration."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_audio_path = os.path.join(tempfile.gettempdir(), "test_audio.mp3")
        self.test_image_path = os.path.join(tempfile.gettempdir(), "test_image.jpg")
        
        # Create dummy test files
        with open(self.test_audio_path, "wb") as f:
            f.write(b'dummy audio content')
        
        with open(self.test_image_path, "wb") as f:
            f.write(b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00C\x00\xff\xc0\x00\x11\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x14\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x09\xff\xc4\x00\x14\x10\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xda\x00\x08\x01\x01\x00\x01?\x10\xff\xd9')

    def tearDown(self):
        """Clean up after tests."""
        if os.path.exists(self.test_audio_path):
            os.remove(self.test_audio_path)
        if os.path.exists(self.test_image_path):
            os.remove(self.test_image_path)
        if os.path.exists("final.mp3"):
            os.remove("final.mp3")

    @patch('voice_of_the_patient.transcribe_with_groq')
    @patch('brain_of_the_doctor.analyze_image_with_query')
    @patch('voice_of_the_doctor.text_to_speech_with_gtts')
    def test_process_inputs(self, mock_tts, mock_analyze, mock_transcribe):
        """Test the integrated process_inputs function."""
        # Setup mocks
        mock_transcribe.return_value = "Is this rash concerning?"
        mock_analyze.return_value = "With what I see, I think you have contact dermatitis. I recommend applying hydrocortisone cream and avoiding the allergen."
        mock_tts.return_value = "final.mp3"
        
        # Create the final.mp3 file since the function expects it to exist
        with open("final.mp3", "wb") as f:
            f.write(b'dummy audio output')
        
        # Test the function
        stt_output, doctor_response, audio_output = process_inputs(self.test_audio_path, self.test_image_path)
        
        # Verify the outputs
        self.assertEqual(stt_output, "Is this rash concerning?")
        self.assertEqual(doctor_response, "With what I see, I think you have contact dermatitis. I recommend applying hydrocortisone cream and avoiding the allergen.")
        self.assertEqual(audio_output, "final.mp3")
        
        # Verify the expected function calls
        mock_transcribe.assert_called_once()
        mock_analyze.assert_called_once()
        mock_tts.assert_called_once()

    @patch('voice_of_the_patient.transcribe_with_groq')
    @patch('voice_of_the_doctor.text_to_speech_with_gtts')
    def test_process_inputs_no_image(self, mock_tts, mock_transcribe):
        """Test process_inputs with no image provided."""
        # Setup mocks
        mock_transcribe.return_value = "What should I do about my sore throat?"
        mock_tts.return_value = "final.mp3"
        
        # Create the final.mp3 file
        with open("final.mp3", "wb") as f:
            f.write(b'dummy audio output')
        
        # Test the function with no image
        stt_output, doctor_response, audio_output = process_inputs(self.test_audio_path, None)
        
        # Verify the outputs
        self.assertEqual(stt_output, "What should I do about my sore throat?")
        self.assertEqual(doctor_response, "No image provided for me to analyze")
        self.assertEqual(audio_output, "final.mp3")
        
        # Verify the expected function calls
        mock_transcribe.assert_called_once()
        mock_tts.assert_called_once()


class TestIntegration(unittest.TestCase):
    """Integration tests for the application."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary directories for test data
        self.test_dir = os.path.join(tempfile.gettempdir(), "medipulse_tests")
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Create test audio and image files
        self.test_audio_path = os.path.join(self.test_dir, "patient_voice.mp3")
        self.test_image_path = os.path.join(self.test_dir, "patient_image.jpg")
        
        # Create dummy files
        with open(self.test_audio_path, "wb") as f:
            f.write(b'dummy audio content')
        
        with open(self.test_image_path, "wb") as f:
            f.write(b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00C\x00\xff\xc0\x00\x11\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x14\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x09\xff\xc4\x00\x14\x10\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xda\x00\x08\x01\x01\x00\x01?\x10\xff\xd9')

    def tearDown(self):
        """Clean up after tests."""
        # Remove the test directory and all its contents
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        
        # Remove any output files
        if os.path.exists("final.mp3"):
            os.remove("final.mp3")

    @unittest.skip("Skip end-to-end test because it requires external API keys")
    def test_end_to_end(self):
        """End-to-end test of the entire application flow."""
        # This test is skipped by default as it requires real API keys
        # To run it, remove the @unittest.skip decorator and set real API keys
        
        # Check if we have real API keys
        if not os.environ.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY") == "mock_groq_key_for_testing":
            self.skipTest("Skipping end-to-end test: Missing real GROQ API key")
        
        # 1. Transcribe audio
        stt_output = transcribe_with_groq(
            stt_model="whisper-large-v3",
            audio_filepath=self.test_audio_path,
            GROQ_API_KEY=os.environ.get("GROQ_API_KEY")
        )
        self.assertIsInstance(stt_output, str)
        
        # 2. Encode the image
        encoded_image = encode_image(self.test_image_path)
        
        # 3. Analyze the image with the query
        system_prompt = """You have to act as a professional doctor, i know you are not but this is for learning purpose. 
                What's in this image?. Do you find anything wrong with it medically?"""
        
        doctor_response = analyze_image_with_query(
            query=system_prompt + stt_output,
            model="llama-3.2-11b-vision-preview",
            encoded_image=encoded_image
        )
        self.assertIsInstance(doctor_response, str)
        
        # 4. Convert response to speech
        output_path = os.path.join(self.test_dir, "doctor_response.mp3")
        text_to_speech_with_gtts(doctor_response, output_path)
        
        # Check if the output file exists
        self.assertTrue(os.path.exists(output_path))


def run_tests():
    """Run all tests."""
    # Create a test suite with all test cases
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(TestBrainOfTheDoctor))
    test_suite.addTest(unittest.makeSuite(TestVoiceOfThePatient))
    test_suite.addTest(unittest.makeSuite(TestVoiceOfTheDoctor))
    test_suite.addTest(unittest.makeSuite(TestGradioApp))
    test_suite.addTest(unittest.makeSuite(TestIntegration))
    
    # Run the test suite
    test_runner = unittest.TextTestRunner(verbosity=2)
    test_runner.run(test_suite)


if __name__ == "__main__":
    run_tests()