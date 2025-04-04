# test_error_handling.py
import os
import pytest
from voice_of_the_patient import transcribe_with_groq
from brain_of_the_doctor import encode_image, analyze_image_with_query
from voice_of_the_doctor import text_to_speech_with_gtts, text_to_speech_with_elevenlabs

def test_missing_inputs():
    """Test how the system handles missing or invalid inputs."""
    # Test case 1: No audio file
    print("Test case 1: Missing audio file")
    try:
        transcription = transcribe_with_groq(
            stt_model="whisper-large-v3",
            audio_filepath="nonexistent_audio.mp3",
            GROQ_API_KEY=os.environ.get("GROQ_API_KEY")
        )
        print("Error: Should have raised an exception for missing audio file")
    except Exception as e:
        print(f"Successfully caught exception: {e}")
    
    # Test case 2: No image file
    print("\nTest case 2: Missing image file")
    try:
        encoded_image = encode_image("nonexistent_image.jpg")
        print("Error: Should have raised an exception for missing image file")
    except Exception as e:
        print(f"Successfully caught exception: {e}")
    
    # Test case 3: Empty query
    print("\nTest case 3: Empty query")
    # Create a small test image if needed
    test_image_path = "AIMedicalBot-main/AIMedicalBot-main/acne.jpg"
    with open(test_image_path, "wb") as f:
        f.write(b"test")  # Minimal file to allow encoding
    
    try:
        encoded_image = encode_image(test_image_path)
        result = analyze_image_with_query(
            query="", 
            model="llama-3.2-11b-vision-preview", 
            encoded_image=encoded_image
        )
        print(f"Query result with empty prompt: {result}")
    except Exception as e:
        print(f"Exception on empty query: {e}")
    
    # Clean up test file
    if os.path.exists(test_image_path):
        os.remove(test_image_path)

def test_different_tts_services():
    """Test both text-to-speech services."""
    test_text = "This is a test of different text to speech services."
    output_gtts = "test_gtts_output.mp3"
    output_eleven = "test_eleven_output.mp3"
    
    print("\nTesting gTTS service...")
    try:
        text_to_speech_with_gtts(input_text=test_text, output_filepath=output_gtts)
        if os.path.exists(output_gtts):
            print(f"gTTS test successful. Output saved to {output_gtts}")
        else:
            print("gTTS test failed: No output file produced")
    except Exception as e:
        print(f"gTTS test exception: {e}")
    
    print("\nTesting ElevenLabs service...")
    if os.environ.get("ELEVENLABS_API_KEY"):
        try:
            text_to_speech_with_elevenlabs(input_text=test_text, output_filepath=output_eleven)
            if os.path.exists(output_eleven):
                print(f"ElevenLabs test successful. Output saved to {output_eleven}")
            else:
                print("ElevenLabs test failed: No output file produced")
        except Exception as e:
            print(f"ElevenLabs test exception: {e}")
    else:
        print("ElevenLabs API key not found. Skipping this test.")

def test_input_variations():
    """Test the system with different types of inputs."""
    # Create a very short audio file for testing
    short_audio = "short_test.mp3"
    if not os.path.exists(short_audio):
        from pydub import AudioSegment
        from pydub.generators import Sine
        
        # Generate a 1-second sine wave
        sine_wave = Sine(440).to_audio_segment(duration=1000)
        sine_wave.export(short_audio, format="mp3")
    
    print("\nTesting with very short audio...")
    try:
        transcription = transcribe_with_groq(
            stt_model="whisper-large-v3",
            audio_filepath=short_audio,
            GROQ_API_KEY=os.environ.get("GROQ_API_KEY")
        )
        print(f"Short audio transcription: {transcription}")
    except Exception as e:
        print(f"Short audio exception: {e}")
    

if __name__ == "__main__":
    print("RUNNING ERROR HANDLING TESTS\n" + "="*30)
    test_missing_inputs()
    test_different_tts_services()
    test_input_variations()
    print("\nAll tests completed.")