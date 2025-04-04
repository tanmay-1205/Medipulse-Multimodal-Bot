# test_performance.py
import os
import time
import tempfile
from voice_of_the_patient import transcribe_with_groq
from brain_of_the_doctor import encode_image, analyze_image_with_query
from voice_of_the_doctor import text_to_speech_with_gtts, text_to_speech_with_elevenlabs

def measure_performance(test_name, func, *args, **kwargs):
    """Measure and report the execution time of a function."""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"{test_name} completed in {execution_time:.2f} seconds")
    return result, execution_time

def test_different_model_sizes():
    """Test performance with different model sizes if available."""
    # Prepare test data
    test_prompt = "What do you see on my skin?"
    test_image_path = "AIMedicalBot-main/AIMedicalBot-main/acne.jpg"  # Make sure this exists
    
    if not os.path.exists(test_image_path):
        # Create a placeholder image if the test image doesn't exist
        with open(test_image_path, "wb") as f:
            f.write(b"X" * 1000)  # Simple placeholder
    
    encoded_image = encode_image(test_image_path)
    system_prompt = """You have to act as a professional doctor, i know you are not but this is for learning purpose. 
            What's in this image?. Do you find anything wrong with it medically?
            Keep your answer concise (max 2 sentences). No preamble, start your answer right away please"""
    
    # Test with different model sizes if available
    models = [
        "llama-3.2-11b-vision-preview",  # Your current model
        "llama-3.2-90b-vision-preview"   # Larger model if needed/available
    ]
    
    results = {}
    for model in models:
        try:
            print(f"\nTesting with {model}...")
            query = system_prompt + test_prompt
            result, exec_time = measure_performance(
                f"{model} inference",
                analyze_image_with_query,
                query=query,
                model=model,
                encoded_image=encoded_image
            )
            results[model] = {
                "response": result,
                "time": exec_time
            }
            print(f"Response: {result[:100]}..." if len(result) > 100 else f"Response: {result}")
        except Exception as e:
            print(f"Error with {model}: {e}")
            results[model] = {
                "response": f"Error: {str(e)}",
                "time": None
            }
    
    # Compare results if multiple models were tested
    if len(results) > 1:
        print("\nModel Performance Comparison:")
        for model, data in results.items():
            print(f"{model}: {data['time']:.2f}s" if data['time'] else f"{model}: Failed")

def test_audio_quality_impact():
    """Test how audio quality impacts transcription accuracy."""
    # Prepare test files
    good_quality = "gtts_testing.mp3"  # A clear, well-recorded audio file
    poor_quality = "short_test.mp3"  # A noisy or low-quality audio file
    
    # If test files don't exist, you would need to create or obtain them
    # For this example, we'll just use the same file twice if needed
    if not os.path.exists(good_quality) and os.path.exists("test_audio.mp3"):
        good_quality = "test_audio.mp3"
    
    if not os.path.exists(poor_quality) and os.path.exists(good_quality):
        # In a real test, you'd create a degraded version
        poor_quality = good_quality
    
    # Skip test if files aren't available
    if not (os.path.exists(good_quality) and os.path.exists(poor_quality)):
        print("Skipping audio quality test - test files not available")
        return
    
    # Test transcription with different quality audio
    print("\nTesting with good quality audio...")
    good_result, good_time = measure_performance(
        "Good quality transcription",
        transcribe_with_groq,
        stt_model="whisper-large-v3",
        audio_filepath=good_quality,
        GROQ_API_KEY=os.environ.get("GROQ_API_KEY")
    )
    
    print("\nTesting with poor quality audio...")
    poor_result, poor_time = measure_performance(
        "Poor quality transcription",
        transcribe_with_groq,
        stt_model="whisper-large-v3",
        audio_filepath=poor_quality,
        GROQ_API_KEY=os.environ.get("GROQ_API_KEY")
    )
    
    print(f"\nGood quality transcription: {good_result}")
    print(f"Poor quality transcription: {poor_result}")



if __name__ == "__main__":
    print("RUNNING PERFORMANCE TESTS\n" + "="*30)
    test_different_model_sizes()
    test_audio_quality_impact()
    print("\nAll performance tests completed.")