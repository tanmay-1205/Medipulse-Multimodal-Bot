# test_medical_app.py
import os
from voice_of_the_patient import transcribe_with_groq
from brain_of_the_doctor import encode_image, analyze_image_with_query
from voice_of_the_doctor import text_to_speech_with_gtts

def test_end_to_end_flow():
    # Step 1: Set up test files
    test_audio_path = "AIMedicalBot-main/AIMedicalBot-main/patient_voice_test.mp3"  # You should have a pre-recorded test audio file
    test_image_path = "AIMedicalBot-main/AIMedicalBot-main/skin_rash.jpg"  # You should have a test image with some visible skin condition
    output_speech_path = "test_output.mp3"
    
    # Step 2: Test speech-to-text
    print("Testing speech-to-text...")
    transcription = transcribe_with_groq(
        stt_model="whisper-large-v3",
        audio_filepath=test_audio_path,
        GROQ_API_KEY=os.environ.get("GROQ_API_KEY")
    )
    print(f"Transcription result: {transcription}")
    
    # Step 3: Test image analysis
    print("\nTesting image analysis...")
    system_prompt = """You have to act as a professional doctor, i know you are not but this is for learning purpose. 
            What's in this image?. Do you find anything wrong with it medically? 
            If you make a differential, suggest some remedies for them. Donot add any numbers or special characters in 
            your response. Your response should be in one long paragraph. Also always answer as if you are answering to a real person.
            Donot say 'In the image I see' but say 'With what I see, I think you have ....'
            Dont respond as an AI model in markdown, your answer should mimic that of an actual doctor not an AI bot, 
            Keep your answer concise (max 2 sentences). No preamble, start your answer right away please"""
    
    encoded_image = encode_image(test_image_path)
    query = system_prompt + transcription
    doctor_response = analyze_image_with_query(
        query=query, 
        model="llama-3.2-11b-vision-preview", 
        encoded_image=encoded_image
    )
    print(f"Doctor's response: {doctor_response}")
    
    # Step 4: Test text-to-speech
    print("\nTesting text-to-speech...")
    text_to_speech_with_gtts(
        input_text=doctor_response,
        output_filepath=output_speech_path
    )
    print(f"Speech output saved to: {output_speech_path}")
    
    # Verify all steps completed
    if os.path.exists(output_speech_path):
        print("\nTest completed successfully!")
    else:
        print("\nTest failed: Speech output file was not created.")

if __name__ == "__main__":
    test_end_to_end_flow()