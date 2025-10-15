# IBM watsonx.ai integration for speech-to-text and text-to-speech
from ibm_watson import SpeechToTextV1, TextToSpeechV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import os
from dotenv import load_dotenv

load_dotenv()

# Authenticate with IBM Cloud
authenticator = IAMAuthenticator(os.getenv("WATSONX_APIKEY"))

# Initialize STT and TTS services
stt = SpeechToTextV1(authenticator=authenticator)
stt.set_service_url(os.getenv("WATSONX_URL"))

tts = TextToSpeechV1(authenticator=authenticator)
tts.set_service_url(os.getenv("WATSONX_URL"))

# Convert audio to text
def transcribe_audio(audio_file):
    with open(audio_file, 'rb') as f:
        result = stt.recognize(audio=f, content_type='audio/wav').get_result()
    return result['results'][0]['alternatives'][0]['transcript']

# Convert text to audio
def synthesize_speech(text):
    response = tts.synthesize(text, voice='en-US_AllisonV3Voice', accept='audio/wav').get_result()
    return response.content
