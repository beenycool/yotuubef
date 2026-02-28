import re

with open("src/integrations/tts_service.py", "r") as f:
    content = f.read()

# Replace VoiceSettings with "VoiceSettings" or Any if typing doesn't have it, but we can also just use Any or a string type hint "VoiceSettings"
content = re.sub(r'-> VoiceSettings:', '-> "VoiceSettings":', content)

with open("src/integrations/tts_service.py", "w") as f:
    f.write(content)
