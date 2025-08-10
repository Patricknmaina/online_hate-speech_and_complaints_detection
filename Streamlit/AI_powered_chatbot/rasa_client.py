import requests
import json
from typing import Dict, Any, Optional

class RasaClient:
    def __init__(self, rasa_url: str = "http://localhost:5005"):
        self.rasa_url = rasa_url
        self.session_id = "streamlit_user"
    
    def send_message(self, message: str) -> Optional[Dict[str, Any]]:
        """
        Send a message to Rasa and get the response
        """
        try:
            payload = {
                "sender": self.session_id,
                "message": message
            }
            
            response = requests.post(
                f"{self.rasa_url}/webhooks/rest/webhook",
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Rasa API error: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"Error communicating with Rasa: {str(e)}")
            return None
    
    def is_available(self) -> bool:
        """
        Check if Rasa server is available
        """
        try:
            response = requests.get(f"{self.rasa_url}/status", timeout=5)
            return response.status_code == 200
        except:
            return False
