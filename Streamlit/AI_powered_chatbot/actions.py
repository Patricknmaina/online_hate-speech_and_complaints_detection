from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import requests
import json

class ActionAnalyzeTweet(Action):
    def name(self) -> Text:
        return "action_analyze_tweet"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        # Get the tweet text from the entity
        tweet_text = None
        for entity in tracker.latest_message['entities']:
            if entity['entity'] == 'tweet_text':
                tweet_text = entity['value']
                break
        
        # If no entity found, try to get from the message text
        if not tweet_text:
            message_text = tracker.latest_message.get('text', '')
            # Simple fallback - use the entire message as tweet
            tweet_text = message_text
        
        if tweet_text:
            # Call the FastAPI endpoint for analysis
            try:
                api_url = "http://localhost:8000/predict"
                payload = {
                    "text": tweet_text,
                    "user_id": None
                }
                
                response = requests.post(api_url, json=payload, timeout=10)
                
                if response.status_code == 200:
                    result = response.json()
                    prediction = result['prediction']
                    confidence = result['confidence']
                    
                    # Generate response based on prediction
                    if 'positive' in prediction.lower():
                        response_text = f"�� This tweet appears to be **positive** towards Safaricom with {confidence:.1%} confidence. It seems like a satisfied customer experience!"
                    elif 'negative' in prediction.lower():
                        response_text = f"�� This tweet appears to be **negative** towards Safaricom with {confidence:.1%} confidence. This might indicate a customer service issue or network problem."
                    elif 'complaint' in prediction.lower():
                        response_text = f"⚠️ This tweet appears to be a **complaint** with {confidence:.1%} confidence. Safaricom should address this customer concern."
                    elif 'hate' in prediction.lower():
                        response_text = f"🚨 This tweet contains **hate speech** with {confidence:.1%} confidence. This type of content should be flagged for review."
                    else:
                        response_text = f"�� This tweet appears to be **neutral** with {confidence:.1%} confidence. It's neither positive nor negative towards Safaricom."
                    
                    dispatcher.utter_message(text=response_text)
                else:
                    dispatcher.utter_message(text="Sorry, I couldn't analyze that tweet at the moment. Please try again later.")
                    
            except Exception as e:
                dispatcher.utter_message(text=f"Sorry, there was an error analyzing the tweet: {str(e)}")
        else:
            dispatcher.utter_message(text="I didn't catch the tweet text. Could you please share the tweet you'd like me to analyze?")
        
        return []