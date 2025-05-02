from typing import Dict, List, Any
import requests
import json



# --- Snippet 1: get_response_v1 (Appears incomplete or simple delegate) ---
def get_response(self, message: List[Dict[str, str]]) -> str:
    # timeLimie = self.spent() # Commented out or unrelated line
    return self.client.get_response(message)

class LLMClient:
    # --- Snippet 2: Configuration Variables ---
    # vertex ai connection with the llm provider, service account auth
    API_ENDPOINT = "generativelanguage.googleapis.com"
    PROJECT_ID = "v1beta"
    MODEL_ID = "gemini-1.5-flash" # Model specified

    # bearer_token = generate_token() # Assumes generate_token() is defined elsewhere
    bearer_token = "ya29.a0AZYkNZjEovc_7lB-03PXZ2_yz0x3xYIwSQZYoTgOa5-Px54w1TYooJzxUZjsFKELXfYfLiaE4j0f3GEG0UYN_H444YiP0Ks_3Q0saJO7MIim3WdTIz-rKehv33C8GF_Fb0Ue-UGonfB81cOZ7DjjN2LtKbkVKWeEQh0vnXJTUAaCgYKAdQSARISFQHGX2MihcMLd5H21GLyzTrH0UC-0A0177"
    # --- Snippet 3: get_headers ---
    def get_headers(self) -> Dict[str, str]:
        """Generate auth headers.""" # Docstring assumed
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.bearer_token}" # Uses global bearer_token
        }
        return headers

    # # --- Snippet 4: get_payload ---
    # def get_payload(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    #     """Generate payload for the API request.""" # Docstring assumed
    #     # Structure specific to Vertex AI Gemini API
    #     payload = {
    #         "contents": [
    #             {
    #                 "role": "user", # Note: Hardcoded role, might need adjustment for conversation history
    #                 "parts": messages # Assumes 'messages' fits the 'parts' structure
    #             }
    #         ]
    #         # Generation config, safety settings etc. might be missing
    #     }
    #     return payload

    def get_response(self, messages: List[Dict[str, str]]) -> str:

        url = f"https://{self.API_ENDPOINT}/{self.PROJECT_ID}/models/{self.MODEL_ID}:generateContent" 
        print("****************************************messages****************************************")
        print(messages)
        print("*****************************************************************************************")
        response = requests.post(url, headers=self.get_headers(), json=messages) 

        content = ""

        try:
            response_data = json.loads(response.text)
            print("****************************************response_data****************************************")
            print(response_data)
            print("*****************************************************************************************")


            print("****************************************response_data['candidates'][0]['content']['parts']****************************************")
            print(response_data["candidates"][0]["content"]["parts"])
            print("*****************************************************************************************")

            return response_data['candidates'][0]['content']['parts']
        except ResponseError as e: # Custom Exception class
            return e.message # Return error message directly
        # --- End of potentially conflicting logic ---


        # --- Start of alternative/potentially old try-except block ---
        # This block seems to contradict the streaming URL and custom parsing above.
        # It looks more like handling a standard REST API non-streaming response,
        # possibly from OpenAI or a similar API format.
        try:
            # Redundant request? This re-sends the request.
            response = requests.post(url, headers=self.get_headers(), json=self.get_payload(messages))
            response_status_code = response.status_code

            if response.status_code != 200:
                # Raise custom error (ResponseError definition not shown)
                raise ResponseError(f"Status code: {response.status_code}")

            # Attempts to parse JSON like a standard OpenAI response
            response_data = response.json() # Assumes response is JSON, not a stream
            # This access pattern is typical for OpenAI API, not Vertex AI stream
            return response_data['choices'][0]['message']['content']

        except ResponseError as e: # Catching the custom error
            return e.message

        except requests.exceptions.RequestException as e: # Catching general request errors
            # Handle connection errors, timeouts etc.
            return f"Request failed: {e}"
        except Exception as e: # Catch any other error
            # General fallback
            return f"An unexpected error occurred: {e}"

        # --- End of alternative try-except block ---

        # return code might be unreachable due to earlier returns