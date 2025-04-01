import requests

API_URL = "http://127.0.0.1:5003/generate_response"


query = "i bought washing machine from your store, it makes loud noise, what should i do?"

def call_chatbot_api(query):
    # Prepare request data
    data = {"query": query}

    try:
        # Send POST request
        response = requests.post(API_URL, data=data)

        # Parse response
        if response.status_code == 200:
            result = response.json()
            print("Response from Chatbot:", result.get("response", "No response received"))
        else:
            print("Error:", response.status_code, response.text)

    except Exception as e:
        print("Failed to connect to API:", str(e))


call_chatbot_api(query)
