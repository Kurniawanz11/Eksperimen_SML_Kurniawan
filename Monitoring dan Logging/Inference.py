import requests

url = "http://127.0.0.1:8000/predict"

data = {
    "0": 63,
    "1": 1,
    "2": 3,
    "3": 145,
    "4": 233,
    "5": 1,
    "6": 0,
    "7": 150,
    "8": 0,
    "9": 2.3,
    "10": 0,
    "11": 0,
    "12": 1
}

response = requests.post(
    url,
    json=data
)

print("Status:", response.status_code)
print(response.text)