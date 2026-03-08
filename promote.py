import requests

res = requests.post("http://localhost:8000/auth/promote", json={
    "email": "tarunpangala7@gmail.com",
    "admin_secret": "Tarun0909@2005"
})
print(res.json())