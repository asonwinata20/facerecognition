import firebase_admin
from firebase_admin import credentials, db
from datetime import datetime

cred = credentials.Certificate("firebase_key.json")  # Your Firebase service account
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://face-recognition-6c715-default-rtdb.asia-southeast1.firebasedatabase.app/'
})

def mark_attendance(user_id):
    ref = db.reference(f'/attendance/{user_id}')
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    ref.push({"timestamp": now})
