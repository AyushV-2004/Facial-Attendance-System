from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mysql.connector
import datetime
import logging
from typing import Optional

app = FastAPI()

class AttendanceRequest(BaseModel):
    name: str

# MySQL connection
def get_db_connection():
    try:
        conn = mysql.connector.connect(
            host='51.20.6.164',
            user='krish',
            password='krish@ml', 
            database='attendance' 
        )
        return conn
    except mysql.connector.Error as err:
        logging.error(f"Error connecting to MySQL: {err}")
        return None

@app.post("/mark_attendance")
async def mark_attendance(attendance: AttendanceRequest):
    name = attendance.name

    current_date = datetime.datetime.now().strftime('%Y-%m-%d')
    conn = get_db_connection()
    if conn is None:
        raise HTTPException(status_code=500, detail="Database connection failed")

    cursor = conn.cursor()
    try:
        cursor.execute("SELECT * FROM attendance WHERE name = %s AND date = %s", (name, current_date))
        existing_entry = cursor.fetchone()

        if existing_entry:
            return {"message": f"{name} is already marked as present for {current_date}"}
        else:
            current_time = datetime.datetime.now().strftime('%H:%M:%S')
            cursor.execute("INSERT INTO attendance (name, time, date) VALUES (%s, %s, %s)", (name, current_time, current_date))
            conn.commit()
            return {"message": f"{name} marked as present for {current_date} at {current_time}"}
    except mysql.connector.Error as err:
        logging.error(f"Error inserting attendance record: {err}")
        raise HTTPException(status_code=500, detail="Failed to mark attendance")
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)