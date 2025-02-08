from flask import Flask, render_template, request
import mysql.connector
from datetime import datetime

app = Flask(__name__)

db_config = {
    'user': 'krish',
    'password': 'krish@ml',
    'host': '51.20.6.164',
    'database': 'attendance'
}

@app.route('/')
def index():
    return render_template('index.html', selected_date='', no_data=False)

@app.route('/attendance', methods=['POST'])
def attendance():
    selected_date = request.form.get('selected_date')
    selected_date_obj = datetime.strptime(selected_date, '%Y-%m-%d')
    formatted_date = selected_date_obj.strftime('%Y-%m-%d')


    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()

    cursor.execute("SELECT name, time FROM attendance WHERE date = %s", (formatted_date,))
    attendance_data = cursor.fetchall()

    conn.close()

    if not attendance_data:
        return render_template('index.html', selected_date=selected_date, no_data=True)
    
    return render_template('index.html', selected_date=selected_date, attendance_data=attendance_data)


if __name__ == '__main__':
    app.run(debug=True)
