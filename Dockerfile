# ใช้ Python 3.10
FROM python:3.10-slim

# ตั้ง working directory
WORKDIR /app

# คัดลอก requirements.txt ไปใน container
COPY requirements.txt .

# ติดตั้ง dependencies
RUN pip install --no-cache-dir -r requirements.txt

# คัดลอกไฟล์โปรเจกต์ทั้งหมดไปใน container
COPY . .

# รัน uvicorn ตอน container start
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
