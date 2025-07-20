# ใช้ Python 3.10 slim image
FROM python:3.10-slim

# ติดตั้ง lib ที่จำเป็นสำหรับ tflite-runtime
RUN apt-get update && apt-get install -y \
    libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*

# ตั้ง working directory
WORKDIR /app

# คัดลอก requirements.txt
COPY requirements.txt .

# ติดตั้ง dependencies
RUN pip install --no-cache-dir -r requirements.txt

# คัดลอกไฟล์โปรเจกต์ทั้งหมด
COPY . .

# เปิดพอร์ต
EXPOSE 8000

# รัน uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
