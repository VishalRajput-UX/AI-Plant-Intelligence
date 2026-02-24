FROM python:3.11

WORKDIR /app

COPY . .

RUN pip install --upgrade pip
RUN pip install flask flask-cors tensorflow numpy pillow gunicorn

EXPOSE 10000

CMD ["gunicorn","plant_api:app","--bind","0.0.0.0:10000"]