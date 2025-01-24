FROM python:3.11

EXPOSE 8000

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /facial-recognition-backend

COPY ./requirements.txt /requirements.txt

RUN pip install --no-cache-dir --upgrade -r /requirements.txt

COPY ./src /facial-recognition-backend/src

CMD ["fastapi", "run", "src/main.py", "--port", "8000"]