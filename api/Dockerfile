FROM python:3.9

WORKDIR /env

COPY ./requirements.txt /env/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /env/requirements.txt

COPY ./app /env/app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
