FROM python:3.14
COPY ./requirements.txt /requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt
COPY ./runner.py /runner.py
CMD ["python", "runner.py"]
