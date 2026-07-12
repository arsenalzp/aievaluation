FROM python:3.14
COPY ./requirements.txt /requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt
COPY ./llm /llm
COPY ./runner-agent.py /runner-agent.py
CMD ["python", "runner-agent.py"]
