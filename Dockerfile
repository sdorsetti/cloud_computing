FROM python:3.8-slim-buster


RUN apt update
RUN apt-get --yes install libsndfile1


WORKDIR /app
ADD . /app

# Install dependencies
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt



# Expose port 
EXPOSE 8000

COPY . .


# Run the application:
CMD ["python", "app.py"]