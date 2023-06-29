# FROM node:19.0-alpine As production
FROM ubuntu:20.04

WORKDIR /app


# Update APT packages - Base Layer
RUN apt-get update \
    && apt-get upgrade --yes \
    && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends --yes wget curl



ARG DEBIAN_FRONTEND=noninteractive


# install postgres & nodejs
RUN apt-get update && apt-get install -y python3-django libpq-dev \
    && apt-get install -y python3-pip \
    && pip install psycopg2-binary \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# tensorflow
RUN pip install tensorflow \
tensorflow-estimator \
tensorflow-intel \
tensorflow-io-gcs-filesystem

# Clean up cache file - Service tech@appsmith.comlayer
RUN rm -rf \
    /root/.cache \
    /root/.npm \
    /root/.pip \
    /usr/local/share/doc \
    /usr/share/doc \
    /usr/share/man \
    /var/lib/apt/lists/* \
    /tmp/*


# Define volumes - Service Layer
VOLUME [ "/stacks" ]

COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir


COPY . .

RUN python3 manage.py migrate

# RUN python3 manage.py createsuperuser \
#         --noinput \
#         --username "superadmin" \
#         --email "admin@admin.com"
RUN echo "from app1.models import User; User.objects.create_superuser('superadmin', 'admin@example.com', 'pass')" | python3 manage.py shell

EXPOSE 8000

CMD ["python3" , "manage.py", "runserver", "0.0.0.0:8000"]
