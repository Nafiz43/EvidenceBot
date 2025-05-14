FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

##############################################################################
# 1) Install system packages
##############################################################################
RUN apt-get update && apt-get install -y \
    gnupg \
    curl \
    systemd \
    systemd-sysv \
    build-essential \
    git \
    python3.10 \
    python3-pip \
    pkg-config \
    libssl-dev \
    nano \
 && rm -rf /var/lib/apt/lists/*

##############################################################################
# 2) Install MongoDB
##############################################################################
RUN curl -fsSL https://www.mongodb.org/static/pgp/server-6.0.asc | apt-key add - && \
    echo "deb [ arch=arm64,amd64 ] https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/6.0 multiverse" \
    | tee /etc/apt/sources.list.d/mongodb-org-6.0.list && \
    apt-get update && apt-get install -y mongodb-org && \
    rm -rf /var/lib/apt/lists/*

##############################################################################
# 3) Install Rust 1.85
##############################################################################
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --default-toolchain 1.85.0
ENV PATH="/root/.cargo/bin:${PATH}"

##############################################################################
# 4) Clone & build OSS-scraper (No hardcoded tokens)
##############################################################################
WORKDIR /root

ARG GITHUB_PAT
RUN git clone https://$GITHUB_USERNAME:$GITHUB_PAT@github.com/priyalsoni15/OSS-scraper.git

WORKDIR /root/OSS-scraper
RUN cargo build --release

##############################################################################
# 5) Clone pex-forecaster (No hardcoded Git credentials)
##############################################################################
WORKDIR /root
RUN git clone https://$GITHUB_USERNAME:$GITHUB_PAT@github.com/arjashok/pex-forecaster.git

##############################################################################
# 6) Clone & set up ReACT-API
##############################################################################
WORKDIR /root
RUN git clone https://github.com/Nafiz43/ReACT-API.git

WORKDIR /root/ReACT-API
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install virtualenv && \
    python3 -m virtualenv venv

RUN if [ -f "requirements.txt" ]; then \
      ./venv/bin/pip install -r requirements.txt; \
    else \
      echo "No requirements.txt found in ReACT-API."; \
    fi

RUN if [ -f "setup.py" ] || [ -f "pyproject.toml" ]; then \
      ./venv/bin/pip install -e .; \
    else \
      echo "No setup.py/pyproject.toml in ReACT-API; skipping -e ."; \
    fi

##############################################################################
# 7) Clone & set up OSS-fetch-github-data (No hardcoded GitHub tokens)
##############################################################################
WORKDIR /root
RUN git clone https://github.com/priyalsoni15/OSS-fetch-github-data.git

WORKDIR /root/OSS-fetch-github-data

# Load environment variables from runtime instead of writing them inside the Docker image
ARG GITHUB_TOKEN_1
ARG GITHUB_TOKEN_2
ARG GITHUB_TOKEN_3
ARG GITHUB_TOKEN_4
ARG MONGODB_USER
ARG MONGODB_PASSWORD

ENV MONGODB_URI="mongodb://${MONGODB_USER}:${MONGODB_PASSWORD}@localhost:27017/decal-db?authSource=admin"

RUN python3 -m virtualenv venv && \
    ./venv/bin/pip install -r requirements.txt

##############################################################################
# 8) Install pex-forecaster in the same virtual environment
##############################################################################
RUN /bin/bash -c "source venv/bin/activate && \
    pip install -r /root/pex-forecaster/requirements.txt && \
    pip install -e /root/pex-forecaster"

##############################################################################
# 9) Create Gunicorn service
##############################################################################
RUN echo "[Unit]\n\
Description=Gunicorn instance to serve Flask app\n\
After=network.target\n\
\n\
[Service]\n\
User=root\n\
Group=root\n\
WorkingDirectory=/root/OSS-fetch-github-data\n\
Environment=\"PATH=/root/OSS-fetch-github-data/venv/bin\"\n\
ExecStart=/root/OSS-fetch-github-data/venv/bin/gunicorn -w 10 -b 0.0.0.0:5000 run:app\n\
Restart=on-failure\n\
RestartSec=2\n\
\n\
[Install]\n\
WantedBy=multi-user.target\n" > /etc/systemd/system/gunicorn.service

RUN systemctl enable mongod && systemctl enable gunicorn

##############################################################################
# 10) Secure MongoDB User Setup
##############################################################################
RUN mkdir -p /tmp/dbdata && \
    mongod --dbpath /tmp/dbdata --bind_ip_all --fork --logpath /var/log/mongod.log && \
    sleep 5 && \
    mongosh --eval "use decal-db; db.createUser({ user: '${MONGODB_USER}', pwd: '${MONGODB_PASSWORD}', roles: [ { role: 'readWrite', db: 'decal-db' } ]}); db.shutdownServer();" \
 || echo "Ignoring possible shutdown error"

##############################################################################
# 11) Copy mongo_exports folder
##############################################################################
COPY mongo_exports /root/mongo_exports

##############################################################################
# 12) Secure test.sh script
##############################################################################
RUN echo "#!/bin/bash\n\
DATABASE=\"decal-db\"\n\
USERNAME=\"${MONGODB_USER}\"\n\
PASSWORD=\"${MONGODB_PASSWORD}\"\n\
AUTH_DB=\"decal-db\"\n\
DIRECTORY=\"/root/mongo_exports\"\n\
\n\
# Build connection string with authentication\n\
CONN_STRING=\"mongodb://\$USERNAME:\$PASSWORD@localhost:27017/\$DATABASE?authSource=\$AUTH_DB&authMechanism=SCRAM-SHA-256\"\n\
\n\
cd \"\$DIRECTORY\" || { echo \"Directory \$DIRECTORY not found!\"; exit 1; }\n\
\n\
for file in *.json; do\n\
    collection_name=\$(basename \"\$file\" .json)\n\
    echo \"Importing \$file into collection \$collection_name...\"\n\
    mongoimport --uri \"\$CONN_STRING\" --collection \"\$collection_name\" --file \"\$file\" --jsonArray\n\
done\n\
\n\
echo \"All collections imported successfully!\" \n" > /root/test.sh

RUN chmod +x /root/test.sh

##############################################################################
# 13) Run test.sh and shut down MongoDB
##############################################################################
RUN mongod --auth --dbpath /tmp/dbdata --bind_ip_all --fork --logpath /var/log/mongodimport.log && \
    sleep 5 && \
    bash /root/test.sh && \
    mongosh --eval "db.shutdownServer()" || echo "Ignoring possible import or shutdown error"

##############################################################################
# 14) Expose port 5000 and use systemd
##############################################################################
EXPOSE 5000
STOPSIGNAL SIGRTMIN+3
CMD ["/sbin/init"]