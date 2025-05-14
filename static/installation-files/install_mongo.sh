# 1. Import the MongoDB public GPG key
curl -fsSL https://pgp.mongodb.com/server-6.0.asc | \
  sudo gpg -o /usr/share/keyrings/mongodb-server-6.0.gpg --dearmor

# 2. Create a list file for MongoDB
echo "deb [ signed-by=/usr/share/keyrings/mongodb-server-6.0.gpg ] https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/6.0 multiverse" | \
  sudo tee /etc/apt/sources.list.d/mongodb-org-6.0.list

# 3. Reload the local package database
sudo apt update

# 4. Install MongoDB
sudo apt install -y mongodb-org

# 5. Start MongoDB
sudo systemctl start mongod

# 6. Enable it to start on boot
sudo systemctl enable mongod

# 7. Verify
mongod --version
