# 1. Download and import the public key
curl -fsSL https://pgp.mongodb.com/server-6.0.asc | \
  sudo gpg -o /usr/share/keyrings/mongodb-server-6.0.gpg --dearmor

# 2. Add the MongoDB repo to your sources list
echo "deb [ signed-by=/usr/share/keyrings/mongodb-server-6.0.gpg ] https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/6.0 multiverse" | \
  sudo tee /etc/apt/sources.list.d/mongodb-org-6.0.list

# 3. Update packages
sudo apt update

# 4. Install mongosh only (you don't need full server here)
sudo apt install -y mongodb-mongosh